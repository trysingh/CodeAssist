"""Optional web context for chat: decide -> rewrite query -> SearXNG -> scrape top pages.

Everything fails soft: any problem yields an empty WebContext with a short note, so the
chat falls back to repo-only (or plain LLM) instead of erroring.

Follow-ups: every message gets its own fresh search. "do a new search", "check again" or
"what about tomorrow?" carry no topic of their own, so the query is built from the recent
conversation (plan_query), stamped with today's date for time-bound questions, and varied
when it would repeat the previous query (avoid=...).

Privacy: only a rewritten, code-free search query is sent to SearXNG, never repo content.
Safety:  scraped text is wrapped in <web_source> tags and marked untrusted in the prompt.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import httpx

from config.settings import settings
from services.decision import decide_noul

_WEB_PREFIX = re.compile(r"^\s*/web\b[:\s]*", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_EXPLICIT = re.compile(r"\b(search|google|look\s?up|browse)\b.{0,25}\b(web|internet|online)\b|\bon the (web|internet)\b",
                       re.IGNORECASE)
_FRESH = re.compile(r"\b(latest|newest|current(?:ly)?|recent(?:ly)?|today|this (?:week|month|year)|news|"
                    r"release notes?|changelog|what'?s new|deprecated|breaking changes?|pricing|20\d\d)\b", re.IGNORECASE)
_SKIP_URL = re.compile(r"\.(pdf|zip|gz|exe|dmg|mp4|mp3|png|jpe?g|gif|webp)(\?|$)", re.IGNORECASE)

_Q = ("Does answering this need up-to-date or external information from the internet "
      "(documentation, news, versions, other projects) that would not be in the user's own code repository?")
_TRUE = "The answer needs current or external information from the internet."
_FALSE = "The answer can come from the repository, the conversation, or general knowledge."

_REWRITE = ("Rewrite the user's latest message as ONE web search query (max 14 words). Keep only the general topic and "
            "public library/product/team/person names. Resolve words such as 'them', 'it', 'again' or 'do a new search' "
            "from the recent conversation into the concrete topic they refer to. For time-bound questions (today, "
            "tomorrow, tonight, this week, schedule, score) include the actual date given after 'Today'. Never include "
            "file paths, function or variable names, secrets or private project details. Reply with the query only.\n\n")

# "do a new search" / "search again": the message names no topic, so rules (not a model) must catch it.
_RETRY_SEARCH = re.compile(
    r"\b(?:new|fresh|another|different)\s+(?:web\s+|live\s+|online\s+)?(?:search|look\s?up)\b|"
    r"\b(?:search|look(?:\s?up)?|google|browse)\w*\b.{0,30}\b(?:again|once more|afresh|anew)\b|"
    r"\bre-?search\b", re.IGNORECASE)
# "check again" / "try again": a retry only when the previous turn used the web (or there is no repo to talk about)
_RETRY_SOFT = re.compile(
    r"\b(?:check|try|verify|confirm|find|fetch|retry|re-?try|re-?check|refresh)\w*\b.{0,30}\b(?:again|once more|afresh|anew)\b|"
    r"\b(?:retry|re-?try|refresh)\s+(?:the\s+)?(?:search|results?|web|that|it|this)\b", re.IGNORECASE)
_DEICTIC = re.compile(r"\b(?:them|they|their|it|its|that|this|those|these|there|he|she|him|her|same|also|too|instead)\b|"
                      r"\b(?:what|how) about\b|^\s*and\b|\b(?:tomorrow|yesterday|next|later|tonight)\b", re.IGNORECASE)
_TIMELY = re.compile(r"\b(?:today|tonight|tomorrow|yesterday|right now|live|schedule[ds]?|fixtures?|scores?|"
                     r"this (?:week|month)|upcoming)\b", re.IGNORECASE)
_HAS_DATE = re.compile(r"\b20\d\d\b|\b(?:jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)
_DAY_RANGE = re.compile(r"\b(?:today|tonight|live|right now|this (?:morning|evening))\b", re.IGNORECASE)
_WEEK_RANGE = re.compile(r"\b(?:tomorrow|yesterday|this week|upcoming)\b", re.IGNORECASE)

NoulFn = Callable[..., Awaitable[Optional[float]]]


# ------------------------------------------------------------------ small helpers
def now_local() -> datetime:
    return datetime.now().astimezone()


def today_str(now: Optional[datetime] = None) -> str:
    n = now or now_local()
    return f"{n:%A}, {n.day} {n:%B %Y}"


def now_str(now: Optional[datetime] = None) -> str:
    n = now or now_local()
    return f"{today_str(n)}, {n:%H:%M} {n.tzname() or ''}".strip()


def retry_kind(query: str) -> str:
    """'search' = explicit ask to search again, 'soft' = 'check again' style, '' = neither (short messages only)."""
    q = " ".join((query or "").split())
    if not q or len(q.split()) > 14:
        return ""
    if _RETRY_SEARCH.search(q):
        return "search"
    return "soft" if _RETRY_SOFT.search(q) else ""


def time_range_for(text: str) -> Optional[str]:
    """SearXNG time_range for strongly time-bound questions; evergreen ones get no filter."""
    if _DAY_RANGE.search(text):
        return "day"
    return "week" if _WEEK_RANGE.search(text) else None


def split_web_prefix(query: str, mode: str = "auto") -> Tuple[str, str]:
    """'/web foo' -> ('foo', 'on'); anything else is returned unchanged."""
    m = _WEB_PREFIX.match(query)
    return (query[m.end():].strip(), "on") if m else (query, (mode or "auto").lower())


async def decide_web(query: str, mode: str = "auto", *, repo_scoped: bool, noul_fn: NoulFn = decide_noul,
                     model: Optional[str] = None, prev_web: bool = False) -> Tuple[bool, str]:
    """(use_web, why). Web is opt-in inside a repo, and the default outside one.

    prev_web: the previous turn of this thread used the web. It makes "check again" and short
    follow-ups ("what about tomorrow?") search again instead of being answered from the last answer."""
    mode = (mode or "auto").lower()
    if not settings.web_enabled or mode == "off":
        return False, "off"
    if mode == "on":
        return True, "forced"
    if _URL.search(query) or _EXPLICIT.search(query):
        return True, "rules"
    kind = retry_kind(query)
    if kind == "search" or (kind == "soft" and (prev_web or not repo_scoped)):
        return True, "retry"          # "do a new search": no topic in the message, so only rules can see it
    if prev_web and len(query.split()) <= 14 and _DEICTIC.search(query):
        return True, "follow-up"      # short follow-up to an answer that used the web
    if not repo_scoped and _FRESH.search(query):
        return True, "rules"
    model = settings.router_model if model is None else model
    if model:
        p = await noul_fn(model, query[:600], _Q, true=_TRUE, false=_FALSE, phase="WEB")
        if p is not None:
            return p >= (settings.web_auto_threshold if repo_scoped else 0.4), "model"
    return (not repo_scoped), "default"


async def _llm_rewrite(text: str) -> str:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0, num_ctx=2048)
    return str((await llm.ainvoke(_REWRITE + text[:3000])).content)


def _clean_history(history: Optional[List[Tuple[str, str]]], limit: int = 6, chars: int = 300) -> List[Tuple[str, str]]:
    out = []
    for role, text in list(history or [])[-limit:]:
        t = " ".join(_URL.sub("", str(text or "")).split())
        if t:
            out.append(("User" if str(role).lower().startswith(("u", "h")) else "Assistant", t[:chars]))
    return out


def _anchor(history: List[Tuple[str, str]], current: str) -> str:
    """Latest earlier user message that carries a topic (not itself a retry or a bare reference)."""
    for role, text in reversed(history):
        if role == "User" and len(text.split()) >= 3 and not retry_kind(text) and text.lower() != current.lower():
            return text
    return ""


def _date_for(text: str, n: datetime) -> str:
    d = n + timedelta(days=1) if re.search(r"\btomorrow\b", text, re.I) else \
        n - timedelta(days=1) if re.search(r"\byesterday\b", text, re.I) else n
    return f"{d.day} {d:%B %Y}"


def _freshen(query: str, source: str, now: Optional[datetime], avoid: Optional[List[str]]) -> str:
    """Stamp time-bound queries with the date, and never repeat a query that was just run."""
    n = now or now_local()
    if query and _TIMELY.search(f"{query} {source}") and not _HAS_DATE.search(query):
        query = f"{query} {_date_for(f'{query} {source}', n)}"
    if avoid and query.strip().lower() in {a.strip().lower() for a in avoid}:
        query += " latest update"                      # same words -> same results: nudge the engine
    return query[:200]


def _plan_payload(q: str, history: List[Tuple[str, str]], now: Optional[datetime],
                  avoid: Optional[List[str]], retry: bool) -> str:
    lines = [f"Today: {today_str(now)}"]
    if history:
        lines.append("Recent conversation, oldest first:")
        lines += [f"{r}: {t}" for r, t in history]
    if retry:
        lines.append("The user wants a NEW search on the same topic as before: use different keywords"
                     + (f" than: {'; '.join(avoid)}" if avoid else "") + ".")
    lines.append(f"Message: {q[:400]}")
    return "\n".join(lines)


async def plan_query(question: str, *, repo_scoped: bool, history: Optional[List[Tuple[str, str]]] = None,
                     now: Optional[datetime] = None, avoid: Optional[List[str]] = None,
                     llm_fn: Callable[[str], Awaitable[str]] = _llm_rewrite) -> str:
    """Search query for `question`. history = [(role, text)] of the turns before it, oldest first.

    "do a new search and try checking again" names no topic, so the query comes from the conversation:
    the model resolves it (or, if the model is down, the last real question is reused). Time-bound
    questions are stamped with today's date; `avoid` lists queries already run (they get varied)."""
    q = " ".join(_URL.sub("", question).split())
    if not q:
        return ""
    hist = _clean_history(history)
    retry = bool(retry_kind(q))
    ctx = bool(hist) and (retry or bool(_DEICTIC.search(q)))       # a short query can still stand on its own
    anchor = _anchor(hist, q) if ctx else ""
    base = anchor if (anchor and retry) else f"{anchor} {q}".strip()
    fallback = _freshen(base[:200], q, now, avoid)
    if len(q.split()) <= 8 and not repo_scoped and not ctx:
        return fallback                                   # already a search-sized query
    try:
        payload = _plan_payload(q, hist if ctx else [], now, avoid if retry else None, retry)
        out = (await asyncio.wait_for(llm_fn(payload), 8)).strip()
        planned = out.splitlines()[0].strip(" \"'`")[:200] if out else ""
        return _freshen(planned, q, now, avoid) if planned else fallback
    except Exception:  # noqa: BLE001
        return fallback


# ------------------------------------------------------------------ page text budgeting
def _cut_tables(md: str, budget: int) -> Tuple[str, bool]:
    if len(md) <= budget:
        return md, False
    kept, used = [], 0
    for line in md.split("\n"):                           # whole rows only: a half row reads as data
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept).rstrip(), True


def _cut_prose(text: str, budget: int) -> Tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    window = text[:budget]
    for sep in ("\n\n", ". ", "\n"):
        cut = window.rfind(sep)
        if cut > budget * 0.66:
            return window[:cut].rstrip(), True
    return window.rstrip(), True


def compose_page(res: Any, budget: int) -> Tuple[str, bool]:
    """ScrapeResult -> (text within budget, was_truncated). Tables get up to half the budget."""
    prose = (getattr(res, "content", "") or "").strip()
    tables = (getattr(res, "tables_md", "") or "").strip() if getattr(res, "table_count", 0) else ""
    t_out, t_cut = _cut_tables(tables, budget // 2) if tables else ("", False)
    p_out, p_cut = _cut_prose(prose, budget - len(t_out))
    return "\n\n".join(x for x in (t_out, p_out) if x), (t_cut or p_cut)


# ------------------------------------------------------------------ results
@dataclass
class WebSource:
    n: int
    url: str
    title: str
    snippet: str = ""
    text: str = ""
    truncated: bool = False
    note: str = ""


@dataclass
class WebContext:
    text: str = ""
    sources: List[WebSource] = field(default_factory=list)
    query: str = ""
    note: str = ""
    date: str = ""

    def footer(self) -> str:
        used = [s for s in self.sources if s.text or s.snippet]
        if not used:
            return f"\n\n---\n_Web: {self.note}; answered without it._" if self.note else ""
        def link(s: WebSource) -> str:
            title = re.sub(r"[\[\]]", "", s.title or s.url).strip()[:90] or s.url
            return f"{s.n}. [{title}]({s.url.replace(' ', '%20').replace(')', '%29')})"
        searched = ""
        if self.query:                                   # shows that THIS message ran its own search
            shown = re.sub(r"[*`_]", "", self.query)
            searched = f"\n\n*Searched: {shown}" + (f" \u00b7 {self.date}" if self.date else "") + "*"
        return "\n\n---\n**Web sources**\n" + "\n".join(link(s) for s in used) + searched


def _safe(text: str) -> str:
    return text.replace("<web_source", "‹web_source").replace("</web_source", "‹/web_source")


def _render(sources: List[WebSource]) -> str:
    blocks = []
    for s in sources:
        body = s.text
        if body and s.truncated:
            body += "\n[TRUNCATED to fit the context budget; the page is longer than shown.]"
        if not body and s.snippet:
            body = f"(search snippet only; page could not be read)\n{s.snippet}"
        if body:
            blocks.append(f'<web_source n="{s.n}" url="{s.url}" title="{_safe(s.title)[:120]}">\n{_safe(body)}\n</web_source>')
    return "\n\n".join(blocks)


async def _default_scrape(url: str, allow_playwright: bool) -> Any:
    from services.scraper.engine import scrape_fast        # heavy deps: imported only when web is used
    res, finish = await scrape_fast(url, min_words=settings.web_min_words, allow_playwright=allow_playwright)
    if finish is not None and allow_playwright:
        res = await finish()
    return res


# ------------------------------------------------------------------ service
class WebSearchService:
    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None,
                 scrape_fn: Optional[Callable[[str, bool], Awaitable[Any]]] = None,
                 plan_fn: Optional[Callable[..., Awaitable[str]]] = None):
        self._transport = transport            # tests inject httpx.MockTransport
        self._scrape = scrape_fn or _default_scrape
        self._plan = plan_fn or plan_query

    @staticmethod
    def _log(msg: str) -> None:
        sys.stdout.write(f"[WEB] {msg}\n")

    async def search(self, query: str, time_range: Optional[str] = None) -> List[dict]:
        params = {"q": query, "format": "json"}
        if time_range:
            params["time_range"] = time_range            # day | week: SearXNG passes it to engines that support it
        async with httpx.AsyncClient(timeout=settings.web_search_timeout_s, transport=self._transport) as c:
            r = await c.get(settings.searxng_url, params=params)
            r.raise_for_status()
            data = r.json()
        out, seen = [], set()
        for it in data.get("results", []):
            u = (it.get("url") or "").strip()
            if not u.startswith(("http://", "https://")) or u in seen or _SKIP_URL.search(u):
                continue
            seen.add(u)
            out.append({"url": u, "title": it.get("title") or u, "snippet": (it.get("content") or "")[:500]})
            if len(out) >= settings.web_max_results:
                break
        return out

    async def _fill(self, s: WebSource) -> None:
        try:
            res = await self._scrape(s.url, settings.web_allow_playwright)
            text, cut = compose_page(res, settings.web_page_chars)
            if getattr(res, "word_count", 0) < 40 or not text:
                s.note = "page had too little readable text"
                return
            s.text, s.truncated = text, cut
            s.title = getattr(res, "title", None) or s.title
        except Exception as e:  # noqa: BLE001
            s.note = f"{type(e).__name__}"

    async def gather(self, question: str, *, repo_scoped: bool, history: Optional[List[Tuple[str, str]]] = None,
                     avoid: Optional[List[str]] = None, now: Optional[datetime] = None) -> WebContext:
        t0 = time.monotonic()
        stamp = today_str(now)
        urls = list(dict.fromkeys(_URL.findall(question)))[: settings.web_max_pages]
        query = ""
        try:
            if urls:
                hits = [{"url": u, "title": u, "snippet": ""} for u in urls]
            else:
                query = await self._plan(question, repo_scoped=repo_scoped, history=history, now=now, avoid=avoid)
                if not query:
                    return WebContext(note="nothing to search for", date=stamp)
                rng = time_range_for(f"{question} {query}")
                hits = await asyncio.wait_for(self.search(query, rng), settings.web_search_timeout_s + 2)
                if not hits and rng:                     # nothing that fresh: retry once without the filter
                    hits = await asyncio.wait_for(self.search(query), settings.web_search_timeout_s + 2)
        except Exception as e:  # noqa: BLE001
            self._log(f"search failed: {type(e).__name__}: {e}")
            return WebContext(query=query, note=f"search unavailable ({type(e).__name__})", date=stamp)
        if not hits:
            return WebContext(query=query, note="search returned no results", date=stamp)

        sources = [WebSource(i + 1, h["url"], h["title"], h["snippet"])
                   for i, h in enumerate(hits[: settings.web_max_pages])]
        tasks = [asyncio.ensure_future(self._fill(s)) for s in sources]
        _, pending = await asyncio.wait(tasks, timeout=max(1.0, settings.web_timeout_s - (time.monotonic() - t0)))
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        ctx = WebContext(text=_render(sources), sources=sources, query=query, date=stamp)
        if not ctx.text:
            ctx.note = "no readable pages"
        self._log(f"query={query or '(urls)'!r}: {sum(1 for s in sources if s.text)}/{len(sources)} pages read "
                  f"in {time.monotonic() - t0:.1f}s")
        return ctx
