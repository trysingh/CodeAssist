"""Optional web context for chat: decide -> rewrite query -> SearXNG -> scrape top pages.

Everything fails soft: any problem yields an empty WebContext with a short note, so the
chat falls back to repo-only (or plain LLM) instead of erroring.

Privacy: only a rewritten, code-free search query is sent to SearXNG, never repo content.
Safety:  scraped text is wrapped in <web_source> tags and marked untrusted in the prompt.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
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

_REWRITE = ("Rewrite the user's message as ONE web search query (max 12 words). Keep only the general topic and "
            "public library/product names. Never include file paths, function or variable names, secrets or "
            "private project details. Reply with the query only.\n\nMessage: ")

NoulFn = Callable[..., Awaitable[Optional[float]]]


# ------------------------------------------------------------------ small helpers
def split_web_prefix(query: str, mode: str = "auto") -> Tuple[str, str]:
    """'/web foo' -> ('foo', 'on'); anything else is returned unchanged."""
    m = _WEB_PREFIX.match(query)
    return (query[m.end():].strip(), "on") if m else (query, (mode or "auto").lower())


async def decide_web(query: str, mode: str = "auto", *, repo_scoped: bool, noul_fn: NoulFn = decide_noul,
                     model: Optional[str] = None) -> Tuple[bool, str]:
    """(use_web, why). Web is opt-in inside a repo, and the default outside one."""
    mode = (mode or "auto").lower()
    if not settings.web_enabled or mode == "off":
        return False, "off"
    if mode == "on":
        return True, "forced"
    if _URL.search(query) or _EXPLICIT.search(query):
        return True, "rules"
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
    return str((await llm.ainvoke(_REWRITE + text[:1500])).content)


async def plan_query(question: str, *, repo_scoped: bool,
                     llm_fn: Callable[[str], Awaitable[str]] = _llm_rewrite) -> str:
    q = " ".join(_URL.sub("", question).split())
    if not q:
        return ""
    fallback = q[:200]
    if len(q.split()) <= 8 and not repo_scoped:
        return fallback                                   # already a search-sized query
    try:
        out = (await asyncio.wait_for(llm_fn(q), 8)).strip()
        return out.splitlines()[0].strip(" \"'`")[:200] or fallback if out else fallback
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

    def footer(self) -> str:
        used = [s for s in self.sources if s.text or s.snippet]
        if not used:
            return f"\n\n---\n_Web: {self.note}; answered without it._" if self.note else ""
        def link(s: WebSource) -> str:
            title = re.sub(r"[\[\]]", "", s.title or s.url).strip()[:90] or s.url
            return f"{s.n}. [{title}]({s.url.replace(' ', '%20').replace(')', '%29')})"
        return "\n\n---\n**Web sources**\n" + "\n".join(link(s) for s in used)


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
                 scrape_fn: Optional[Callable[[str, bool], Awaitable[Any]]] = None):
        self._transport = transport            # tests inject httpx.MockTransport
        self._scrape = scrape_fn or _default_scrape

    @staticmethod
    def _log(msg: str) -> None:
        sys.stdout.write(f"[WEB] {msg}\n")

    async def search(self, query: str) -> List[dict]:
        async with httpx.AsyncClient(timeout=settings.web_search_timeout_s, transport=self._transport) as c:
            r = await c.get(settings.searxng_url, params={"q": query, "format": "json"})
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

    async def gather(self, question: str, *, repo_scoped: bool) -> WebContext:
        t0 = time.monotonic()
        urls = list(dict.fromkeys(_URL.findall(question)))[: settings.web_max_pages]
        query = ""
        try:
            if urls:
                hits = [{"url": u, "title": u, "snippet": ""} for u in urls]
            else:
                query = await plan_query(question, repo_scoped=repo_scoped)
                if not query:
                    return WebContext(note="nothing to search for")
                hits = await asyncio.wait_for(self.search(query), settings.web_search_timeout_s + 2)
        except Exception as e:  # noqa: BLE001
            self._log(f"search failed: {type(e).__name__}: {e}")
            return WebContext(query=query, note=f"search unavailable ({type(e).__name__})")
        if not hits:
            return WebContext(query=query, note="search returned no results")

        sources = [WebSource(i + 1, h["url"], h["title"], h["snippet"])
                   for i, h in enumerate(hits[: settings.web_max_pages])]
        tasks = [asyncio.ensure_future(self._fill(s)) for s in sources]
        _, pending = await asyncio.wait(tasks, timeout=max(1.0, settings.web_timeout_s - (time.monotonic() - t0)))
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        ctx = WebContext(text=_render(sources), sources=sources, query=query)
        if not ctx.text:
            ctx.note = "no readable pages"
        self._log(f"query={query or '(urls)'!r}: {sum(1 for s in sources if s.text)}/{len(sources)} pages read "
                  f"in {time.monotonic() - t0:.1f}s")
        return ctx
