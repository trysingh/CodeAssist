"""Web page quality assessment, encoding repair, and the recovery pass (re-decode / screenshot OCR / restricted -> snippet).
No network, no browser, no model: scrape / plan / search / visual are injected fakes. Runs without httpx / langchain (tests/stubbed_imports.py)."""
import asyncio
import types
from types import SimpleNamespace

import pytest

from stubbed_imports import stubbed_imports
from services.scraper.quality import (GARBLED, IMAGE_TEXT, OFF_TOPIC, OK, RESTRICTED, THIN, assess, try_repair_encoding)

PROSE = ("Asyncio timeouts are handled with asyncio.wait_for or the asyncio.timeout context manager in Python. "
         "When the deadline passes the awaited task is cancelled and a TimeoutError is raised to the caller. ") * 8
QUERY = "python asyncio timeout"


def run(coro):
    return asyncio.run(coro)


# ============================================================================= assess()
def test_ok_page():
    q = assess(PROSE, query=QUERY)
    assert q.status == OK and q.ok and q.metrics["words"] > 100


def test_garbled_variants():
    assert assess("Hello ���� " + PROSE).status == GARBLED                       # replacement characters
    assert assess("data \x00\x01\x02\x03 " + PROSE).status == GARBLED                                # control characters
    moji = "CafÃ© menÃ¼ donâ€™t â€œquoteâ€\u009d " * 12
    q = assess(moji)
    assert q.status == GARBLED and "mojibake" in q.reason
    assert assess("@@## $$%% ^^&& ~~|| " * 20 + "▒▓█" * 20).status == GARBLED          # symbol soup, few words
    assert assess("a1 b2 c3 d4 e5 f6 g7 h8 i9 j0 " * 3 + "12 34 56 78 90 11 22 33 44 55 66 77").status == GARBLED   # few real words
    assert assess(PROSE + " café naïve — “quoted”").status == OK              # real accents and quotes are fine


def test_thin_and_cjk():
    assert assess("Just a few words here.").status == THIN
    nav = "\n".join(["Home", "About", "Docs", "Blog", "Login"] * 20)
    q = assess(nav)
    assert q.status == THIN and q.metrics["junk_ratio"] > 0.85
    assert assess("这是一个关于异步编程超时处理的详细说明文章，" * 30).status == OK                    # no spaces between words


def test_image_text_needs_html_signals():
    few = "Annual report 2025 summary"
    assert assess(few).status == THIN                                                                # no html: just thin
    assert assess(few, html_head="<canvas id=c></canvas>").status == IMAGE_TEXT
    assert assess(few, html_head='<embed src="/r.pdf" type="application/pdf">').status == IMAGE_TEXT
    assert assess(few, html_head="<img src=a>" * 6).status == IMAGE_TEXT
    assert assess(few, html_head="<img src=a>" * 2).status == THIN                                   # two icons are not an image page
    assert assess(PROSE, html_head="<canvas></canvas>" + "<img>" * 9).status == OK                  # plenty of text: images irrelevant


@pytest.mark.parametrize("text", [
    "Please verify you are human before continuing. Checking your browser.",
    "Please log in to continue reading this page.",
    "Subscribe to continue reading. Already a subscriber? Sign in.",
    "Complete the CAPTCHA below to access the site " + "filler " * 20,
    "Access denied. You do not have permission.",
])
def test_restricted_markers(text):
    q = assess(text)
    assert q.status == RESTRICTED, q


def test_restricted_status_codes_and_long_pages():
    assert assess(PROSE, status_code=402).status == RESTRICTED
    assert assess(PROSE, status_code=401).status == RESTRICTED
    assert assess(PROSE, status_code=200).status == OK
    assert assess("<title>Just a moment...</title>", html_head="<title>Just a moment...</title>").status == RESTRICTED
    # long article that merely mentions a paywall far from the top is not a wall
    long = PROSE * 3 + " Some sites say subscribe to continue reading."
    assert assess(long).status == OK


def test_off_topic_is_conservative():
    cooking = ("Preheat the oven and whisk the eggs with sugar until pale. Fold in flour and bake the cake for thirty minutes. " * 10)
    assert assess(cooking, query=QUERY).status == OFF_TOPIC
    assert assess(cooking, query="python").status == OK                    # < 3 keywords: no judgement
    assert assess(cooking, query=None).status == OK
    assert assess(cooking + " python", query=QUERY).status == OK            # one keyword hit is enough
    assert assess(PROSE, query="the latest new guide").status == OK        # only stop words


def test_try_repair_encoding():
    bad = "CafÃ© â€œquotedâ€\u009d naÃ¯ve donâ€™t Ã©tÃ© " * 3
    fixed = try_repair_encoding(bad)
    assert fixed.startswith("Café “quoted” naïve don’t été")
    assert "Ã" not in fixed and "â€" not in fixed
    assert try_repair_encoding("café déjà vu — fine") == "café déjà vu — fine"     # clean text untouched
    assert try_repair_encoding("") == ""


# ============================================================================= WebSearchService.gather
@pytest.fixture(scope="module")
def mods():
    with stubbed_imports():
        import importlib
        yield types.SimpleNamespace(ws=importlib.import_module("services.web_search_service"),
                                    settings=importlib.import_module("config.settings").settings)


def page(content, title="T", words=None, **extra):
    return SimpleNamespace(content=content, tables_md="", table_count=0, title=title,
                           word_count=words if words is not None else len(content.split()), **extra)


HITS = [{"url": "https://ok.example/a", "title": "OK page", "snippet": "ok snippet"},
        {"url": "https://js.example/b", "title": "JS page", "snippet": "js snippet"},
        {"url": "https://wall.example/c", "title": "Wall page", "snippet": "wall snippet"}]
PAGES = {"ok.example": page(PROSE, title="Asyncio guide"),
         "js.example": page("Loading...", title="JS page", html_head=None, images=["i"] * 7),                     # image-based: thin text + 7 images
         "wall.example": page("Please log in to continue reading this article.", title="Wall page")}
OCR = "Asyncio timeouts: use asyncio.timeout(5). Summary: the context manager cancels the task after the deadline."


class Harness:
    def __init__(self, mods, monkeypatch, *, visual_result=OCR, mode="auto", vision=True, hits=HITS, pages=PAGES, max_pages=1):
        st = mods.settings
        for k, v in {"web_visual_fallback": mode, "web_visual_max_pages": max_pages, "web_offtopic_filter": True, "web_max_pages": 3,
                     "web_timeout_s": 5.0, "web_visual_timeout_s": 2.0}.items():
            monkeypatch.setattr(st, k, v)
        monkeypatch.setattr(mods.ws, "vision_available", lambda: vision)
        self.visual_calls = []

        async def scrape(url, allow_pw):
            return pages[url.split("/")[2]]

        async def plan(question, **kw):
            return QUERY

        async def visual(url, query):
            self.visual_calls.append((url, query))
            return visual_result

        async def search(query, time_range=None):
            return [dict(h) for h in hits]

        self.svc = mods.ws.WebSearchService(scrape_fn=scrape, plan_fn=plan, visual_fn=visual)
        self.svc.search = search

    def gather(self):
        return run(self.svc.gather("python asyncio timeout?", repo_scoped=False))


def by_url(ctx):
    return {s.url.split("/")[2].split(".")[0]: s for s in ctx.sources}


def test_ok_page_untouched_and_thin_page_recovered_via_ocr(mods, monkeypatch):
    h = Harness(mods, monkeypatch)
    ctx = h.gather()
    s = by_url(ctx)
    assert (s["ok"].status, s["ok"].method) == ("OK", "text") and s["ok"].text.startswith("Asyncio timeouts are handled")
    assert s["ok"].title == "Asyncio guide"
    assert s["js"].method == "visual-ocr" and s["js"].status == "IMAGE_TEXT" and s["js"].text == OCR
    assert h.visual_calls == [("https://js.example/b", QUERY)]                       # query-focused, only the failed page
    f = ctx.footer()
    assert "1. [Asyncio guide](https://ok.example/a)\n" in f                          # OK page: no badge at all
    assert "2. [JS page](https://js.example/b) · read via screenshot OCR" in f
    assert "[Recovered via screenshot OCR of the rendered page; may contain recognition errors.]" in ctx.text
    note = ctx.recovery_note()
    assert "1 of 3 pages was image-based; recovered via OCR" in note and "1 restricted (snippet only)" in note


def test_restricted_page_never_sent_to_visual_and_labelled(mods, monkeypatch):
    h = Harness(mods, monkeypatch, mode="always")
    ctx = h.gather()
    s = by_url(ctx)
    assert (s["wall"].status, s["wall"].method, s["wall"].text) == ("RESTRICTED", "restricted", "")
    assert all("wall.example" not in u for u, _ in h.visual_calls)
    assert "(access-restricted page; search snippet only)\nwall snippet" in ctx.text
    assert "3. [Wall page](https://wall.example/c) · ⚠ restricted, snippet only" in ctx.footer()
    assert ctx.degraded


def test_http_402_is_restricted_even_with_text(mods, monkeypatch):
    pages = dict(PAGES, **{"wall.example": page(PROSE, status_code=402)})
    h = Harness(mods, monkeypatch, mode="always", pages=pages)
    s = by_url(h.gather())
    assert s["wall"].method == "restricted" and not h.visual_calls or all("wall" not in u for u, _ in h.visual_calls)


def test_visual_off_gives_snippet_only(mods, monkeypatch):
    h = Harness(mods, monkeypatch, mode="off")
    ctx = h.gather()
    s = by_url(ctx)
    assert h.visual_calls == [] and s["js"].method == "snippet-only" and s["js"].text == ""
    assert "(search snippet only; page could not be read)\njs snippet" in ctx.text
    assert "2. [JS page](https://js.example/b) · snippet only" in ctx.footer()
    assert "1 unreadable (snippet only)" in ctx.recovery_note()


def test_no_relevant_content_is_not_used(mods, monkeypatch):
    h = Harness(mods, monkeypatch, visual_result="NO_RELEVANT_CONTENT")
    ctx = h.gather()
    s = by_url(ctx)
    assert len(h.visual_calls) == 1 and s["js"].method == "snippet-only" and s["js"].text == ""
    assert "Recovered via screenshot OCR" not in ctx.text and "no relevant content" in s["js"].note


def test_missing_vision_capability_skips_visual(mods, monkeypatch):
    h = Harness(mods, monkeypatch, vision=False)
    s = by_url(h.gather())
    assert h.visual_calls == [] and s["js"].method == "snippet-only" and "no vision" in s["js"].note


def test_auto_caps_pages_and_always_does_not(mods, monkeypatch):
    two_js = [dict(HITS[1], url="https://js.example/b"), dict(HITS[1], url="https://js2.example/b", title="JS2")]
    pages = dict(PAGES, **{"js2.example": PAGES["js.example"]})
    auto = Harness(mods, monkeypatch, hits=two_js, pages=pages, max_pages=1)
    s = by_url(auto.gather())
    assert len(auto.visual_calls) == 1 and sorted(x.method for x in s.values()) == ["snippet-only", "visual-ocr"]
    always = Harness(mods, monkeypatch, hits=two_js, pages=pages, max_pages=1, mode="always")
    always.gather()
    assert len(always.visual_calls) == 2


def test_garbled_page_is_redecoded_without_visual(mods, monkeypatch):
    bad = ("Asyncio timeouts in Python: Ã©tÃ© donâ€™t forget the timeout â€œquotedâ€\u009d text. " * 20)
    pages = dict(PAGES, **{"js.example": page(bad, title="JS page")})
    h = Harness(mods, monkeypatch, pages=pages)
    ctx = h.gather()
    s = by_url(ctx)
    assert s["js"].method == "re-decoded" and "Ã" not in s["js"].text and "été" in s["js"].text and not h.visual_calls
    assert "2. [JS page](https://js.example/b) · encoding repaired" in ctx.footer()
    assert "fixed by re-decoding" in ctx.recovery_note()


def test_unrepairable_garbled_goes_to_visual(mods, monkeypatch):
    pages = dict(PAGES, **{"js.example": page(("data ���� binary noise " * 20))})
    h = Harness(mods, monkeypatch, pages=pages)
    s = by_url(h.gather())
    assert s["js"].status == "GARBLED" and s["js"].method == "visual-ocr" and len(h.visual_calls) == 1


def test_off_topic_dropped_but_snippet_kept(mods, monkeypatch):
    cooking = "Preheat the oven and whisk the eggs with sugar until pale. Fold in flour and bake the cake. " * 10
    pages = dict(PAGES, **{"ok.example": page(cooking)})
    h = Harness(mods, monkeypatch, pages=pages)
    ctx = h.gather()
    s = by_url(ctx)
    assert s["ok"].status == "OFF_TOPIC" and s["ok"].method == "snippet-only" and s["ok"].text == ""
    assert "Preheat" not in ctx.text and "ok snippet" in ctx.text
    assert "off-topic" in ctx.recovery_note()


def test_visual_failure_and_timeout_fall_back_to_snippet(mods, monkeypatch):
    h = Harness(mods, monkeypatch)

    async def boom(url, query):
        raise RuntimeError("no browser")
    h.svc._visual = boom
    assert by_url(h.gather())["js"].method == "snippet-only"

    async def slow(url, query):
        await asyncio.sleep(10)
    h.svc._visual = slow
    monkeypatch.setattr(mods.settings, "web_visual_timeout_s", 0.2)
    assert by_url(h.gather())["js"].method == "snippet-only"


def test_visual_runs_even_after_text_pass_timeout(mods, monkeypatch):
    """The text pass is cancelled at web_timeout_s; the recovery pass has its own budget and is not."""
    h = Harness(mods, monkeypatch)
    monkeypatch.setattr(mods.settings, "web_timeout_s", 0.3)
    orig = h.svc._scrape

    async def scrape(url, pw):
        if "ok.example" in url:
            await asyncio.sleep(5)                     # cancelled by the text-pass budget
        return await orig(url, pw)
    h.svc._scrape = scrape
    s = by_url(h.gather())
    assert s["ok"].method == "snippet-only" and s["ok"].status == "FAILED"
    assert s["js"].method == "visual-ocr"


def test_ocr_summarise_and_tiles():
    from services.scraper.visual import NO_RELEVANT, ocr_summarise, split_tiles
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (200, 4000), "white").save(buf, "PNG")
    tiles = split_tiles(buf.getvalue(), 3)
    assert len(tiles) == 3 and Image.open(io.BytesIO(tiles[0])).height == 1600

    class Chat:
        def __init__(self, reply):
            self.reply, self.seen = reply, []

        async def ainvoke(self, msgs):
            self.seen.append(msgs)
            return SimpleNamespace(content=self.reply)

    with stubbed_imports():
        chat = Chat("Summary: ok")
        assert run(ocr_summarise(tiles, QUERY, chat_factory=lambda: chat)) == "Summary: ok" and len(chat.seen) == 1
        assert run(ocr_summarise(tiles, QUERY, chat_factory=lambda: Chat("  NO_RELEVANT_CONTENT "))) == NO_RELEVANT
        assert run(ocr_summarise([], QUERY, chat_factory=lambda: chat)) == NO_RELEVANT


def test_screenshot_tiles_without_playwright_is_empty():
    from services.scraper.visual import screenshot_tiles
    assert run(screenshot_tiles("https://example.invalid/", max_tiles=2, timeout_s=1)) == []
