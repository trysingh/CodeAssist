"""Visual fallback for pages whose text could not be extracted: screenshot the rendered page, let the configured vision model read it.

Only for pages that failed TEXT extraction (JS shells, canvas, text-in-images). It never goes past an access wall: if the rendered
page looks restricted (captcha / login / paywall / HTTP 401-403) nothing is captured. No stealth plugins, no cookie or header tricks.
"""
from __future__ import annotations

import asyncio
import io
from typing import Any, Awaitable, Callable, List, Optional

from config.settings import settings
from services.scraper.quality import RESTRICTED, assess

TILE_HEIGHT = 1600           # px per tile: tall enough for a few paragraphs, small enough for a vision model's image budget
NO_RELEVANT = "NO_RELEVANT_CONTENT"
OCR_QUERY_PROMPT = (
    "These images are consecutive screenshots (top to bottom) of one web page. The user is searching for: {query!r}.\n"
    "1. Transcribe ONLY the text that is relevant to that search; ignore menus, ads, cookie banners and unrelated text.\n"
    "2. Then write 'Summary:' followed by a short factual summary (max 8 sentences) of what the relevant text says. "
    "Do not add facts that are not visible on the page.\n"
    f"If nothing on the page is relevant, or the text is unreadable, reply with exactly {NO_RELEVANT}.\n"
    "Output only the extracted content and the summary, no commentary."
)


def split_tiles(png: bytes, max_tiles: int, tile_h: int = TILE_HEIGHT) -> List[bytes]:
    """One tall PNG -> up to max_tiles PNGs of tile_h px (top of the page first). Without PIL: the single image as is."""
    try:
        from PIL import Image
    except ImportError:     # pragma: no cover
        return [png][:max_tiles]
    img = Image.open(io.BytesIO(png))
    out: List[bytes] = []
    for top in range(0, img.height, tile_h):
        if len(out) >= max_tiles:
            break
        buf = io.BytesIO()
        img.crop((0, top, img.width, min(top + tile_h, img.height))).save(buf, "PNG")
        out.append(buf.getvalue())
    return out


async def _capture(url: str, max_tiles: int) -> List[bytes]:
    from playwright.async_api import async_playwright          # lazy: only installed / needed when the fallback runs
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=settings.playwright_headless)
        try:
            page = await (await browser.new_context(viewport={"width": 1280, "height": 900})).new_page()
            resp = await page.goto(url, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=4000)     # best effort: images / canvas need a moment
            except Exception:  # noqa: BLE001
                pass
            body = ""
            try:
                body = await page.inner_text("body")
            except Exception:  # noqa: BLE001
                pass
            q = assess(body, html_head=await page.title(), status_code=resp.status if resp else None)
            if q.status == RESTRICTED:                           # rendered page is a wall: do not capture, do not look further
                return []
            return split_tiles(await page.screenshot(full_page=True, type="png"), max_tiles)
        finally:
            await browser.close()


async def screenshot_tiles(url: str, *, max_tiles: int = 3, timeout_s: float = 20.0) -> List[bytes]:
    """PNG tiles (~1600px tall, top first) of the rendered page; [] when playwright is missing, it times out, fails,
    or the rendered page is access-restricted."""
    try:
        return await asyncio.wait_for(_capture(url, max(1, max_tiles)), timeout_s)
    except Exception:  # noqa: BLE001 - ImportError, timeout, navigation error: all mean "no screenshot"
        return []


ChatFactory = Callable[[], Any]


async def ocr_summarise(tiles: List[bytes], query: str, *, chat_factory: Optional[ChatFactory] = None) -> str:
    """Read the tiles with the configured vision model; returns query-focused text + summary, or NO_RELEVANT_CONTENT.
    chat_factory() -> object with `async ainvoke([message])` (tests inject one; default = settings.vision_model via the runner)."""
    if not tiles:
        return NO_RELEVANT
    from services.vision import image_message, vision_chat
    llm = (chat_factory or (lambda: vision_chat("WEB_VISUAL_OCR")))()
    msg = image_message(OCR_QUERY_PROMPT.format(query=(query or "the page's main content")[:200]), list(tiles))
    out = str((await llm.ainvoke([msg])).content or "").strip()
    return NO_RELEVANT if not out or NO_RELEVANT in out[:60] else out
