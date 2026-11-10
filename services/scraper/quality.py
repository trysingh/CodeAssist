"""Is this scraped page usable? Pure, dependency-free checks that decide how a page is recovered (or not).

Status -> action (see WebSearchService):
  OK          use as is
  GARBLED     wrong charset / mojibake / binary soup: try_repair_encoding(), else screenshot OCR
  THIN        too little text (JS shell, nav soup): screenshot OCR
  IMAGE_TEXT  little text but the HTML is full of <img>/<canvas>/PDF embeds: screenshot OCR
  RESTRICTED  captcha / login / paywall / HTTP 401-402-403: NEVER bypassed, search snippet only
  OFF_TOPIC   readable, but shares no keywords with the query: dropped, search snippet only
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Optional

OK, GARBLED, THIN, IMAGE_TEXT, RESTRICTED, OFF_TOPIC = "OK", "GARBLED", "THIN", "IMAGE_TEXT", "RESTRICTED", "OFF_TOPIC"

THIN_WORDS = 40              # below this a page is "too little text" (same floor the service always used)
IMAGE_TEXT_WORDS = 150       # image signals only matter when the text is this short
MIN_CHARS_GARBLE = 40        # shorter texts are not judged for garbling (too little evidence)

# ---- restricted: access walls. Extends scraper.engine.IS_BLOCKED / IS_WALL. Detection only: nothing here (or anywhere) bypasses a wall.
_WALL_PATTERNS = [
    # bot checks / captcha
    r"captcha", r"verify (?:that )?you(?:'re| are) (?:a )?human", r"are you a robot", r"checking your browser",
    r"cf-browser-verification", r"unusual traffic", r"request unsuccessful", r"access to this page has been denied",
    r"ddos protection by", r"enable javascript and cookies to continue", r"just a moment\.\.\.",
    # login walls
    r"(?:please )?(?:log ?in|sign ?in) to (?:continue|view|read|access|see|download)", r"you must (?:be )?(?:log(?:ged)? ?in|sign(?:ed)? ?in)",
    r"(?:log ?in|sign ?in) required", r"create (?:a )?(?:free )?account to (?:continue|read|view|access)",
    # paywalls
    r"subscribe (?:now )?to (?:continue|read|unlock|access|view)", r"subscribers? only", r"subscription required",
    r"(?:for|available to|reserved for) (?:paid |premium |registered )?(?:subscribers|members) only",
    r"unlock (?:this|the full) (?:article|story|content)", r"you(?:'ve| have) reached your (?:free |monthly )?(?:article )?limit",
    r"continue reading with a (?:subscription|premium)", r"already a subscriber\?",
    # HTTP-level
    r"\b40[123]\b[^\n]{0,24}(?:unauthori[sz]ed|payment required|forbidden)", r"payment required", r"access denied",
    r"host not in allowlist", r"403 forbidden",
]
_WALL = re.compile("|".join(f"(?:{p})" for p in _WALL_PATTERNS), re.I)
RESTRICTED_STATUS = frozenset({401, 402, 403})
_WALL_HEAD_CHARS = 800       # a wall phrase this early is the page; later it may just be an article about paywalls
_WALL_MAX_WORDS = 300        # long pages are only judged by their opening (a footer "subscribe to continue" is not a wall)

# ---- garbled
_FFFD = "�"
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MOJI = re.compile(r"Ã[\u0080-¿‘-›€ŒœŠšŸŽžˆ˜™ƒ]"
                   r"|â€[\u0080-ÿ‘-›€™˜œšžŸ]|Â[ -¿]|ï»¿")
_PLAIN_PUNCT = set(".,;:!?'\"()-_/[]{}<>=+*#@%&|\\$~^`")

# ---- image-based pages
_IMG = re.compile(r"<img\b", re.I)
_CANVAS = re.compile(r"<canvas\b", re.I)
_PDF = re.compile(r"<(?:embed|object|iframe)\b[^>]*(?:\.pdf\b|application/pdf)|pdf\.js|pdfjs|viewer\.html\?file=", re.I)
_SVG = re.compile(r"<svg\b", re.I)

_STOP = frozenset("""about above after again against all also and any are because been before being below between both but can
could did does doing down during each few for from further had has have having her here hers him his how into its itself just
let more most much must not now off once only other our out over own same she should some such than that the their them then there
these they this those through too under until upon very was were what when where which while who whom why will with would you your
latest new newest best vs versus guide tutorial example examples using use used get make made""".split())
_WORD = re.compile(r"[^\W\d_]{4,}", re.U)


@dataclass
class Quality:
    status: str = OK
    reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == OK


def _junk_ratio(text: str) -> float:
    """Fraction of lines too short to be prose (same measure as scraper.engine._junk_ratio, which is not imported: it is heavy)."""
    lines = [l for l in text.split("\n") if l.strip()]
    return sum(1 for l in lines if len(l) < 45) / len(lines) if lines else 1.0


def _count_words(text: str) -> int:
    n = len(text.split())
    cjk = sum(1 for c in text if "぀" <= c <= "鿿" or "가" <= c <= "힯")
    return max(n, cjk // 2) if cjk > len(text) * 0.3 else n         # no spaces between words in CJK


def _keywords(query: str) -> list:
    seen: Dict[str, None] = {}
    for w in _WORD.findall((query or "").lower()):
        if w not in _STOP:
            seen[w] = None
    return list(seen)


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 5 and w.endswith("s") else w


def _overlap(text: str, kws: list) -> float:
    low = text.lower()
    return sum(1 for k in kws if _stem(k) in low) / len(kws)


def mojibake_count(text: str) -> int:
    return len(_MOJI.findall(text or ""))


def _garble_reason(text: str, m: Dict[str, float]) -> str:
    n = len(text)
    if m["fffd"] >= 3 or m["fffd"] > n * 0.005:
        return f"{int(m['fffd'])} replacement characters (wrong charset)"
    if m["mojibake"] >= 3 or m["mojibake"] > max(m["words"], 1) * 0.02:
        return f"{int(m['mojibake'])} mojibake sequences (UTF-8 read as latin-1/cp1252)"
    if m["ctrl"] >= 3 or m["ctrl"] > n * 0.005:
        return f"{int(m['ctrl'])} control characters (binary or wrong charset)"
    if m["words"] >= 20 and m["alpha_ratio"] < 0.4:
        return f"only {m['alpha_ratio']:.0%} of tokens are words"
    if m["symbol_ratio"] > 0.25:
        return f"{m['symbol_ratio']:.0%} of characters are symbols"
    return ""


def assess(text: Optional[str], *, query: Optional[str] = None, html_head: Optional[str] = None,
           status_code: Optional[int] = None) -> Quality:
    """Classify extracted page text. Order: RESTRICTED, GARBLED, IMAGE_TEXT, THIN, OFF_TOPIC, else OK.

    html_head: (the start of) the page HTML, used only for image/canvas/PDF signals and wall phrases in <title>.
    OFF_TOPIC is deliberately timid: it needs >= 3 query keywords and zero of them anywhere in the text."""
    text = text or ""
    words = _count_words(text)
    m: Dict[str, float] = {"words": words, "chars": len(text)}

    # 1. access walls: decided first, because nothing may be done to get past them
    if status_code in RESTRICTED_STATUS:
        return Quality(RESTRICTED, f"HTTP {status_code}", m)
    probe = text if words < _WALL_MAX_WORDS else text[:_WALL_HEAD_CHARS]
    hit = _WALL.search(probe) or (_WALL.search(html_head[:4000]) if html_head and words < _WALL_MAX_WORDS else None)
    if hit and (words < _WALL_MAX_WORDS or hit.start() < _WALL_HEAD_CHARS):
        return Quality(RESTRICTED, f"access wall marker: {hit.group(0)[:40]!r}", m)

    # 2. garbled
    if len(text) >= MIN_CHARS_GARBLE:
        toks = text.split()
        non_space = [c for c in text if not c.isspace()]
        m.update(
            fffd=text.count(_FFFD), ctrl=len(_CTRL.findall(text)), mojibake=mojibake_count(text),
            alpha_ratio=sum(1 for t in toks if sum(c.isalpha() for c in t) >= 2) / max(len(toks), 1),
            symbol_ratio=sum(1 for c in non_space if not c.isalnum() and c not in _PLAIN_PUNCT
                             and unicodedata.category(c)[0] in "SC") / max(len(non_space), 1))
        why = _garble_reason(text, m)
        if why:
            return Quality(GARBLED, why, m)

    # 3. little text, lots of pictures / canvas / PDF embeds
    if words < IMAGE_TEXT_WORDS and html_head:
        imgs, canv, pdf = len(_IMG.findall(html_head)), len(_CANVAS.findall(html_head)), len(_PDF.findall(html_head))
        m.update(images=imgs, canvas=canv, pdf=pdf)
        if canv or pdf or imgs >= 5:
            sig = "canvas" if canv else "embedded PDF" if pdf else f"{imgs} images"
            return Quality(IMAGE_TEXT, f"{words} words but page is image-based ({sig})", m)

    # 4. thin
    m["junk_ratio"] = jr = _junk_ratio(text)
    if words < THIN_WORDS:
        return Quality(THIN, f"only {words} words", m)
    if jr > 0.85 and words < IMAGE_TEXT_WORDS:
        return Quality(THIN, f"{jr:.0%} of lines are nav-sized fragments", m)

    # 5. off topic (light lexical filter)
    kws = _keywords(query or "")
    if len(kws) >= 3:
        m["overlap"] = ov = _overlap(text, kws)
        if ov == 0:
            return Quality(OFF_TOPIC, f"none of {len(kws)} query keywords appear", m)
    return Quality(OK, "", m)


# ------------------------------------------------------------------ encoding repair
_RUN = re.compile("[\u0080-ÿŒœŠšŸŽžƒˆ˜‘-›€™]+")


def _to_bytes(run: str) -> Optional[bytes]:
    out = bytearray()
    for ch in run:
        try:
            out += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) > 0xFF:
                return None
            out.append(ord(ch))                                      # cp1252 holes (0x81, 0x8d ...) map straight through
    return bytes(out)


def _fix_run(m: "re.Match[str]") -> str:
    run = m.group(0)
    b = _to_bytes(run)
    if b is None:
        return run
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return run                                                   # a genuine accented run: leave it


def try_repair_encoding(text: str) -> str:
    """Undo 'UTF-8 decoded as latin-1/cp1252' (Ã©, â€™ ...), up to two layers deep. ftfy when installed, else a built-in
    run-by-run round trip that only changes runs that decode as valid UTF-8. Text without mojibake is returned unchanged."""
    if not text or not mojibake_count(text):
        return text
    try:
        import ftfy                                                  # type: ignore
        fixed = ftfy.fix_text(text)
        if mojibake_count(fixed) < mojibake_count(text):
            return fixed
    except Exception:  # noqa: BLE001 - ftfy absent or failing: use the built-in repair
        pass
    for _ in range(2):
        text = _RUN.sub(_fix_run, text)
        if not mojibake_count(text):
            break
    return text
