"""Spoken "submit" phrases. Pure functions, no imports from the rest of the app (mirrored by static/js/voice.js: keep the two in step).

split_trigger("fix the bug, Go ahead, now.", phrases) -> ("fix the bug", True)

Rules: case / unicode punctuation / whitespace are ignored; a phrase only counts at the END of the text (trailing fillers such as "please"
are allowed after it); a phrase in the middle ("go ahead and refactor it") never triggers. Letters, combining marks (Devanagari vowel signs)
and digits form words, everything else separates them.
"""
from __future__ import annotations

import unicodedata
from typing import List, Sequence, Tuple

DEFAULT_PHRASES = ("go ahead now,process now,go ahead,submit now,send it now,ab bhejo,process karo,bhej do,अब भेजो,प्रोसेस करो")

# Words that may follow a trigger phrase without cancelling it ("go ahead now, please", "ab bhejo yaar").
FILLERS = frozenset({"please", "pls", "plz", "thanks", "thank", "you", "thankyou", "kindly", "ok", "okay", "ji", "yaar", "na", "bas", "जी", "यार", "ना", "बस"})
_MAX_PASSES = 3
_TRIM = " \t\r\n,;:-–—…"


def _is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN"


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    """[(normalised word, start, end)] over the ORIGINAL string, so a phrase can be cut out exactly."""
    out: List[Tuple[str, int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if _is_word_char(text[i]):
            j = i
            while j < n and _is_word_char(text[j]):
                j += 1
            out.append((unicodedata.normalize("NFKC", text[i:j]).casefold(), i, j))
            i = j
        else:
            i += 1
    return out


def parse_phrases(phrases: "str | Sequence[str] | None") -> List[List[str]]:
    """'a b, c d' (or a list) -> token lists, longest first (so 'go ahead now' wins over 'go ahead'); duplicates and empties dropped."""
    raw = phrases.split(",") if isinstance(phrases, str) else list(phrases or [])
    seen, out = set(), []
    for p in raw:
        toks = [t for t, _, _ in _tokens(str(p))]
        key = tuple(toks)
        if toks and key not in seen:
            seen.add(key)
            out.append(toks)
    out.sort(key=len, reverse=True)
    return out


def split_trigger(text: str, phrases: "str | Sequence[str] | None" = None) -> Tuple[str, bool]:
    """(text without the trailing trigger phrase, triggered). Untriggered text is returned stripped but otherwise unchanged.
    A text that is ONLY a trigger phrase gives ("", True): the caller must not submit an empty prompt."""
    text = (text or "").strip()
    plist = parse_phrases(DEFAULT_PHRASES if phrases is None else phrases)
    if not text or not plist:
        return text, False
    triggered, cur = False, text
    for _ in range(_MAX_PASSES):                     # "go ahead now go ahead now" -> both are stripped
        toks = _tokens(cur)
        end = len(toks)
        while end > 0 and toks[end - 1][0] in FILLERS:
            end -= 1
        hit = 0
        for ph in plist:
            k = len(ph)
            if end >= k and [t for t, _, _ in toks[end - k:end]] == ph:
                hit = k
                break
        if not hit:
            break
        triggered = True
        start = toks[end - hit][1]
        cur = cur[:start].rstrip(_TRIM)
    return (cur.strip() if triggered else text), triggered
