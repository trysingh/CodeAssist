"""Tidy a transcript into a clean English prompt with the user's chosen chat runner. Never raises: any failure returns the raw transcript."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Tuple

from config.settings import settings
from services.speech.providers import CLEAN_ENGLISH_RULES

log = logging.getLogger("voice")

# Hinglish markers. Strong words are rare in English, so one is enough; weak ones also occur in English ("me", "to") or are short, so two are needed.
_STRONG = re.compile(r"\b(hai|hain|karo|kro|kya|mujhe|mujhko|nahi|nahin|matlab|haan|accha|acha|kaise|karna|chahiye|yeh|yahan|isko|usko|wala|wali|"
                     r"bhejo|batao|dikhao|karke|abhi|samajh|theek|thik|mein)\b", re.I)
_WEAK = re.compile(r"\b(ko|ka|ke|ki|aur|toh|bhi|kar|dena|me|ye)\b", re.I)


SYSTEM = ("You clean up speech-to-text transcripts for a coding assistant. " + CLEAN_ENGLISH_RULES +
          " Reply with the final English prompt text only: no quotes, no preface, no markdown.")


def needs_refine(text: str) -> bool:
    """Cheap check: anything that is not plain English-looking (non-ASCII letters: Devanagari, Gurmukhi ...; two or more Hinglish words)."""
    t = text or ""
    if any(ord(c) > 127 and c.isalpha() for c in t):
        return True
    return bool(_STRONG.search(t)) or len(_WEAK.findall(t)) >= 2


def _blocking_refine(text: str, runner: str) -> str:
    from services.runners import make_chat, resolve_model, runner_scope
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [SystemMessage(content=SYSTEM), HumanMessage(content="Transcript:\n" + text)]
    except Exception:  # noqa: BLE001
        msgs = [("system", SYSTEM), ("human", "Transcript:\n" + text)]
    with runner_scope(runner):
        llm = make_chat(resolve_model(""), temperature=0, max_tokens=512, phase="VOICE")
        resp = llm.invoke(msgs)
    out = resp.content if isinstance(getattr(resp, "content", None), str) else str(getattr(resp, "content", resp))
    return out


def _clean_reply(out: str) -> str:
    out = (out or "").strip()
    out = re.sub(r"^```[\w-]*\s*|\s*```$", "", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    return out


async def refine_to_english(text: str, runner: str = "") -> Tuple[str, bool]:
    """(english text, refined). refined=False -> `text` came back unchanged (disabled, already English, or the model failed / timed out)."""
    text = (text or "").strip()
    if not text or not settings.voice_translate_to_english or not needs_refine(text):
        return text, False
    try:
        out = await asyncio.wait_for(asyncio.to_thread(_blocking_refine, text, runner), timeout=max(0.05, float(settings.voice_refine_timeout_s)))
        out = _clean_reply(out)
        if not out:
            return text, False
        return out, True
    except BaseException as exc:  # noqa: BLE001 - incl. TimeoutError; a cancelled request must still propagate
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        log.warning("voice refine failed (%s): %s", type(exc).__name__, str(exc)[:200])
        return text, False
