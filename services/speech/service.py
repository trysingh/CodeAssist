"""What the /api/v1/voice/* routes call. Nothing here raises to the route except SpeakError (-> 503): a failed transcription is a JSON answer."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Tuple

import httpx

from config.settings import settings
from services.speech.providers import SttEmpty, SttResult
from services.speech.refine import refine_to_english
from services.speech.resolver import resolve_chain
from services.speech.triggers import parse_phrases, split_trigger

log = logging.getLogger("voice")


class SpeakError(RuntimeError):
    pass


def _phrases() -> str:
    return settings.voice_trigger_phrases


def _safe_runner(runner: str) -> str:
    """The request's runner when it exists and is configured, else '' (= default). Voice must work whatever the runner picker says."""
    r = (runner or "").strip().lower()
    if not r:
        return ""
    try:
        from services.runners import require_runner
        return require_runner(r)
    except Exception:  # noqa: BLE001
        return ""


def tts_info() -> Dict[str, Any]:
    prov = (settings.tts_provider or "ollama").strip().lower()
    return {"enabled": bool(settings.tts_enabled) and prov != "off", "provider": prov}


def voice_config(runner: str = "") -> Dict[str, Any]:
    chain = [] if not settings.voice_enabled else resolve_chain(runner)
    return {"enabled": bool(settings.voice_enabled), "mode": "server" if chain else "browser", "providers": [p.name for p in chain],
            "pause_ms": int(settings.voice_pause_ms), "max_seconds": int(settings.voice_max_seconds),
            "trigger_phrases": [" ".join(p) for p in parse_phrases(_phrases())], "translate": bool(settings.voice_translate_to_english),
            "tts": tts_info()}


def _finish(raw: str, english: str, *, language: str, provider: str, refined: bool, provider_trigger: bool, errors: List[str]) -> Dict[str, Any]:
    """Strip a trailing trigger phrase (looked for in the raw transcript, the English text and the provider's own verdict)."""
    text, trig_en = split_trigger(english, _phrases())
    _, trig_raw = split_trigger(raw, _phrases())
    triggered = bool(provider_trigger or trig_en or trig_raw)
    return {"ok": True, "text": text, "transcript": raw, "language": language, "provider": provider, "triggered": triggered,
            "refined": refined, "errors": errors}


async def refine_text(text: str, runner: str = "") -> Dict[str, Any]:
    """Browser-mode transcripts (Web Speech): trigger detection on the raw text first, then translate / tidy the rest."""
    raw = (text or "").strip()
    clean_raw, trig_raw = split_trigger(raw, _phrases())
    english, refined = await refine_to_english(clean_raw, _safe_runner(runner))
    return _finish(raw, english, language="", provider="browser", refined=refined, provider_trigger=trig_raw, errors=[])


async def transcribe_audio(wav: bytes, runner: str = "", lang: str = "auto") -> Dict[str, Any]:
    chain = resolve_chain(runner) if settings.voice_enabled else []
    if not chain:
        return {"ok": False, "fallback": "browser", "errors": ["no server speech-to-text provider is configured"]}
    errors: List[str] = []
    for p in chain:
        try:
            res: SttResult = await asyncio.to_thread(p.transcribe, wav, lang or "auto")
        except SttEmpty:
            return {"ok": True, "text": "", "transcript": "", "language": "", "provider": p.name, "triggered": False, "refined": False,
                    "empty": True, "errors": errors}
        except Exception as exc:  # noqa: BLE001 - next provider
            msg = (str(exc) if str(exc).startswith(p.name + ":") else f"{p.name}: {type(exc).__name__}: {exc}")[:240]
            log.warning("voice stt failed: %s", msg)
            errors.append(msg)
            continue
        raw = (res.transcript or res.text or "").strip()
        if res.already_english:
            return _finish(raw, res.text, language=res.language, provider=res.provider or p.name, refined=True,
                           provider_trigger=bool(res.triggered), errors=errors)
        clean_raw, trig_raw = split_trigger(raw, _phrases())          # BEFORE refining, so "ab bhejo" never reaches the translator
        english, refined = await refine_to_english(clean_raw, _safe_runner(runner))
        return _finish(raw, english, language=res.language, provider=res.provider or p.name, refined=refined, provider_trigger=trig_raw, errors=errors)
    return {"ok": False, "fallback": "browser", "errors": errors}


async def speak(text: str) -> Tuple[bytes, str]:
    """Read-aloud audio from Ollama's OpenAI-style /v1/audio/speech. SpeakError -> the route answers 503 and the browser speaks instead."""
    info = tts_info()
    if not info["enabled"] or info["provider"] != "ollama":
        raise SpeakError("server read-aloud is not enabled (TTS_ENABLED=true, TTS_PROVIDER=ollama)")
    text = (text or "").strip()[:4000]
    if not text:
        raise SpeakError("nothing to read")
    base = (settings.tts_base_url or settings.ollama_base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    body: Dict[str, Any] = {"model": settings.tts_model, "input": text, "response_format": "wav"}
    if settings.tts_voice:
        body["voice"] = settings.tts_voice

    def _post() -> httpx.Response:
        with httpx.Client(timeout=max(5.0, float(settings.stt_timeout_s or 60))) as c:
            return c.post(f"{base}/v1/audio/speech", json=body)

    try:
        r = await asyncio.to_thread(_post)
    except httpx.HTTPError as exc:
        raise SpeakError(f"{type(exc).__name__}: {exc}"[:200]) from None
    if r.status_code >= 400 or not r.content:
        raise SpeakError(f"speech server answered HTTP {r.status_code}: {(r.text or '')[:160]}")
    return r.content, (r.headers.get("content-type") or "audio/wav").split(";")[0]
