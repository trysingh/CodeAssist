"""Voice routes (included by views/api_view.py). Thin: all logic lives in services/speech. Kept in its own module so it imports without the
rest of the app (no controller / vector store), which is also what lets tests/test_voice.py mount it alone."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from config.settings import settings
from models.schemas import VoiceTextRequest
from services.speech import service as speech

router = APIRouter()


@router.get("/api/v1/voice/config")
async def voice_config(runner: str = ""):
    """{enabled, mode: server|browser, providers, pause_ms, max_seconds, trigger_phrases, translate, tts: {enabled, provider}} for the UI."""
    return speech.voice_config(runner)


@router.post("/api/v1/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...), runner: str = Form(""), lang: str = Form("auto")):
    """16 kHz mono wav (any format the provider accepts) -> {ok, text, transcript, language, provider, triggered, refined, errors}.
    Every provider failed / none configured: HTTP 200 {ok: false, fallback: "browser", errors} so the client switches to the browser's speech recognition."""
    if not settings.voice_enabled:
        raise HTTPException(status_code=404, detail="Voice input is disabled (VOICE_ENABLED=false).")
    limit = int(settings.voice_max_bytes)
    data = await audio.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Recording exceeds {limit // (1024 * 1024) or 1} MB.")
    try:
        return await speech.transcribe_audio(data, runner, (lang or "auto").strip().lower())
    except Exception as exc:  # noqa: BLE001 - voice must never 500: the client falls back to the browser
        return {"ok": False, "fallback": "browser", "errors": [f"{type(exc).__name__}: {exc}"[:240]]}


@router.post("/api/v1/voice/refine")
async def voice_refine(req: VoiceTextRequest):
    """Browser-mode transcript (Hindi / Hinglish included) -> clean English + trigger detection. Same shape as /transcribe."""
    try:
        return await speech.refine_text(req.text, req.runner)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"[:240]]}


@router.post("/api/v1/voice/speak")
async def voice_speak(req: VoiceTextRequest):
    """Read-aloud audio (audio/wav) from the Ollama TTS model; 503 JSON when unavailable so the client uses speechSynthesis."""
    try:
        audio, media = await speech.speak(req.text)
    except speech.SpeakError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc), "fallback": "browser"})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"detail": f"{type(exc).__name__}: {exc}"[:200], "fallback": "browser"})
    return Response(content=audio, media_type=media)
