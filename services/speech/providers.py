"""Speech-to-text providers: one HTTP call each, wav bytes in, SttResult out. Blocking (httpx.Client): the service runs them in a worker thread.

    p.available() -> (ok, reason)          configuration check only, never touches the network
    p.transcribe(wav_bytes, lang_hint)     lang_hint: "auto" | "en" | "hi" | "pa" ...  -> SttResult; raises SttError on any failure
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

from config.settings import settings
from services.speech.triggers import parse_phrases

log = logging.getLogger("voice")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
CF_API = "https://api.cloudflare.com/client/v4"

CLEAN_ENGLISH_RULES = (
    "Output a CLEAN ENGLISH PROMPT: translate / tidy what was said into one English instruction or question. Keep code identifiers, file names, paths, "
    "CLI commands, library and product names and technical terms verbatim (do not translate or respell them). Remove fillers and hesitations "
    "(umm, uh, like, matlab, haan, accha, toh). Do not add content, do not answer or carry out the request, do not explain. If the speech is already "
    "clean English, return it with only light punctuation fixes."
)
WHISPER_PROMPT = ("Dictation of a software developer's instruction. The speaker may mix Hindi, Punjabi and English (Hinglish). "
                  "Technical terms: Python, FastAPI, pytest, git, pip, npm, Docker, API, JSON, SQL, endpoint, refactor.")


class SttError(RuntimeError):
    """A provider failed (HTTP error, bad reply). The message is safe to show: it never contains a key."""


class SttEmpty(SttError):
    """The provider worked but heard no speech. Not a failure to fail over from: silence stays silence on every provider."""


@dataclass
class SttResult:
    text: str                       # what the caller should use (English when already_english)
    language: str = ""
    provider: str = ""
    already_english: bool = False   # the provider also did the clean-English step (Gemini): no second model call needed
    transcript: str = ""            # the raw transcript in the spoken language ("" = same as text)
    triggered: Optional[bool] = None  # the provider's own verdict on a trailing trigger phrase (Gemini); None = not reported


def _http(timeout: float) -> httpx.Client:
    """One place to build the client: tests replace it with a MockTransport client."""
    return httpx.Client(timeout=timeout, follow_redirects=False)


def _fail_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        msg = err.get("message") if isinstance(err, dict) else str(err)
        msg = msg or json.dumps(body)[:200]
    except Exception:  # noqa: BLE001
        msg = (resp.text or "")[:200]
    return f"HTTP {resp.status_code}: {str(msg)[:200]}"


def _parse_json_loose(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```[\w-]*\s*|\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    for cand in (raw, m.group(0) if m else ""):
        try:
            val = json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(val, dict):
            return val
    return None


class SttProvider:
    name = ""

    def available(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def transcribe(self, wav: bytes, lang_hint: str = "auto") -> SttResult:
        raise NotImplementedError

    def _timeout(self) -> float:
        return max(1.0, float(settings.stt_timeout_s or 60))


def _lang(lang_hint: str) -> str:
    l = (lang_hint or "").strip().lower().split("-")[0]
    return "" if l in ("", "auto", "mixed") else l


class GeminiStt(SttProvider):
    """Gemini hears the audio and returns the original transcript AND the clean English prompt in one call (so already_english=True)."""
    name = "gemini"

    def available(self) -> Tuple[bool, str]:
        return (True, "") if settings.gemini_api_key else (False, "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

    def _prompt(self, lang_hint: str) -> str:
        phrases = ", ".join('"' + " ".join(p) + '"' for p in parse_phrases(settings.voice_trigger_phrases))
        hint = _lang(lang_hint)
        return (
            "Transcribe the speech in this audio. The speaker may use Hindi, Punjabi, English or a mix (Hinglish)."
            + (f" Likely language: {hint}." if hint else "")
            + " Reply with ONE JSON object and nothing else: "
              '{"transcript": the words exactly as spoken in the original language(s) (Devanagari / Gurmukhi / Latin as spoken), '
              '"language": a short code such as "en", "hi", "pa" or "mixed", '
              '"english": the clean English prompt, "triggered": true or false}. '
            + CLEAN_ENGLISH_RULES
            + f" The speaker may end with a spoken send command such as {phrases}: leave that command OUT of \"english\" and set \"triggered\" to true; "
              "otherwise \"triggered\" is false. If there is no intelligible speech, use empty strings."
        )

    def transcribe(self, wav: bytes, lang_hint: str = "auto") -> SttResult:
        model = (settings.aistudio_stt_model or "gemini-2.5-flash").strip()
        url = f"{GEMINI_BASE}/models/{model}:generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": self._prompt(lang_hint)},
                                                         {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        try:
            with _http(self._timeout()) as c:
                r = c.post(url, json=body, headers={"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"})
        except httpx.HTTPError as exc:
            raise SttError(f"gemini: {type(exc).__name__}: {exc}"[:240]) from None
        if r.status_code >= 400:
            raise SttError("gemini: " + _fail_text(r))
        try:
            data = r.json()
            parts = data["candidates"][0]["content"]["parts"]
            raw = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        except Exception:  # noqa: BLE001
            raise SttError("gemini: unexpected reply (no candidates)") from None
        obj = _parse_json_loose(raw)
        if obj is None:                              # model ignored the JSON instruction: treat the text as the transcript
            text = raw.strip()
            if not text:
                raise SttEmpty("gemini: empty transcript")
            return SttResult(text=text, provider=self.name, already_english=False, transcript=text)
        english = str(obj.get("english") or "").strip()
        transcript = str(obj.get("transcript") or "").strip()
        trig = obj.get("triggered")
        trig = trig if isinstance(trig, bool) else (str(trig).strip().lower() == "true" if trig is not None else None)
        if not english and not transcript:
            if trig:                                  # only a send command was spoken
                return SttResult(text="", provider=self.name, already_english=True, triggered=True, language=str(obj.get("language") or ""))
            raise SttEmpty("gemini: empty transcript")
        return SttResult(text=english or transcript, language=str(obj.get("language") or "").strip().lower(), provider=self.name,
                         already_english=bool(english), transcript=transcript or english, triggered=trig)


class CloudflareStt(SttProvider):
    name = "cloudflare"

    def available(self) -> Tuple[bool, str]:
        if settings.cf_account_id and settings.cf_api_token:
            return True, ""
        return False, "CF_ACCOUNT_ID / CF_API_TOKEN are not set"

    def transcribe(self, wav: bytes, lang_hint: str = "auto") -> SttResult:
        url = f"{CF_API}/accounts/{settings.cf_account_id}/ai/run/{settings.cf_stt_model}"
        body = {"audio": base64.b64encode(wav).decode("ascii"), "task": "transcribe", "vad_filter": True}
        if _lang(lang_hint):
            body["language"] = _lang(lang_hint)
        try:
            with _http(self._timeout()) as c:
                r = c.post(url, json=body, headers={"Authorization": f"Bearer {settings.cf_api_token}"})
        except httpx.HTTPError as exc:
            raise SttError(f"cloudflare: {type(exc).__name__}: {exc}"[:240]) from None
        if r.status_code >= 400:
            raise SttError("cloudflare: " + _fail_text(r))
        try:
            data = r.json()
            res = data.get("result", data) if isinstance(data, dict) else {}
            text = str(res.get("text") or "").strip()
            info = res.get("transcription_info") or {}
            language = str(info.get("language") or "").strip().lower() if isinstance(info, dict) else ""
        except Exception:  # noqa: BLE001
            raise SttError("cloudflare: unexpected reply") from None
        if not text:
            raise SttEmpty("cloudflare: empty transcript")
        return SttResult(text=text, language=language, provider=self.name, transcript=text)


class OpenAICompatStt(SttProvider):
    """POST {base}/audio/transcriptions (multipart). One class for a self-hosted server ("local"), OpenAI itself ("openai") and the
    OPENAI_COMPAT_* endpoint of the openai runner ("openai_compat", e.g. Groq)."""

    def __init__(self, kind: str):
        if kind not in ("local", "openai", "openai_compat"):
            raise ValueError(kind)
        self.name = kind

    def _cfg(self) -> Tuple[str, str, str]:          # (base_url, api_key, model)
        s = settings
        if self.name == "local":
            return s.stt_base_url, s.stt_api_key, s.stt_model
        if self.name == "openai":
            return s.openai_stt_base_url, s.openai_api_key, s.openai_stt_model
        return s.openai_compat_base_url, s.openai_compat_api_key, s.openai_compat_stt_model

    def available(self) -> Tuple[bool, str]:
        base, key, _ = self._cfg()
        if self.name == "local":
            return (True, "") if base else (False, "STT_BASE_URL is not set")
        if self.name == "openai":
            return (True, "") if key else (False, "OPENAI_API_KEY is not set")
        return (True, "") if base else (False, "OPENAI_COMPAT_BASE_URL is not set")

    @staticmethod
    def endpoint(base: str) -> str:
        """http://nuc:8000 -> http://nuc:8000/v1/audio/transcriptions ; https://api.groq.com/openai/v1 -> .../v1/audio/transcriptions"""
        b = (base or "").strip().rstrip("/")
        if b.endswith("/audio/transcriptions"):
            return b
        if not re.sub(r"^https?://[^/]+", "", b):     # no path at all: servers like speaches serve under /v1
            b += "/v1"
        return b + "/audio/transcriptions"

    def transcribe(self, wav: bytes, lang_hint: str = "auto") -> SttResult:
        base, key, model = self._cfg()
        data = {"model": model, "response_format": "json", "prompt": WHISPER_PROMPT}
        if _lang(lang_hint):
            data["language"] = _lang(lang_hint)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            with _http(self._timeout()) as c:
                r = c.post(self.endpoint(base), data=data, files={"file": ("audio.wav", wav, "audio/wav")}, headers=headers)
        except httpx.HTTPError as exc:
            raise SttError(f"{self.name}: {type(exc).__name__}: {exc}"[:240]) from None
        if r.status_code >= 400:
            raise SttError(f"{self.name}: " + _fail_text(r))
        try:
            body = r.json()
            text = str(body.get("text") or "").strip()
            language = str(body.get("language") or "").strip().lower()
        except Exception:  # noqa: BLE001
            text, language = (r.text or "").strip(), ""
            if text.startswith("{"):
                raise SttError(f"{self.name}: unexpected reply") from None
        if not text:
            raise SttEmpty(f"{self.name}: empty transcript")
        return SttResult(text=text, language=language, provider=self.name, transcript=text)


def make_provider(name: str) -> Optional[SttProvider]:
    n = (name or "").strip().lower()
    if n in ("gemini", "aistudio"):
        return GeminiStt()
    if n == "cloudflare":
        return CloudflareStt()
    if n in ("local", "openai", "openai_compat"):
        return OpenAICompatStt(n)
    return None
