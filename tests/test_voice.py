"""Voice input: trigger phrases, STT provider resolution, English refinement fallbacks, provider request shapes, fail-over and the HTTP routes.
No network and no real model: httpx goes through MockTransport, the chat model is a fake."""
import asyncio
import base64
import json
import sys
import time
import types

import httpx
import pytest

from config.settings import settings
from services.speech import providers as prov
from services.speech import refine as refine_mod
from services.speech import resolver, service
from services.speech.triggers import DEFAULT_PHRASES, parse_phrases, split_trigger

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in dict(llm_runner="ollama", stt_provider="auto", stt_base_url="", stt_api_key="", stt_model="Systran/faster-whisper-large-v3-turbo",
                     stt_timeout_s=5.0, gemini_api_key="", cf_account_id="", cf_api_token="", openai_compat_base_url="", openai_compat_api_key="",
                     openai_api_key="", openai_stt_base_url="https://api.openai.com/v1", openai_stt_model="gpt-4o-transcribe",
                     openai_compat_stt_model="whisper-large-v3-turbo", cf_stt_model="@cf/openai/whisper-large-v3-turbo",
                     aistudio_stt_model="gemini-2.5-flash", voice_enabled=True, voice_cloud_fallback=True, voice_translate_to_english=True,
                     voice_trigger_phrases=DEFAULT_PHRASES, voice_max_bytes=10 * 1024 * 1024, voice_refine_timeout_s=5.0,
                     tts_enabled=False, tts_provider="ollama", tts_model="phonon", tts_base_url="", tts_voice="").items():
        monkeypatch.setattr(settings, k, v, raising=False)


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ split_trigger
@pytest.mark.parametrize("text,want_text,want_trig", [
    ("Fix the login bug in auth.py go ahead now", "Fix the login bug in auth.py", True),
    ("Fix the login bug. Go ahead, now.", "Fix the login bug.", True),
    ("Fix the login bug, GO AHEAD NOW!", "Fix the login bug", True),
    ("refactor utils.py -- process now", "refactor utils.py", True),
    ("add a test, submit now", "add a test", True),
    ("rename it send it now", "rename it", True),
    ("do the thing go ahead", "do the thing", True),
    ("do the thing go ahead now please", "do the thing", True),
    ("do the thing, go   ahead \n now …", "do the thing", True),
    ("isko fix karo ab bhejo", "isko fix karo", True),
    ("isko fix karo, Process Karo.", "isko fix karo", True),
    ("sab theek hai bhej do yaar", "sab theek hai", True),
    ("इसे ठीक करो अब भेजो", "इसे ठीक करो", True),
    ("इसे ठीक करो, प्रोसेस करो।", "इसे ठीक करो", True),
    ("go ahead now go ahead now", "", True),
])
def test_split_trigger_matches_at_the_end(text, want_text, want_trig):
    assert split_trigger(text) == (want_text, want_trig)


@pytest.mark.parametrize("text", [
    "go ahead and refactor the parser",
    "please process now the queue is empty and then stop",
    "I will submit now-ish changes later",
    "fix the bug",
    "ahead now",
    "",
    "   ",
])
def test_split_trigger_ignores_a_phrase_that_is_not_at_the_end(text):
    assert split_trigger(text) == (text.strip(), False)


def test_split_trigger_only_the_phrase_leaves_empty_text():
    assert split_trigger("Go ahead now.") == ("", True)
    assert split_trigger("अब भेजो") == ("", True)


def test_split_trigger_custom_phrases_and_parse_order():
    assert split_trigger("deploy it please engage", "engage, make it so") == ("deploy it please", True)
    assert split_trigger("go ahead now", "make it so") == ("go ahead now", False)
    assert parse_phrases("go ahead, go ahead now")[0] == ["go", "ahead", "now"]       # longest first
    assert parse_phrases(" , ,") == []
    assert split_trigger("a go ahead", []) == ("a go ahead", False)


# ------------------------------------------------------------------ resolver
def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setattr(settings, k, v)


def names(runner=""):
    return [p.name for p in resolver.resolve_chain(runner)]


def test_resolver_runner_mapping(monkeypatch):
    _set(monkeypatch, gemini_api_key="g", cf_account_id="a", cf_api_token="t", openai_compat_base_url="http://x/v1", voice_cloud_fallback=False)
    assert names("aistudio") == ["gemini"]
    assert names("cloudflare") == ["cloudflare"]
    assert names("openai") == ["openai_compat"]
    assert names("ollama") == [] and names("claude_code") == [] and names("nonsense") == []


def test_resolver_chatgpt_codex_needs_the_openai_key(monkeypatch):
    _set(monkeypatch, voice_cloud_fallback=False)
    assert names("chatgpt_codex") == []
    _set(monkeypatch, openai_api_key="sk-x")
    assert names("chatgpt_codex") == ["openai"]


def test_resolver_unconfigured_runner_provider_is_dropped(monkeypatch):
    assert names("aistudio") == []                    # no GEMINI_API_KEY
    assert names("cloudflare") == []


def test_resolver_empty_runner_uses_the_server_default(monkeypatch):
    _set(monkeypatch, llm_runner="aistudio", gemini_api_key="g")
    assert names("") == ["gemini"]
    assert names("ollama") == ["gemini"]              # not mapped, but any configured cloud provider still follows (fallback on)


def test_resolver_local_and_fallback_order(monkeypatch):
    _set(monkeypatch, stt_base_url="http://nuc:8000", gemini_api_key="g", cf_account_id="a", cf_api_token="t")
    assert names("ollama") == ["local", "gemini", "cloudflare"]
    assert names("cloudflare") == ["cloudflare", "local", "gemini"]
    _set(monkeypatch, voice_cloud_fallback=False)
    assert names("ollama") == ["local"]
    assert names("cloudflare") == ["cloudflare", "local"]


def test_resolver_ollama_without_anything_is_browser_mode(monkeypatch):
    cfg = service.voice_config("ollama")
    assert cfg["mode"] == "browser" and cfg["providers"] == []
    _set(monkeypatch, stt_base_url="http://nuc:8000")
    cfg = service.voice_config("ollama")
    assert cfg["mode"] == "server" and cfg["providers"] == ["local"]


def test_resolver_explicit_provider_wins_and_dedupes(monkeypatch):
    _set(monkeypatch, stt_provider="local", stt_base_url="http://nuc:8000", gemini_api_key="g")
    assert names("aistudio") == ["local", "gemini"]
    _set(monkeypatch, stt_provider="openai", openai_api_key="k", openai_compat_base_url="http://x/v1")
    assert names("ollama") == ["openai", "local", "gemini", "openai_compat"]
    _set(monkeypatch, stt_provider="aistudio")
    assert names("ollama")[0] == "gemini"
    for off in ("browser", "off"):
        _set(monkeypatch, stt_provider=off)
        assert names("aistudio") == []


def test_voice_config_payload(monkeypatch):
    _set(monkeypatch, tts_enabled=True, voice_pause_ms=1800)
    cfg = service.voice_config("")
    assert cfg["enabled"] and cfg["pause_ms"] == 1800 and cfg["max_seconds"] == 120 and cfg["translate"] is True
    assert cfg["tts"] == {"enabled": True, "provider": "ollama"}
    assert "go ahead now" in cfg["trigger_phrases"] and "अब भेजो" in cfg["trigger_phrases"]
    _set(monkeypatch, voice_enabled=False)
    assert service.voice_config("")["enabled"] is False


# ------------------------------------------------------------------ refine_to_english
def test_refine_skips_plain_english(monkeypatch):
    def boom(*a):
        raise AssertionError("model must not be called")
    monkeypatch.setattr(refine_mod, "_blocking_refine", boom)
    for t in ("Please add a retry to fetch_user() in api/client.py.", "Do not process now, wait."):
        assert run(refine_mod.refine_to_english(t, "")) == (t, False)


@pytest.mark.parametrize("t,want", [("isko fix karo", True), ("इसे ठीक करो", True), ("ਇਸਨੂੰ ਠੀਕ ਕਰੋ", True), ("this ko ka thing", True),
                                    ("send it to me", False), ("add a test for the parser", False)])
def test_needs_refine_heuristic(t, want):
    assert refine_mod.needs_refine(t) is want


def test_refine_uses_the_model_for_hinglish(monkeypatch):
    seen = {}
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda text, runner: seen.update(text=text, runner=runner) or '  "Fix the bug in auth.py"  ')
    assert run(refine_mod.refine_to_english("auth.py mein bug fix karo", "openai")) == ("Fix the bug in auth.py", True)
    assert seen == {"text": "auth.py mein bug fix karo", "runner": "openai"}


def test_refine_falls_back_when_the_model_raises_times_out_or_is_empty(monkeypatch):
    raw = "auth.py mein bug fix karo"
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda *a: (_ for _ in ()).throw(RuntimeError("model down")))
    assert run(refine_mod.refine_to_english(raw, "")) == (raw, False)
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda *a: "   ")
    assert run(refine_mod.refine_to_english(raw, "")) == (raw, False)
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda *a: time.sleep(0.6) or "late")
    monkeypatch.setattr(settings, "voice_refine_timeout_s", 0.1)

    async def timed():
        t0 = time.time()
        return await refine_mod.refine_to_english(raw, ""), time.time() - t0
    out, took = run(timed())
    assert out == (raw, False) and took < 0.5           # gave up at the bound, did not wait for the model


def test_refine_disabled_by_setting(monkeypatch):
    monkeypatch.setattr(settings, "voice_translate_to_english", False)
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda *a: "x")
    assert run(refine_mod.refine_to_english("fix karo", "")) == ("fix karo", False)


def test_blocking_refine_calls_make_chat_inside_runner_scope(monkeypatch):
    import contextlib
    calls = {}

    class FakeLLM:
        def invoke(self, msgs):
            calls["msgs"] = msgs
            return types.SimpleNamespace(content="Fix the bug")

    fake = types.ModuleType("services.runners")
    fake.resolve_model = lambda m="": "model-x"
    fake.make_chat = lambda model, **kw: calls.update(model=model, kw=kw, scope=calls.get("scope")) or FakeLLM()

    @contextlib.contextmanager
    def scope(name):
        calls["scope"] = name
        yield
    fake.runner_scope = scope
    monkeypatch.setitem(sys.modules, "services.runners", fake)
    assert refine_mod._blocking_refine("bug fix karo", "cloudflare") == "Fix the bug"
    assert calls["scope"] == "cloudflare" and calls["model"] == "model-x" and calls["kw"]["temperature"] == 0
    flat = json.dumps([getattr(m, "content", m) for m in calls["msgs"]], ensure_ascii=False)
    assert "bug fix karo" in flat and "verbatim" in flat


# ------------------------------------------------------------------ providers (MockTransport)
def mock_http(monkeypatch, handler):
    seen = []

    def wrapped(request):
        seen.append(request)
        return handler(request)
    monkeypatch.setattr(prov, "_http", lambda timeout: httpx.Client(transport=httpx.MockTransport(wrapped), timeout=timeout))
    return seen


def test_gemini_request_and_parsing(monkeypatch):
    _set(monkeypatch, gemini_api_key="g-key")
    reply = {"transcript": "auth.py mein bug fix karo ab bhejo", "language": "mixed", "english": "Fix the bug in auth.py", "triggered": True}
    seen = mock_http(monkeypatch, lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(reply)}]}}]}))
    g = prov.GeminiStt()
    assert g.available() == (True, "")
    res = g.transcribe(WAV, "hi")
    req = seen[0]
    assert str(req.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert req.headers["x-goog-api-key"] == "g-key" and "g-key" not in str(req.url)
    body = json.loads(req.content)
    parts = body["contents"][0]["parts"]
    assert parts[1]["inline_data"] == {"mime_type": "audio/wav", "data": base64.b64encode(WAV).decode()}
    assert "verbatim" in parts[0]["text"] and "ab bhejo" in parts[0]["text"] and "hi" in parts[0]["text"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert (res.text, res.transcript, res.language, res.already_english, res.triggered, res.provider) == (
        "Fix the bug in auth.py", "auth.py mein bug fix karo ab bhejo", "mixed", True, True, "gemini")


def test_gemini_tolerates_fenced_and_plain_text_replies(monkeypatch):
    _set(monkeypatch, gemini_api_key="k")
    fenced = "```json\n" + json.dumps({"transcript": "hello", "language": "en", "english": "Hello.", "triggered": "false"}) + "\n```"
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": fenced}]}}]}))
    res = prov.GeminiStt().transcribe(WAV)
    assert res.text == "Hello." and res.triggered is False
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "just words"}]}}]}))
    res = prov.GeminiStt().transcribe(WAV)
    assert res.text == "just words" and res.already_english is False
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"english":"","transcript":""}'}]}}]}))
    with pytest.raises(prov.SttEmpty):
        prov.GeminiStt().transcribe(WAV)
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"promptFeedback": {}}))
    with pytest.raises(prov.SttError):
        prov.GeminiStt().transcribe(WAV)


def test_cloudflare_request_and_parsing(monkeypatch):
    _set(monkeypatch, cf_account_id="acct1", cf_api_token="tok")
    seen = mock_http(monkeypatch, lambda r: httpx.Response(200, json={"success": True, "result": {"text": " namaste ", "transcription_info": {"language": "hi"}}}))
    res = prov.CloudflareStt().transcribe(WAV, "hi-IN")
    req = seen[0]
    assert str(req.url) == "https://api.cloudflare.com/client/v4/accounts/acct1/ai/run/@cf/openai/whisper-large-v3-turbo"
    assert req.headers["authorization"] == "Bearer tok"
    body = json.loads(req.content)
    assert body == {"audio": base64.b64encode(WAV).decode(), "task": "transcribe", "vad_filter": True, "language": "hi"}
    assert (res.text, res.language, res.already_english, res.provider) == ("namaste", "hi", False, "cloudflare")
    prov.CloudflareStt().transcribe(WAV, "auto")
    assert "language" not in json.loads(seen[1].content)


@pytest.mark.parametrize("base,want", [
    ("http://nuc:8000", "http://nuc:8000/v1/audio/transcriptions"),
    ("http://nuc:8000/", "http://nuc:8000/v1/audio/transcriptions"),
    ("http://nuc:8000/v1", "http://nuc:8000/v1/audio/transcriptions"),
    ("https://api.groq.com/openai/v1/", "https://api.groq.com/openai/v1/audio/transcriptions"),
    ("https://h/v1/audio/transcriptions", "https://h/v1/audio/transcriptions"),
])
def test_openai_compat_endpoint(base, want):
    assert prov.OpenAICompatStt.endpoint(base) == want


@pytest.mark.parametrize("kind,cfg,url,model,auth", [
    ("local", dict(stt_base_url="http://nuc:8000", stt_api_key="", stt_model="Systran/faster-whisper-large-v3-turbo"),
     "http://nuc:8000/v1/audio/transcriptions", "Systran/faster-whisper-large-v3-turbo", None),
    ("openai", dict(openai_api_key="sk-1"), "https://api.openai.com/v1/audio/transcriptions", "gpt-4o-transcribe", "Bearer sk-1"),
    ("openai_compat", dict(openai_compat_base_url="https://api.groq.com/openai/v1", openai_compat_api_key="gk"),
     "https://api.groq.com/openai/v1/audio/transcriptions", "whisper-large-v3-turbo", "Bearer gk"),
])
def test_openai_compat_request_and_parsing(monkeypatch, kind, cfg, url, model, auth):
    _set(monkeypatch, **cfg)
    seen = mock_http(monkeypatch, lambda r: httpx.Response(200, json={"text": " Fix the bug. ", "language": "english"}))
    p = prov.OpenAICompatStt(kind)
    assert p.available()[0]
    res = p.transcribe(WAV, "pa")
    req = seen[0]
    assert str(req.url) == url and req.headers.get("authorization") == auth
    assert req.headers["content-type"].startswith("multipart/form-data")
    raw = req.content
    for needle in (b'name="file"; filename="audio.wav"', b'name="model"', model.encode(), b'name="response_format"', b"json", b'name="language"', b"pa",
                   b'name="prompt"', b"Hinglish", WAV[:12]):
        assert needle in raw, needle
    assert (res.text, res.provider, res.already_english) == ("Fix the bug.", kind, False)
    prov.OpenAICompatStt(kind).transcribe(WAV, "auto")
    assert b'name="language"' not in seen[1].content


def test_provider_availability_reasons():
    assert prov.GeminiStt().available()[0] is False
    assert prov.CloudflareStt().available()[0] is False
    assert "STT_BASE_URL" in prov.OpenAICompatStt("local").available()[1]
    assert "OPENAI_API_KEY" in prov.OpenAICompatStt("openai").available()[1]
    assert "OPENAI_COMPAT_BASE_URL" in prov.OpenAICompatStt("openai_compat").available()[1]


def test_provider_http_errors_become_stt_error_without_the_key(monkeypatch):
    _set(monkeypatch, openai_api_key="sk-SECRET", gemini_api_key="g-SECRET", cf_account_id="a", cf_api_token="t-SECRET")
    mock_http(monkeypatch, lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    for p in (prov.OpenAICompatStt("openai"), prov.GeminiStt(), prov.CloudflareStt()):
        with pytest.raises(prov.SttError) as e:
            p.transcribe(WAV)
        assert "401" in str(e.value) and "SECRET" not in str(e.value)

    def boom(r):
        raise httpx.ConnectError("refused", request=r)
    mock_http(monkeypatch, boom)
    with pytest.raises(prov.SttError):
        prov.OpenAICompatStt("openai").transcribe(WAV)
    mock_http(monkeypatch, lambda r: httpx.Response(200, json={"text": "  "}))
    with pytest.raises(prov.SttEmpty):
        prov.OpenAICompatStt("openai").transcribe(WAV)


# ------------------------------------------------------------------ service: fail-over, triggers, refinement
class Fake(prov.SttProvider):
    def __init__(self, name, result=None, exc=None):
        self.name, self.result, self.exc, self.calls = name, result, exc, 0

    def available(self):
        return True, ""

    def transcribe(self, wav, lang_hint="auto"):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.result


def chain(monkeypatch, *ps):
    monkeypatch.setattr(service, "resolve_chain", lambda runner="": list(ps))


def test_failover_to_the_next_provider(monkeypatch):
    a = Fake("gemini", exc=prov.SttError("gemini: HTTP 429: quota"))
    b = Fake("local", prov.SttResult(text="Fix the bug", provider="local", transcript="Fix the bug"))
    chain(monkeypatch, a, b)
    out = run(service.transcribe_audio(WAV, "aistudio", "auto"))
    assert out["ok"] and out["provider"] == "local" and out["text"] == "Fix the bug" and out["triggered"] is False
    assert out["errors"] == ["gemini: HTTP 429: quota"] and a.calls == 1 and b.calls == 1 and out["refined"] is False


def test_all_providers_failing_answers_ok_false_with_browser_fallback(monkeypatch):
    chain(monkeypatch, Fake("a", exc=prov.SttError("a: boom")), Fake("b", exc=RuntimeError("weird")))
    out = run(service.transcribe_audio(WAV, "", "auto"))
    assert out["ok"] is False and out["fallback"] == "browser" and len(out["errors"]) == 2 and out["errors"][0] == "a: boom"
    chain(monkeypatch)
    out = run(service.transcribe_audio(WAV, "", "auto"))
    assert out["ok"] is False and out["fallback"] == "browser"


def test_silence_is_not_a_failure(monkeypatch):
    b = Fake("b", prov.SttResult(text="x"))
    chain(monkeypatch, Fake("a", exc=prov.SttEmpty("a: empty transcript")), b)
    out = run(service.transcribe_audio(WAV, "", "auto"))
    assert out["ok"] and out["text"] == "" and out["triggered"] is False and b.calls == 0


def test_trigger_is_detected_on_the_raw_transcript_before_refining(monkeypatch):
    seen = {}
    monkeypatch.setattr(service, "refine_to_english", lambda text, runner: _async((seen.update(text=text), ("Fix the bug in auth.py", True))[1]))
    chain(monkeypatch, Fake("local", prov.SttResult(text="auth.py mein bug fix karo ab bhejo", transcript="auth.py mein bug fix karo ab bhejo", provider="local")))
    out = run(service.transcribe_audio(WAV, "", "hi"))
    assert seen["text"] == "auth.py mein bug fix karo"                      # the translator never sees the command
    assert out["text"] == "Fix the bug in auth.py" and out["triggered"] is True and out["refined"] is True
    assert out["transcript"] == "auth.py mein bug fix karo ab bhejo"


def test_trigger_in_the_refined_english_is_stripped_too(monkeypatch):
    monkeypatch.setattr(service, "refine_to_english", lambda text, runner: _async(("Fix the bug. Go ahead now.", True)))
    chain(monkeypatch, Fake("local", prov.SttResult(text="bug fix karo", provider="local")))
    out = run(service.transcribe_audio(WAV, "", "auto"))
    assert out["text"] == "Fix the bug." and out["triggered"] is True


def test_gemini_result_is_not_refined_again_and_reports_trigger(monkeypatch):
    monkeypatch.setattr(service, "refine_to_english", lambda *a: (_ for _ in ()).throw(AssertionError("already English")))
    res = prov.SttResult(text="Fix the bug", transcript="bug fix karo", language="hi", provider="gemini", already_english=True, triggered=True)
    chain(monkeypatch, Fake("gemini", res))
    out = run(service.transcribe_audio(WAV, "aistudio", "auto"))
    assert out["text"] == "Fix the bug" and out["triggered"] is True and out["language"] == "hi" and out["provider"] == "gemini"
    res = prov.SttResult(text="Fix the bug, go ahead now", transcript="x", provider="gemini", already_english=True, triggered=False)
    chain(monkeypatch, Fake("gemini", res))
    out = run(service.transcribe_audio(WAV, "aistudio", "auto"))
    assert out["text"] == "Fix the bug" and out["triggered"] is True


def test_only_a_trigger_phrase_gives_empty_text(monkeypatch):
    chain(monkeypatch, Fake("local", prov.SttResult(text="Go ahead now.", provider="local")))
    out = run(service.transcribe_audio(WAV, "", "auto"))
    assert out["ok"] and out["text"] == "" and out["triggered"] is True


def test_refine_text_for_browser_transcripts(monkeypatch):
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda text, runner: "Fix the bug")
    out = run(service.refine_text("bug fix karo, ab bhejo", "no-such-runner"))
    assert out["ok"] and out["text"] == "Fix the bug" and out["triggered"] is True and out["provider"] == "browser"
    out = run(service.refine_text("add a unit test", ""))
    assert out["text"] == "add a unit test" and out["triggered"] is False and out["refined"] is False


def _async(v):
    async def f():
        return v
    return f()


# ------------------------------------------------------------------ HTTP routes
@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from views.voice_routes import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_route_config(client, monkeypatch):
    r = client.get("/api/v1/voice/config", params={"runner": "ollama"})
    assert r.status_code == 200 and r.json()["mode"] == "browser" and r.json()["enabled"] is True
    _set(monkeypatch, stt_base_url="http://nuc:8000")
    j = client.get("/api/v1/voice/config").json()
    assert j["mode"] == "server" and j["providers"] == ["local"] and j["pause_ms"] == 2500


def test_route_transcribe_happy_path(client, monkeypatch):
    chain(monkeypatch, Fake("local", prov.SttResult(text="Fix the bug, go ahead now", provider="local", language="en")))
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")}, data={"runner": "ollama", "lang": "en"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and j["text"] == "Fix the bug" and j["triggered"] is True and j["provider"] == "local" and j["language"] == "en"
    assert set(j) >= {"ok", "text", "transcript", "language", "provider", "triggered", "refined", "errors"}


def test_route_transcribe_size_limit_413(client, monkeypatch):
    _set(monkeypatch, voice_max_bytes=100)
    chain(monkeypatch, Fake("local", prov.SttResult(text="x")))
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", b"\0" * 101, "audio/wav")})
    assert r.status_code == 413
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", b"\0" * 100, "audio/wav")})
    assert r.status_code == 200


def test_route_transcribe_all_fail_is_200_with_fallback(client, monkeypatch):
    chain(monkeypatch, Fake("gemini", exc=prov.SttError("gemini: HTTP 500")))
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")}, data={"runner": "bogus"})
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["fallback"] == "browser" and r.json()["errors"]
    monkeypatch.setattr(service, "resolve_chain", lambda runner="": (_ for _ in ()).throw(RuntimeError("bug")))
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 200 and r.json()["ok"] is False


def test_route_transcribe_disabled(client, monkeypatch):
    _set(monkeypatch, voice_enabled=False)
    r = client.post("/api/v1/voice/transcribe", files={"audio": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 404


def test_route_refine(client, monkeypatch):
    monkeypatch.setattr(refine_mod, "_blocking_refine", lambda text, runner: "Fix the bug")
    r = client.post("/api/v1/voice/refine", json={"text": "bug fix karo ab bhejo", "runner": ""})
    assert r.status_code == 200 and r.json()["text"] == "Fix the bug" and r.json()["triggered"] is True


def test_route_speak(client, monkeypatch):
    r = client.post("/api/v1/voice/speak", json={"text": "hello"})
    assert r.status_code == 503 and r.json()["fallback"] == "browser"          # TTS off by default
    _set(monkeypatch, tts_enabled=True, ollama_base_url="http://localhost:11434")
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, content=b"RIFFwav", headers={"content-type": "audio/wav"})
    real = httpx.Client
    monkeypatch.setattr(service.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = client.post("/api/v1/voice/speak", json={"text": "x" * 5000})
    assert r.status_code == 200 and r.content == b"RIFFwav" and r.headers["content-type"].startswith("audio/wav")
    assert str(seen[0].url) == "http://localhost:11434/v1/audio/speech"
    body = json.loads(seen[0].content)
    assert body["model"] == "phonon" and len(body["input"]) == 4000 and body["response_format"] == "wav" and "voice" not in body
    monkeypatch.setattr(service.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(lambda q: httpx.Response(500, text="no")), **kw))
    assert client.post("/api/v1/voice/speak", json={"text": "hi"}).status_code == 503
