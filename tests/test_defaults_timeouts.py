"""Default-model semantics on the non-Ollama runners, readable timeouts / transport errors, per-runner timeout settings, telemetry's
requested_model, AI Studio's reasoning_effort, and the UI wiring (task timer, failure bubble). No network, no provider packages:
exceptions are fakes named like the real ones (the classifier matches class names along the MRO), FastAPI is stubbed here so the
HTTP mapping is checked through the response-builder functions and a raw ASGI call."""
import asyncio
import json
import pathlib
import shutil
import subprocess
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import EngineSettings, settings
from services import action_log, llm_errors, model_calls, rag_graph
from services.llm_errors import (InternalErrorMiddleware, ModelQuotaError, ModelTimeoutError, ModelUnavailableError, as_timeout_error,
                                 as_unavailable_error, fallback_for, internal_payload, is_timeout, is_unreachable, timeout_payload,
                                 unavailable_payload, with_model_fallback)
from services.model_calls import announce_model, reasoning_model, role_model
from services.runner_info import models_payload, runners_payload
from services.runners import get_runner, make_chat, resolve_model, runner_scope, runner_status
from services.runners import aistudio_runner as asr
from services.runners.aistudio_runner import GoogleAIStudioRunner
from services.runners.openai_runner import OpenAIRunner
from services.telemetry import StdoutTelemetryCallback

run = asyncio.run
OLLAMA_DEFAULT = "gemma4:31b-cloud"
FLASH, G31, G26 = "gemini-2.5-flash", "gemma-4-31b-it", "gemma-4-26b-a4b-it"
CF = "@cf/google/gemma-4-26b-a4b-it"
ROOT = pathlib.Path(__file__).resolve().parent.parent


class Rec:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in dict(llm_runner="ollama", llm_model=OLLAMA_DEFAULT, llm_fallback_model="", llm_quota_cooldown_s=300.0,
                     patcher_model="", change_escalation_model="", diagnoser_model="", vision_model="",
                     cf_account_id="acc", cf_api_token="tok", cf_llm_model=CF, cf_vision_model=CF, cf_ai_gateway_url="",
                     cf_timeout_s=120.0, cf_max_retries=2,
                     gemini_api_key="AIza-key", aistudio_llm_model=FLASH, aistudio_vision_model="gemini-2.5-pro",
                     aistudio_models=f"{G31},{G26}", aistudio_list_all_models=False, aistudio_timeout_s=180.0, aistudio_max_retries=1,
                     aistudio_reasoning_effort="", aistudio_fold_system="auto", aistudio_json_mode="auto",
                     compat_timeout_s=120.0, compat_max_retries=2, openai_compat_timeout_s=None, openai_compat_max_retries=None,
                     openai_compat_base_url="http://localhost:8000/v1", openai_compat_api_key="", openai_compat_models="",
                     openai_compat_llm_model="my-served-model", openai_compat_list_all_models=False).items():
        monkeypatch.setattr(settings, k, v)
    llm_errors.reset_quota_state()
    yield
    llm_errors.reset_quota_state()


# ------------------------------------------------------------------ A: default-model mapping per runner
@pytest.mark.parametrize("given, expected", [
    ("", FLASH), ("   ", FLASH), (OLLAMA_DEFAULT, FLASH),                      # empty / the global default -> AISTUDIO_LLM_MODEL (NOT the table's Gemma)
    ("gemma4:26b", G26), ("gemma4:31b", G31),                                  # other explicit Ollama names: the table
    ("qwen2.5-coder:7b", FLASH),                                               # other Ollama tag: the default
    (G31, G31), ("models/" + G26, G26), ("gemini-2.5-pro", "gemini-2.5-pro"),  # native ids are never remapped
])
def test_aistudio_default_semantics(given, expected):
    assert get_runner("aistudio").map_model(given) == expected


def test_aistudio_vision_default_and_table_is_skipped_for_images():
    r = get_runner("aistudio")
    assert r.map_model("", vision=True) == "gemini-2.5-pro" and r.map_model(OLLAMA_DEFAULT, vision=True) == "gemini-2.5-pro"
    assert r.map_model("gemma4:26b", vision=True) == "gemini-2.5-pro"


@pytest.mark.parametrize("given, expected", [
    ("", CF), (OLLAMA_DEFAULT, CF), ("qwen2.5-coder:7b", CF), ("gemma4:26b", CF),
    ("@cf/google/gemma-3-12b-it", "@cf/google/gemma-3-12b-it"),
])
def test_cloudflare_default_semantics(given, expected):
    assert get_runner("cloudflare").map_model(given) == expected


@pytest.mark.parametrize("given, expected", [
    ("", "my-served-model"), (OLLAMA_DEFAULT, "my-served-model"), ("qwen2.5-coder:7b", "my-served-model"),
    ("meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"), ("gpt-4o", "gpt-4o"),
])
def test_generic_openai_default_semantics(given, expected):
    assert get_runner("openai").map_model(given) == expected


def test_a_native_global_default_is_kept(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "gemini-2.5-pro")       # not an Ollama tag: an explicit native id
    assert get_runner("aistudio").map_model("gemini-2.5-pro") == "gemini-2.5-pro"


def test_ollama_is_untouched():
    r = get_runner("ollama")
    assert r.map_model(OLLAMA_DEFAULT) == OLLAMA_DEFAULT and r.map_model("qwen3:32b") == "qwen3:32b"
    assert resolve_model() == OLLAMA_DEFAULT


def test_make_chat_calls_the_runners_default_when_nothing_is_picked():
    with runner_scope("aistudio"):
        llm = make_chat(settings.llm_model, chat_cls=Rec)
        assert llm.kw["model"] == FLASH and llm.kw["timeout"] == 180.0 and llm.kw["max_retries"] == 1


def test_models_payload_default_is_the_runners_configured_default():
    for runner, expected in (("aistudio", FLASH), ("cloudflare", CF), ("openai", "my-served-model"), ("ollama", OLLAMA_DEFAULT)):
        p = run(models_payload(runner))
        assert p["default"] == expected and expected in p["models"] and p["runner"] == runner


# ------------------------------------------------------------------ A: role models
def test_role_models_that_are_ollama_tags_mean_the_runner_default():
    with runner_scope("aistudio"):
        assert role_model("qwen3-coder:30b") == "" and role_model("") == ""
        assert role_model("gemini-2.5-pro") == "gemini-2.5-pro"
        settings.patcher_model = "qwen3-coder:30b"
        assert resolve_model(reasoning_model(settings.patcher_model)) == FLASH
        settings.patcher_model = "gemini-2.5-pro"
        assert reasoning_model(settings.patcher_model) == "gemini-2.5-pro"
    settings.patcher_model = "qwen3-coder:30b"
    assert reasoning_model(settings.patcher_model) == "qwen3-coder:30b"       # Ollama: the tag is the model
    with runner_scope("cloudflare"):
        assert role_model("qwen3-coder:30b") == "" and role_model("@cf/google/gemma-3-12b-it") == "@cf/google/gemma-3-12b-it"


def test_the_users_pick_beats_the_role_model():
    with runner_scope("aistudio"), model_calls.model_scope("gemini-2.5-pro"):
        assert reasoning_model("qwen3-coder:30b") == "gemini-2.5-pro"


def test_vision_role_resolves_to_the_vision_default():
    settings.vision_model = "llava:13b"
    with runner_scope("aistudio"):
        assert resolve_model(settings.vision_model or settings.llm_model, vision=True) == "gemini-2.5-pro"


# ------------------------------------------------------------------ A: no same-model fallback (every runner)
@pytest.mark.parametrize("runner", ["aistudio", "cloudflare", "openai"])
def test_a_fallback_that_resolves_to_the_same_model_is_dropped(monkeypatch, runner):
    with runner_scope(runner):
        for fb in ("qwen2.5-coder:7b", OLLAMA_DEFAULT):
            monkeypatch.setattr(settings, "llm_fallback_model", fb)
            assert fallback_for(OLLAMA_DEFAULT) == "" and fallback_for("") == ""


def test_a_different_fallback_is_kept():
    settings.llm_fallback_model = "gemma4:26b"
    with runner_scope("aistudio"):
        assert fallback_for(OLLAMA_DEFAULT) == "gemma4:26b"               # maps to gemma-4-26b-a4b-it, not Flash
    settings.llm_fallback_model = "qwen2.5-coder:7b"
    assert fallback_for(OLLAMA_DEFAULT) == "qwen2.5-coder:7b"             # Ollama: a different local model


def test_quota_with_a_same_model_fallback_raises_instead_of_calling_it_again():
    class Status(Exception):
        status_code = 429

    settings.llm_fallback_model = "qwen2.5-coder:7b"
    seen = []

    async def call(m):
        seen.append(m)
        raise Status("RESOURCE_EXHAUSTED quota exceeded")
    with runner_scope("aistudio"), pytest.raises(ModelQuotaError):
        run(with_model_fallback(call, OLLAMA_DEFAULT))
    assert seen == [OLLAMA_DEFAULT]


# ------------------------------------------------------------------ B: classification by class name
class APIConnectionError(Exception):            # openai.APIConnectionError
    pass


class APITimeoutError(APIConnectionError):      # openai.APITimeoutError subclasses the connection error
    pass


class OpenAITimeoutError(APITimeoutError):
    pass


class TimeoutException(Exception):              # httpx
    pass


class ReadTimeout(TimeoutException):
    pass


class ConnectTimeout(TimeoutException):
    pass


class TransportError(Exception):
    pass


class ConnectError(TransportError):             # httpx.ConnectError
    pass


@pytest.mark.parametrize("exc", [APITimeoutError("Request timed out."), OpenAITimeoutError("Request timed out."), ReadTimeout("read"),
                                 ConnectTimeout("connect"), TimeoutError("t"), asyncio.TimeoutError()])
def test_timeouts_are_recognised_by_class_name(exc):
    assert is_timeout(exc) and not is_unreachable(exc)
    with runner_scope("aistudio"):
        assert isinstance(as_timeout_error(exc, OLLAMA_DEFAULT), ModelTimeoutError)


@pytest.mark.parametrize("exc", [APIConnectionError("Connection error."), ConnectError("refused"), ConnectionRefusedError("refused")])
def test_connection_failures_are_unavailable_not_timeouts(exc):
    assert is_unreachable(exc) and not is_timeout(exc)
    assert isinstance(as_unavailable_error(exc, OLLAMA_DEFAULT), ModelUnavailableError)


def test_a_wrapped_timeout_is_found_and_other_errors_are_not():
    try:
        try:
            raise ReadTimeout("slow")
        except ReadTimeout as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        assert is_timeout(outer)
    assert not is_timeout(ValueError("x")) and not is_unreachable(ValueError("x"))
    assert as_timeout_error(ValueError("x"), "m") is None and as_unavailable_error(ValueError("x"), "m") is None


def test_with_model_fallback_raises_a_readable_timeout_with_the_runners_facts():
    seen = []

    async def call(m):
        seen.append(m)
        raise OpenAITimeoutError("Request timed out.")
    settings.llm_fallback_model = "gemma4:26b"                         # a timeout never triggers the fallback (it would wait as long)
    with runner_scope("aistudio"), pytest.raises(ModelTimeoutError) as ei:
        run(with_model_fallback(call, OLLAMA_DEFAULT))
    e = ei.value
    assert seen == [OLLAMA_DEFAULT]
    assert str(e) == (f"Google AI Studio did not answer within 180s (model {FLASH}, 2 attempt(s)). "
                      "Try a faster model (e.g. a Gemini Flash model), Lean mode, or raise AISTUDIO_TIMEOUT_S.")
    assert (e.runner, e.model, e.timeout_s, e.attempts) == ("aistudio", FLASH, 180.0, 2)


def test_timeout_message_names_each_runners_setting():
    with runner_scope("cloudflare"):
        assert "raise CF_TIMEOUT_S." in str(ModelTimeoutError("x")) and "Cloudflare Workers AI did not answer within 120s" in str(ModelTimeoutError("x"))
        assert "3 attempt(s)" in str(ModelTimeoutError("x"))
    with runner_scope("openai"):
        assert "raise OPENAI_COMPAT_TIMEOUT_S." in str(ModelTimeoutError("")) and "my-served-model" in str(ModelTimeoutError(""))
    ollama = str(ModelTimeoutError("qwen3:32b"))
    assert "Ollama (local) did not answer in time (model qwen3:32b, 1 attempt(s))" in ollama and "raise" not in ollama


def test_with_model_fallback_maps_a_connection_failure_to_unavailable():
    async def call(m):
        raise ConnectError("All connection attempts failed")
    with runner_scope("openai"), pytest.raises(ModelUnavailableError) as ei:
        run(with_model_fallback(call, ""))
    assert "OPENAI_COMPAT_BASE_URL" in str(ei.value) and "my-served-model" in str(ei.value)


def test_a_timeout_reaches_the_user_even_when_every_model_times_out():
    async def call(m):
        raise APITimeoutError("Request timed out.")
    with runner_scope("aistudio"), pytest.raises(ModelTimeoutError):
        run(with_model_fallback(call, FLASH))


def test_cancellation_and_other_errors_pass_through():
    async def boom(m):
        raise KeyError("bug")
    with pytest.raises(KeyError):
        run(with_model_fallback(boom, "m"))

    async def cancelled(m):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        run(with_model_fallback(cancelled, "m"))


def test_the_answer_step_does_not_retry_a_timeout_as_a_single_shot(monkeypatch):
    """A hung code-agent call must surface at once: the single-shot fallback would wait another full timeout."""
    calls = []

    async def answer(*a, **k):
        raise ModelTimeoutError("m")

    class Boom:
        def __init__(self, *a, **k):
            calls.append("single-shot")
    monkeypatch.setattr(rag_graph, "ChatOllama", Boom)
    monkeypatch.setattr(settings, "ask_agent", True)
    m = rag_graph.RAGWorkflowManager.__new__(rag_graph.RAGWorkflowManager)
    m.code_agent = types.SimpleNamespace(answer=answer)
    monkeypatch.setattr(m, "_system_prompt", lambda state: "S", raising=False)
    state = {"repo_id": "r1", "thread_id": "t1", "context_str": "", "attachment_context": "", "web_context": "", "web_footer": "",
             "web_mode": "off", "web_query": "", "web_note": "", "retrieval_meta": {}, "messages": [HumanMessage(content="q")]}
    with pytest.raises(ModelTimeoutError):
        run(m._generate_answer_inner(state))
    assert calls == []


# ------------------------------------------------------------------ B: HTTP payloads
def test_timeout_payload():
    with runner_scope("aistudio"):
        status, body = timeout_payload(ModelTimeoutError(OLLAMA_DEFAULT))
    assert status == 504 and body["kind"] == "timeout" and body["runner"] == "aistudio" and body["model"] == FLASH
    assert body["timeout_s"] == 180.0 and body["detail"].startswith("Google AI Studio did not answer within 180s")


def test_unavailable_payload():
    with runner_scope("aistudio"):
        status, body = unavailable_payload(ModelUnavailableError(""))
    assert status == 503 and body["kind"] == "unavailable" and body["runner"] == "aistudio" and body["model"] == FLASH
    assert "could not be reached" in body["detail"]


def test_internal_payload_is_short_and_typed():
    status, body = internal_payload(KeyError("missing\nkey " + "x" * 500))
    assert status == 500 and body["kind"] == "internal" and body["detail"].startswith("KeyError: ") and len(body["detail"]) < 330 and "\n" not in body["detail"]
    assert internal_payload(RuntimeError()) == (500, {"detail": "RuntimeError", "kind": "internal"})


def test_api_view_registers_the_handlers_and_the_safety_net():
    src = (ROOT / "views" / "api_view.py").read_text()
    for needle in ("add_exception_handler(ModelTimeoutError, timeout_error_response)",
                   "add_exception_handler(ModelUnavailableError, unavailable_error_response)", "add_middleware(InternalErrorMiddleware)"):
        assert needle in src


def _asgi(app, path="/api/v1/chat"):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)
    run(InternalErrorMiddleware(app)({"type": "http", "method": "POST", "path": path}, receive, send))
    return sent


def test_the_safety_net_turns_an_escaped_exception_into_json_500(caplog):
    async def app(scope, receive, send):
        raise ValueError("kaboom")
    with caplog.at_level("ERROR", logger="coderag.errors"):
        sent = _asgi(app)
    assert sent[0]["status"] == 500 and (b"content-type", b"application/json") in sent[0]["headers"]
    assert json.loads(sent[1]["body"]) == {"detail": "ValueError: kaboom", "kind": "internal"}
    logged = [r for r in caplog.records if r.name == "coderag.errors"]
    assert len(logged) == 1 and logged[0].exc_info and "kaboom" in str(logged[0].exc_info[1])      # the traceback is logged exactly once


def test_the_safety_net_leaves_handled_responses_streams_and_cancellation_alone():
    async def ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})
    assert [m.get("status") for m in _asgi(ok)] == [404, None]

    async def half(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ValueError("after the headers")
    with pytest.raises(ValueError):
        _asgi(half)

    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _asgi(cancelled)


def test_the_safety_net_ignores_non_http_scopes():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])
    run(InternalErrorMiddleware(app)({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


# ------------------------------------------------------------------ B: per-runner timeouts and retries
def test_runners_payload_exposes_timeout_and_retries():
    by = {r["name"]: r for r in runners_payload()["runners"]}
    assert (by["aistudio"]["timeout_s"], by["aistudio"]["max_retries"]) == (180.0, 1)
    assert (by["cloudflare"]["timeout_s"], by["cloudflare"]["max_retries"]) == (120.0, 2)
    assert (by["openai"]["timeout_s"], by["openai"]["max_retries"]) == (120.0, 2)           # defaults to compat_*
    assert (by["ollama"]["timeout_s"], by["ollama"]["max_retries"]) == (None, 0)
    settings.openai_compat_timeout_s, settings.openai_compat_max_retries = 45.0, 0
    by = {r["name"]: r for r in runner_status()}
    assert (by["openai"]["timeout_s"], by["openai"]["max_retries"]) == (45.0, 0)


def test_defaults_of_the_new_settings():
    f = EngineSettings.model_fields
    assert f["aistudio_timeout_s"].default == 180.0 and f["aistudio_max_retries"].default == 1
    assert f["openai_compat_timeout_s"].default is None and f["openai_compat_max_retries"].default is None
    assert f["aistudio_reasoning_effort"].default == "" and f["compat_max_retries"].default == 2 and f["cf_max_retries"].default == 2


def test_each_runner_hands_its_own_timeout_to_the_client():
    cases = (("aistudio", 180.0, 1), ("cloudflare", 120.0, 2), ("openai", 120.0, 2))
    for name, t, n in cases:
        with runner_scope(name):
            kw = make_chat("", chat_cls=Rec).kw
            assert (kw["timeout"], kw["max_retries"]) == (t, n)
    settings.aistudio_timeout_s, settings.aistudio_max_retries = 30.0, 0
    with runner_scope("aistudio"):
        assert (make_chat("", chat_cls=Rec).kw["timeout"], make_chat("", chat_cls=Rec).kw["max_retries"]) == (30.0, 0)


# ------------------------------------------------------------------ C: reasoning_effort only when configured
def _kw(runner="aistudio", **chat_kw):
    with runner_scope(runner):
        return make_chat(FLASH if runner == "aistudio" else "", chat_cls=Rec, **chat_kw).kw


def test_reasoning_effort_is_not_sent_by_default():
    assert "reasoning_effort" not in _kw() and "reasoning_effort" not in _kw(reasoning=False)


@pytest.mark.parametrize("value", ["none", "minimal", "low", "medium", "high", " LOW "])
def test_a_configured_effort_is_sent_on_every_call(value):
    settings.aistudio_reasoning_effort = value
    assert _kw()["reasoning_effort"] == value.strip().lower() and _kw(reasoning=True)["reasoning_effort"] == value.strip().lower()


def test_auto_follows_ask_reasoning_and_junk_is_ignored():
    settings.aistudio_reasoning_effort = "auto"
    assert _kw(reasoning=False)["reasoning_effort"] == "none" and _kw(reasoning="high")["reasoning_effort"] == "high"
    assert "reasoning_effort" not in _kw() and "reasoning_effort" not in _kw(reasoning=True)
    settings.aistudio_reasoning_effort = "bogus"
    assert "reasoning_effort" not in _kw(reasoning=False)


def test_other_runners_never_get_reasoning_effort():
    settings.aistudio_reasoning_effort = "none"
    assert "reasoning_effort" not in _kw("cloudflare") and "reasoning_effort" not in _kw("openai")


# ------------------------------------------------------------------ telemetry: requested_model + actual model
def _finish(cb):
    gen = type("G", (), {"message": None, "generation_info": {}, "text": "hello"})()
    resp = type("Resp", (), {"generations": [[gen]], "llm_output": {"token_usage": {"prompt_tokens": 7, "completion_tokens": 9}}})()
    cb.start_time, cb.prompts = 0.0, ["p"]
    cb.on_llm_end(resp)


def test_telemetry_row_has_requested_and_actual_model():
    rows = []
    with runner_scope("aistudio"):
        cb = StdoutTelemetryCallback(model_name=OLLAMA_DEFAULT, repo_id="r", phase="RAG_GENERATION", emitter=lambda d: rows.append(json.loads(d)))
        make_chat(OLLAMA_DEFAULT, chat_cls=Rec, callbacks=[cb])             # relabels the callback exactly like rag_graph does
        _finish(cb)
    assert rows[0]["model"] == FLASH and rows[0]["requested_model"] == OLLAMA_DEFAULT and rows[0]["runner"] == "aistudio"


def test_model_calls_build_passes_both_names():
    with runner_scope("aistudio"):
        cbs = model_calls._callbacks(FLASH, "r", "PATCH", requested=OLLAMA_DEFAULT)
        assert cbs[0].model_name == FLASH and cbs[0].requested_model == OLLAMA_DEFAULT
        rows = []
        cbs[0].emitter = lambda d: rows.append(json.loads(d))
        _finish(cbs[0])
        assert rows[0]["requested_model"] == OLLAMA_DEFAULT and rows[0]["model"] == FLASH
    cb = StdoutTelemetryCallback("m", "r", emitter=lambda d: rows.append(json.loads(d)))
    _finish(cb)
    assert rows[-1]["requested_model"] == "m"                              # no separate request: same as the model


def test_request_start_event_names_the_resolved_model():
    rid = "rid-" + "resolve-1"
    with runner_scope("aistudio"), action_log.request_scope(rid, "t1", repo_id="r1"):
        announce_model()
        announce_model()                                                   # once per request
        with model_calls.model_scope(G26):
            announce_model()
    evs = [e for e in action_log.recent(request_id=rid) if e.get("action") == "model_resolved"]
    assert len(evs) == 1
    assert evs[0]["detail"] == f"model resolved: runner=aistudio picked=Default -> {FLASH}" and evs[0]["model"] == FLASH
    rid2 = "rid-resolve-2"
    with runner_scope("aistudio"), model_calls.model_scope(G26), action_log.request_scope(rid2, "t1", repo_id="r1"):
        announce_model()
    ev = [e for e in action_log.recent(request_id=rid2) if e.get("action") == "model_resolved"][0]
    assert ev["detail"] == f"model resolved: runner=aistudio picked={G26} -> {G26}"


# ------------------------------------------------------------------ UI wiring (static checks; the browser itself is not run here)
def _read(*parts):
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def test_the_ui_shows_every_failure_and_has_the_task_timer():
    js, html = _read("static", "js", "app.js"), _read("templates", "index.html")
    for needle in ("async function describeFailure(res)", "function showFailure(f)", "function errorBubbleHtml(", "showFailure(failureOf(err))",
                   "function taskStart()", "function taskEnd(outcome)", "function taskStep(ev)", "ev.action === 'llm_call'", "LS.get('taskTimer', true)",
                   "async function stopChat()", "per model call", "Done in ", "Failed after ", "S.runnerTimeouts", "(asked "):
        assert needle in js, needle
    assert js.count("fetch('/api/v1/chat'") == 1 and js.count("fetch('/api/v1/chat/attach'") == 1       # both send paths share the same error handling
    assert "throw new Error((await res.json()" not in js
    assert 'id="task-timer-toggle"' in html and "Show task timer" in html
    assert "app.js?v=6" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(ROOT / "static" / "js" / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
