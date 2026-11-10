"""Ollama Cloud answers 402 "Add usage credits or subscribe ..." when the credits/free usage of a `*-cloud` model run out.

That is the provider refusing the call, not an app bug. Expected behaviour (services/llm_errors.py):
  * no fallback model set  -> one clear ModelQuotaError (the API maps it to HTTP 402 + readable `detail`), never a raw traceback;
  * ENGINE_LLM_FALLBACK_MODEL set -> the call is retried once on that local model, later calls skip the cloud model for a while,
    and the answer carries a "Model note" line;
  * anything else (network, bad model name, a busy local Ollama answering 429) is NOT treated as a quota problem.
No Ollama, Docker or network: the model is scripted.
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from services import llm_errors, model_calls, rag_graph
from services.llm_errors import (ModelQuotaError, as_quota_error, fallback_note, fallback_scope, is_cloud_model,
                                 quota_error_response, quota_status, with_model_fallback)

run = asyncio.run
CLOUD, LOCAL = "gemma4:31b-cloud", "qwen2.5-coder:7b"
OLLAMA_402 = "Add usage credits or subscribe and save up to 66%: https://ollama.com/plans (ref: bf884f69) (status code: 402)"


class ResponseError(Exception):
    """Same shape as ollama.ResponseError: .error and .status_code."""
    def __init__(self, error, status_code=-1):
        super().__init__(error)
        self.error, self.status_code = error, status_code

    def __str__(self):
        return f"{self.error} (status code: {self.status_code})"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    llm_errors.reset_quota_state()
    monkeypatch.setattr(settings, "llm_fallback_model", "")
    monkeypatch.setattr(settings, "llm_quota_cooldown_s", 300.0)
    yield
    llm_errors.reset_quota_state()


def scripted(refuse=(CLOUD,), error=None):
    """call(model) that records the model it was given; models in `refuse` answer 402."""
    seen = []

    async def call(m):
        seen.append(m)
        if m in refuse:
            raise error or ResponseError(OLLAMA_402, 402)
        return f"answer from {m}"
    call.seen = seen
    return call


# ------------------------------------------------------------------ recognising the failure
def test_402_is_a_quota_error_and_429_only_with_a_quota_text():
    assert quota_status(ResponseError(OLLAMA_402, 402)) == 402
    assert quota_status(ResponseError("you have reached your weekly usage limit, upgrade at ollama.com/plans", 429)) == 429
    assert quota_status(ResponseError("server busy, please try again. maximum pending requests exceeded", 429)) is None
    assert quota_status(ResponseError("boom", 500)) is None
    assert quota_status(ValueError("nope")) is None


def test_a_wrapped_402_is_found_through_the_cause_chain():
    try:
        try:
            raise ResponseError(OLLAMA_402, 402)
        except ResponseError as inner:
            raise RuntimeError("langchain wrapper") from inner
    except RuntimeError as outer:
        q = as_quota_error(outer, CLOUD)
    assert q is not None and q.status == 402 and q.model == CLOUD and "ollama.com/plans" in q.provider_message


def test_cloud_model_names():
    assert is_cloud_model("gemma4:31b-cloud") and is_cloud_model("gpt-oss:cloud")
    assert not is_cloud_model("qwen2.5-coder:7b") and not is_cloud_model("")


# ------------------------------------------------------------------ with_model_fallback
def test_without_a_fallback_the_error_is_readable_and_the_model_is_called_once():
    call = scripted()
    with pytest.raises(ModelQuotaError) as ei:
        run(with_model_fallback(call, CLOUD))
    msg = str(ei.value)
    assert call.seen == [CLOUD]
    assert CLOUD in msg and "402" in msg and "ENGINE_LLM_FALLBACK_MODEL" in msg and "ollama.com/plans" in msg


def test_with_a_fallback_the_local_model_answers_and_the_cloud_model_is_skipped_afterwards(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    call = scripted()
    assert run(with_model_fallback(call, CLOUD)) == f"answer from {LOCAL}"
    assert run(with_model_fallback(call, CLOUD)) == f"answer from {LOCAL}"
    assert call.seen == [CLOUD, LOCAL, LOCAL]           # second call went straight to the fallback (cooldown)


def test_cooldown_zero_tries_the_cloud_model_every_time(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    monkeypatch.setattr(settings, "llm_quota_cooldown_s", 0)
    call = scripted()
    run(with_model_fallback(call, CLOUD))
    run(with_model_fallback(call, CLOUD))
    assert call.seen == [CLOUD, LOCAL, CLOUD, LOCAL]


def test_a_failing_fallback_reports_both_facts(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    call = scripted(error=None)

    async def worse(m):
        if m == LOCAL:
            raise ResponseError(f"model '{LOCAL}' not found", 404)
        return await call(m)
    with pytest.raises(ModelQuotaError) as ei:
        run(with_model_fallback(worse, CLOUD))
    assert ei.value.fallback_tried == LOCAL and "not found" in str(ei.value) and CLOUD in str(ei.value)


def test_a_fallback_equal_to_the_model_is_no_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", CLOUD)
    call = scripted()
    with pytest.raises(ModelQuotaError):
        run(with_model_fallback(call, CLOUD))
    assert call.seen == [CLOUD]


def test_other_failures_and_cancellation_pass_through_untouched(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    call = scripted(error=ResponseError("boom", 500))
    with pytest.raises(ResponseError):
        run(with_model_fallback(call, CLOUD))
    assert call.seen == [CLOUD]                          # no fallback for a non-quota failure

    async def cancelled(m):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        run(with_model_fallback(cancelled, CLOUD))


def test_fallback_scope_collects_one_note_per_pair(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    call = scripted()

    async def twice():
        with fallback_scope() as events:
            await with_model_fallback(call, CLOUD)
            await with_model_fallback(call, CLOUD)
        return events
    events = run(twice())
    assert len(events) == 1 and events[0]["from"] == CLOUD and events[0]["to"] == LOCAL
    note = fallback_note(events)
    assert CLOUD in note and LOCAL in note and "402" in note
    assert fallback_note([]) == ""


# ------------------------------------------------------------------ model_calls (patcher / locator / JSON helpers)
class FakeChat:
    def __init__(self, model, **kw):
        self.model = model

    async def ainvoke(self, prompt):
        if self.model == CLOUD:
            raise ResponseError(OLLAMA_402, 402)
        return AIMessage(content='{"ok": true}' if "json" in str(prompt) else f"hi from {self.model}")


def test_ask_text_and_ask_json_fall_back_to_the_local_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    monkeypatch.setattr(model_calls, "_build", lambda m, **kw: FakeChat(m))
    assert run(model_calls.ask_text(CLOUD, "hello")) == f"hi from {LOCAL}"
    assert run(model_calls.ask_json(CLOUD, "give json")) == {"ok": True}


def test_ask_text_without_fallback_raises_the_readable_error(monkeypatch):
    monkeypatch.setattr(model_calls, "_build", lambda m, **kw: FakeChat(m))
    with pytest.raises(ModelQuotaError):
        run(model_calls.ask_text(CLOUD, "hello"))
    assert run(model_calls.ask_json(CLOUD, "give json")) is None      # JSON helper keeps its "any failure -> None" contract


# ------------------------------------------------------------------ the chat answer step (rag_graph)
class GraphChat(FakeChat):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)


def _state():
    return {"repo_id": "r1", "thread_id": "t1", "context_str": "", "attachment_context": "", "web_context": "",
            "web_footer": "", "web_mode": "off", "web_query": "", "web_note": "", "retrieval_meta": {},
            "messages": [HumanMessage(content="what does app.py do?")]}


def _manager(monkeypatch):
    monkeypatch.setattr(rag_graph, "ChatOllama", GraphChat)
    monkeypatch.setattr(settings, "llm_model", CLOUD)
    m = rag_graph.RAGWorkflowManager.__new__(rag_graph.RAGWorkflowManager)
    m.code_agent = None
    monkeypatch.setattr(m, "_system_prompt", lambda state: "SYSTEM", raising=False)
    return m


def test_answer_node_falls_back_and_says_so(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", LOCAL)
    out = run(_manager(monkeypatch)._generate_answer_node(_state()))
    text = out["messages"][-1].content
    assert text.startswith(f"hi from {LOCAL}") and "Model note" in text and CLOUD in text


def test_answer_node_without_fallback_raises_the_readable_error(monkeypatch):
    with pytest.raises(ModelQuotaError) as ei:
        run(_manager(monkeypatch)._generate_answer_node(_state()))
    assert "402" in str(ei.value)


# ------------------------------------------------------------------ HTTP mapping
def test_api_maps_the_error_to_http_402_with_a_detail_the_ui_can_show():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    app = fastapi.FastAPI()
    app.add_exception_handler(ModelQuotaError, quota_error_response)

    @app.post("/boom")
    async def boom():
        raise ModelQuotaError(CLOUD, 402, OLLAMA_402)
    r = TestClient(app).post("/boom")
    assert r.status_code == 402
    body = r.json()
    assert CLOUD in body["detail"] and body["error"] == "model_quota" and body["fallback_tried"] is None


def test_api_view_registers_the_handler():
    from views import api_view
    assert api_view.app.exception_handlers.get(ModelQuotaError) is quota_error_response
