"""Pluggable inference runners (services/runners/): registry, Cloudflare mapping + client kwargs, error mapping, reply metadata.

No network, no Ollama, no langchain_openai: provider classes are replaced by recording fakes (the `chat_cls` / `embeddings_cls` seams).
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from config.settings import settings
from services import llm_errors, model_calls
from services.code_agent import describe_meta, reply_meta
from services.llm_errors import (ModelAuthError, ModelQuotaError, as_quota_error, quota_status, with_model_fallback)
from services.runners import (RunnerConfigError, available_runners, check_runner_config, get_embedding_runner, get_runner,
                              make_chat, register_runner, resolve_model)

run = asyncio.run
CF_DEFAULT = "@cf/google/gemma-4-26b-a4b-it"


class Rec:
    """Stands in for ChatOllama / ChatOpenAI / OpenAIEmbeddings: remembers its constructor kwargs."""
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in dict(llm_runner="ollama", embedding_runner="ollama", cf_account_id="acc123", cf_api_token="tok-secret",
                     cf_llm_model=CF_DEFAULT, cf_vision_model="@cf/vision/x", cf_embedding_model="@cf/baai/bge-base-en-v1.5",
                     cf_ai_gateway_url="", llm_fallback_model="", llm_quota_cooldown_s=300.0).items():
        monkeypatch.setattr(settings, k, v)
    llm_errors.reset_quota_state()
    yield
    llm_errors.reset_quota_state()


class APIStatusError(Exception):
    """openai.APIStatusError shape: .status_code, .message, .body."""
    def __init__(self, message, status_code, body=None):
        super().__init__(f"Error code: {status_code} - {message}")
        self.message, self.status_code, self.body = message, status_code, body


class HttpxLike(Exception):
    """httpx.HTTPStatusError shape: status only on .response."""
    def __init__(self, text, status):
        super().__init__(text)
        self.response = type("R", (), {"status_code": status})()


# ------------------------------------------------------------------ registry
def test_default_runner_is_ollama_and_both_are_registered():
    assert get_runner().name == "ollama" and get_embedding_runner().name == "ollama"
    assert {"ollama", "cloudflare"} <= set(available_runners())


def test_runner_is_chosen_by_name_or_by_setting(monkeypatch):
    assert get_runner("cloudflare").name == "cloudflare"
    assert get_runner("Cloudflare ").name == "cloudflare"
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    assert get_runner().name == "cloudflare"


def test_unknown_runner_lists_the_known_ones():
    with pytest.raises(RunnerConfigError) as ei:
        get_runner("nope")
    assert "ollama" in str(ei.value) and "cloudflare" in str(ei.value)


def test_a_new_runner_registers_in_one_line(monkeypatch):
    class Mine:
        name, supports = "mine", {}
        def map_model(self, m, *, vision=False): return m
        def chat(self, model, **kw): return ("mine", model, kw)
    monkeypatch.setattr("services.runners._REGISTRY", dict(__import__("services.runners", fromlist=["_REGISTRY"])._REGISTRY))
    register_runner("mine", Mine)
    monkeypatch.setattr(settings, "llm_runner", "mine")
    assert make_chat("m1", temperature=0)[:2] == ("mine", "m1")


def test_settings_read_the_documented_env_names(monkeypatch):
    monkeypatch.setenv("LLM_RUNNER", "cloudflare")
    monkeypatch.setenv("CF_ACCOUNT_ID", "from-env")
    s = type(settings)()
    assert s.llm_runner == "cloudflare" and s.cf_account_id == "from-env"
    assert settings.cf_model_list[:2] == ["@cf/google/gemma-4-26b-a4b-it", "@cf/google/gemma-3-12b-it"]


# ------------------------------------------------------------------ ollama stays as it was
def test_ollama_chat_kwargs_are_the_ones_chatollama_always_got(monkeypatch):
    monkeypatch.setattr(model_calls, "_callbacks", lambda *a: ["CB"])
    llm = get_runner().chat("gemma4:31b-cloud", temperature=0, max_tokens=2048, json_mode=True, reasoning=False,
                            num_ctx=16384, callbacks=["CB"], chat_cls=Rec)
    assert llm.kw == dict(model="gemma4:31b-cloud", base_url=model_calls.chat_base_url(), temperature=0, num_ctx=16384,
                          callbacks=["CB"], num_predict=2048, format="json", reasoning=False)


def test_ollama_optional_kwargs_are_left_out_when_not_asked_for():
    kw = get_runner().chat("m", temperature=0.1, num_ctx=8, chat_cls=Rec).kw
    assert set(kw) == {"model", "base_url", "temperature", "num_ctx"}


def test_ollama_embeddings_keep_model_and_url(monkeypatch):
    e = get_embedding_runner().embeddings(embeddings_cls=Rec)
    assert e.kw == dict(model=settings.embedding_model, base_url=settings.ollama_base_url)


# ------------------------------------------------------------------ cloudflare
def test_ollama_style_names_map_to_the_cloudflare_model_and_cf_ids_pass_through():
    r = get_runner("cloudflare")
    assert r.map_model("gemma4:31b-cloud") == CF_DEFAULT
    assert r.map_model("qwen2.5-coder:7b") == CF_DEFAULT
    assert r.map_model("") == CF_DEFAULT
    assert r.map_model("@cf/google/gemma-3-12b-it") == "@cf/google/gemma-3-12b-it"
    assert r.map_model("gemma4:31b-cloud", vision=True) == "@cf/vision/x"
    assert get_runner("ollama").map_model("gemma4:31b-cloud") == "gemma4:31b-cloud"


def test_resolve_model_follows_the_active_runner(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "gemma4:31b-cloud")
    assert resolve_model() == "gemma4:31b-cloud"
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    assert resolve_model() == CF_DEFAULT


def test_cloudflare_chat_kwargs():
    llm = get_runner("cloudflare").chat("gemma4:31b-cloud", temperature=0, max_tokens=512, json_mode=True, reasoning="high",
                                        num_ctx=32768, callbacks=["CB"], chat_cls=Rec)
    kw = llm.kw
    assert kw["base_url"] == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"
    assert kw["api_key"] == "tok-secret" and kw["model"] == CF_DEFAULT
    assert kw["max_tokens"] == 512 and kw["callbacks"] == ["CB"] and kw["temperature"] == 0
    assert kw["model_kwargs"] == {"response_format": {"type": "json_object"}}
    assert "num_ctx" not in kw and "reasoning" not in kw and "format" not in kw
    assert kw["timeout"] > 0 and kw["max_retries"] >= 0


def test_cloudflare_plain_text_call_has_no_response_format_or_output_cap():
    kw = get_runner("cloudflare").chat("@cf/google/gemma-3-12b-it", chat_cls=Rec).kw
    assert "model_kwargs" not in kw and "max_tokens" not in kw and kw["model"] == "@cf/google/gemma-3-12b-it"


def test_vision_calls_use_the_vision_model():
    assert get_runner("cloudflare").chat("gemma4:31b-cloud", vision=True, chat_cls=Rec).kw["model"] == "@cf/vision/x"


def test_gateway_url_overrides_the_account_base_url(monkeypatch):
    monkeypatch.setattr(settings, "cf_ai_gateway_url", "https://gateway.ai.cloudflare.com/v1/a/g/workers-ai/v1/")
    assert get_runner("cloudflare").chat("m:1", chat_cls=Rec).kw["base_url"] == "https://gateway.ai.cloudflare.com/v1/a/g/workers-ai/v1"


@pytest.mark.parametrize("field,word", [("cf_account_id", "CF_ACCOUNT_ID"), ("cf_api_token", "CF_API_TOKEN")])
def test_missing_credentials_fail_fast_with_a_clear_message(monkeypatch, field, word):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, field, "  ")
    with pytest.raises(RunnerConfigError) as ei:
        check_runner_config()
    assert word in str(ei.value)
    with pytest.raises(RunnerConfigError):
        get_runner().chat("m", chat_cls=Rec)


def test_ollama_users_need_no_cloudflare_settings(monkeypatch):
    monkeypatch.setattr(settings, "cf_account_id", "")
    monkeypatch.setattr(settings, "cf_api_token", "")
    check_runner_config()           # must not raise


def test_the_openai_package_is_imported_lazily_and_its_absence_is_explained(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langchain_openai", None)      # makes `import langchain_openai` raise ImportError
    with pytest.raises(RunnerConfigError) as ei:
        get_runner("cloudflare").chat("m")
    assert "langchain-openai" in str(ei.value)


def test_build_routes_through_the_runner_and_maps_the_name(monkeypatch):
    seen = {}
    class Fake:
        name, supports = "cloudflare", {}
        def map_model(self, m, *, vision=False): return CF_DEFAULT if ":" in m else m
        def chat(self, model, **kw): seen.update(model=model, **kw); return "llm"
    monkeypatch.setattr(model_calls, "get_runner", lambda *a: Fake())
    assert model_calls._build("gemma4:31b-cloud", json_mode=True, num_predict=2048, num_ctx=16384, repo_id="r", phase="X") == "llm"
    assert seen["model"] == CF_DEFAULT and seen["json_mode"] is True and seen["max_tokens"] == 2048 and seen["num_ctx"] == 16384


def test_model_scope_accepts_cloudflare_ids():
    with model_calls.model_scope("@cf/google/gemma-3-12b-it"):
        assert model_calls.reasoning_model() == "@cf/google/gemma-3-12b-it"


def test_cloudflare_model_list_is_the_configured_one(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "cf_models", "@cf/a/x, @cf/b/y,")
    assert run(model_calls.list_chat_models()) == ["@cf/a/x", "@cf/b/y"]


# ------------------------------------------------------------------ embeddings are independent of the chat runner
def test_embedding_runner_is_independent_of_the_chat_runner(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    assert get_runner().name == "cloudflare" and get_embedding_runner().name == "ollama"
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    monkeypatch.setattr(settings, "embedding_runner", "cloudflare")
    assert get_runner().name == "ollama" and get_embedding_runner().name == "cloudflare"


def test_cloudflare_embeddings_use_the_cf_model_not_the_ollama_one():
    e = get_runner("cloudflare").embeddings(settings.embedding_model, embeddings_cls=Rec)       # an Ollama-style name from settings
    assert e.kw["model"] == "@cf/baai/bge-base-en-v1.5"
    assert e.kw["base_url"].endswith("/accounts/acc123/ai/v1") and e.kw["api_key"] == "tok-secret"
    assert e.kw["check_embedding_ctx_length"] is False


# ------------------------------------------------------------------ reply metadata: OpenAI shape
def test_finish_reason_length_is_read_as_done_reason_length():
    msg = AIMessage(content="", response_metadata={"finish_reason": "length", "model_name": "@cf/x",
                                                   "token_usage": {"prompt_tokens": 11, "completion_tokens": 22}})
    meta = reply_meta(msg)
    assert meta["done_reason"] == "length" and meta["eval_count"] == 22 and meta["prompt_eval_count"] == 11
    assert "done_reason=length" in describe_meta(meta)


def test_usage_metadata_is_used_when_token_usage_is_absent():
    msg = AIMessage(content="x", response_metadata={"finish_reason": "stop"}, usage_metadata={"input_tokens": 3, "output_tokens": 4})
    meta = reply_meta(msg)
    assert (meta["done_reason"], meta["eval_count"], meta["prompt_eval_count"]) == ("stop", 4, 3)


def test_ollama_metadata_is_unchanged():
    msg = AIMessage(content="x", response_metadata={"done_reason": "length", "eval_count": 5, "prompt_eval_count": 6, "model": "m"})
    assert reply_meta(msg) == {"model": "m", "done_reason": "length", "eval_count": 5, "prompt_eval_count": 6}


def test_telemetry_counts_openai_style_usage():
    import json
    from services.telemetry import StdoutTelemetryCallback
    seen = []
    cb = StdoutTelemetryCallback(model_name="m", repo_id="r", phase="P", emitter=lambda d: seen.append(json.loads(d)))
    gen = type("G", (), {"message": None, "generation_info": {}, "text": "hello"})()
    resp = type("Resp", (), {"generations": [[gen]], "llm_output": {"token_usage": {"prompt_tokens": 7, "completion_tokens": 9}}})()
    cb.start_time, cb.prompts = 0.0, ["p"]
    cb.on_llm_end(resp)
    assert seen and seen[0]["input_tokens"] == 7 and seen[0]["output_tokens"] == 9


# ------------------------------------------------------------------ error mapping
NEURONS = "you have used up your daily free allocation of 10,000 neurons, please upgrade to Cloudflare's Workers Paid plan"


def test_openai_style_errors_are_recognised_as_quota():
    assert quota_status(APIStatusError(NEURONS, 429)) == 429
    assert quota_status(APIStatusError("payment required", 402)) == 402
    assert quota_status(HttpxLike(NEURONS, 429)) == 429                      # status only on .response
    assert quota_status(APIStatusError("Too many requests, slow down", 429)) is None       # a per-minute rate limit is not a quota
    assert quota_status(APIStatusError("boom", 500)) is None


def test_quota_message_names_cloudflare_when_that_is_the_runner(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    q = as_quota_error(APIStatusError(NEURONS, 429), CF_DEFAULT)
    assert isinstance(q, ModelQuotaError) and "Cloudflare Workers AI" in str(q) and "Ollama Cloud" not in str(q)
    assert CF_DEFAULT in str(q) and "429" in str(q)


def test_ollama_quota_message_is_unchanged():
    q = ModelQuotaError("gemma4:31b-cloud", 402, "x")
    assert "Ollama Cloud" in str(q) and "ENGINE_LLM_FALLBACK_MODEL" in str(q)


def test_auth_errors_become_a_readable_error_with_no_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "llm_fallback_model", "@cf/google/gemma-3-12b-it")
    calls = []

    async def call(m):
        calls.append(m)
        raise APIStatusError("Authentication error", 401)
    with pytest.raises(ModelAuthError) as ei:
        run(with_model_fallback(call, CF_DEFAULT))
    assert "CF_API_TOKEN" in str(ei.value) and ei.value.status == 401 and calls == [CF_DEFAULT]
    with pytest.raises(ModelAuthError):
        async def forbidden(m):
            raise HttpxLike("forbidden", 403)
        run(with_model_fallback(forbidden, CF_DEFAULT))


def test_other_failures_still_propagate_unchanged():
    async def call(m):
        raise APIStatusError("boom", 500)
    with pytest.raises(APIStatusError):
        run(with_model_fallback(call, CF_DEFAULT))


def test_quota_on_cloudflare_falls_back_to_a_cf_fallback_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "llm_fallback_model", "@cf/google/gemma-3-12b-it")
    seen = []

    async def call(m):
        seen.append(m)
        if m == CF_DEFAULT:
            raise APIStatusError(NEURONS, 429)
        return "ok"
    assert run(with_model_fallback(call, CF_DEFAULT)) == "ok" and seen == [CF_DEFAULT, "@cf/google/gemma-3-12b-it"]


def test_an_ollama_style_fallback_that_maps_to_the_same_cf_model_is_not_a_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "llm_fallback_model", "qwen2.5-coder:7b")         # would map to CF_DEFAULT again
    assert llm_errors.fallback_for(CF_DEFAULT) == ""
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    assert llm_errors.fallback_for("gemma4:31b-cloud") == "qwen2.5-coder:7b"
