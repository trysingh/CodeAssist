"""OpenAI-compatible runner family: shared base (openai_compat), Cloudflare after the refactor, Google AI Studio, generic "openai".

No network, no langchain_openai: provider classes are recording fakes (the `chat_cls` / `embeddings_cls` seams) and the catalogue
request (`_get_json`) is replaced. FastAPI is stubbed here, so the UI -> API -> runner chain is checked at function level.
"""
import asyncio
import json
import pathlib
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import settings
from services import llm_errors, model_calls
from services.llm_errors import ModelAuthError, ModelQuotaError, as_auth_error, as_quota_error, auth_status, quota_status, with_model_fallback
from services.runner_info import check_chat_choice, models_payload, runners_payload
from services.runners import (OpenAICompatRunner, RunnerConfigError, check_runner_config, get_embedding_runner, get_runner, make_chat,
                              register_runner, require_runner, resolve_model, runner_scope, runner_status, thread_scope)
from services.runners import _REGISTRY
from services.runners import aistudio_runner as asr
from services.runners.aistudio_runner import GoogleAIStudioRunner
from services.runners.cloudflare_runner import CloudflareRunner
from services.runners.openai_compat import fold_system_messages, folding_chat_cls, is_folding
from services.runners.openai_runner import OpenAIRunner
from services.runners.scope import session_key
from services.telemetry import StdoutTelemetryCallback
from services.token_budget import effective_mode, ledger_scope, max_input_tokens

run = asyncio.run
CF_DEFAULT = "@cf/google/gemma-4-26b-a4b-it"
G31, G26 = "gemma-4-31b-it", "gemma-4-26b-a4b-it"


class Rec:
    """Stands in for ChatOpenAI / OpenAIEmbeddings: remembers its constructor kwargs."""
    def __init__(self, **kw):
        self.kw = kw


class APIStatusError(Exception):
    """openai.APIStatusError shape: .status_code, .message, .body."""
    def __init__(self, message, status_code, body=None):
        super().__init__(f"Error code: {status_code} - {message}")
        self.message, self.status_code, self.body = message, status_code, body


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in dict(llm_runner="ollama", embedding_runner="ollama", token_mode="auto", max_input_tokens=None, llm_fallback_model="",
                     llm_quota_cooldown_s=300.0, llm_model="gemma4:31b-cloud",
                     cf_account_id="acc123", cf_api_token="tok-secret", cf_llm_model=CF_DEFAULT, cf_ai_gateway_url="",
                     cf_session_affinity=True, cf_models=CF_DEFAULT, cf_list_all_models=False,
                     aistudio_base_url=asr.DEFAULT_BASE_URL, gemini_api_key="AIza-secret-key", aistudio_llm_model=G31,
                     aistudio_vision_model=G31, aistudio_embedding_model="gemini-embedding-001", aistudio_models=f"{G31},{G26}",
                     aistudio_list_all_models=True, aistudio_fold_system="auto", aistudio_json_mode="auto",
                     aistudio_max_input_tokens=60000, compat_models_cache_s=600,
                     openai_compat_base_url="", openai_compat_api_key="", openai_compat_models="", openai_compat_llm_model="",
                     openai_compat_embedding_model="", openai_compat_remote=True, openai_compat_list_all_models=False,
                     openai_compat_max_input_tokens=0).items():
        monkeypatch.setattr(settings, k, v)
    for cls in (GoogleAIStudioRunner, OpenAIRunner):
        cls._CACHE = {"names": [], "at": 0.0, "key": ""}
    llm_errors.reset_quota_state()
    yield
    llm_errors.reset_quota_state()
    for cls in (GoogleAIStudioRunner, OpenAIRunner):
        cls._CACHE = {"names": [], "at": 0.0, "key": ""}


def catalogue(monkeypatch, body, calls=None):
    async def fake(self, url, headers):
        if calls is not None:
            calls.append((url, headers))
        if isinstance(body, Exception):
            raise body
        return body
    monkeypatch.setattr(OpenAICompatRunner, "_get_json", fake)


# ------------------------------------------------------------------ the refactor: Cloudflare is unchanged
def test_runners_share_the_base_and_cloudflare_keeps_its_identity():
    assert all(isinstance(get_runner(n), OpenAICompatRunner) for n in ("cloudflare", "aistudio", "openai"))
    cf = get_runner("cloudflare")
    assert (cf.name, cf.label, cf.remote, cf.max_input_tokens) == ("cloudflare", "Cloudflare Workers AI", True, settings.cf_max_input_tokens)


def test_cloudflare_chat_kwargs_are_exactly_what_they_were():
    kw = get_runner("cloudflare").chat("gemma4:31b-cloud", temperature=0, max_tokens=512, json_mode=True, reasoning="high",
                                       num_ctx=32768, callbacks=["CB"], chat_cls=Rec).kw
    assert kw == dict(model=CF_DEFAULT, base_url="https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1", api_key="tok-secret",
                      temperature=0, timeout=settings.cf_timeout_s, max_retries=settings.cf_max_retries, callbacks=["CB"],
                      max_tokens=512, model_kwargs={"response_format": {"type": "json_object"}})
    assert set(get_runner("cloudflare").chat("@cf/x/y", chat_cls=Rec).kw) == {"model", "base_url", "api_key", "temperature", "timeout", "max_retries"}


def test_cloudflare_keeps_affinity_header_mapping_and_embeddings():
    cf = get_runner("cloudflare")
    with thread_scope("repoA", "th1"):
        assert cf.chat("m", chat_cls=Rec).kw["default_headers"] == {"x-session-affinity": session_key("repoA", "th1")}
    assert cf.map_model("qwen2.5-coder:7b") == CF_DEFAULT and cf.map_model("@cf/a/b") == "@cf/a/b"
    e = cf.embeddings(settings.llm_model, embeddings_cls=Rec).kw
    assert e["model"] == "@cf/baai/bge-base-en-v1.5" and e["check_embedding_ctx_length"] is False and e["api_key"] == "tok-secret"


def test_cloudflare_model_list_still_uses_its_paged_search_and_module_cache(monkeypatch):
    from services.runners import cloudflare_runner as cfr
    cfr._CACHE.update(names=[], at=0.0, key="")
    monkeypatch.setattr(settings, "cf_list_all_models", True)
    calls = []

    async def page(self, n):
        calls.append(n)
        return {"success": True, "result_info": {"total_count": 1}, "result": [{"name": "@cf/m/x", "task": {"name": "Text Generation"}}]}
    monkeypatch.setattr(CloudflareRunner, "_search_page", page)
    try:
        assert run(CloudflareRunner().list_models()) == [CF_DEFAULT, "@cf/m/x"]
        assert run(CloudflareRunner().list_models()) == [CF_DEFAULT, "@cf/m/x"] and calls == [1]
    finally:
        cfr._CACHE.update(names=[], at=0.0, key="")


# ------------------------------------------------------------------ aistudio: identity, config, client kwargs
def test_aistudio_identity_and_client_kwargs():
    r = get_runner("aistudio")
    assert (r.name, r.label, r.remote) == ("aistudio", "Google AI Studio", True)
    kw = r.chat("gemma4:31b-cloud", temperature=0, max_tokens=900, callbacks=["CB"], num_ctx=4096, reasoning=True, chat_cls=Rec).kw
    assert kw["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai" and kw["api_key"] == "AIza-secret-key"
    assert kw["model"] == G31 and kw["max_tokens"] == 900 and kw["callbacks"] == ["CB"]
    assert "default_headers" not in kw and "num_ctx" not in kw and "reasoning" not in kw      # no session affinity header, no Ollama knobs
    with thread_scope("repoA", "th1"):
        assert "default_headers" not in r.chat("m", chat_cls=Rec).kw


def test_aistudio_base_url_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "aistudio_base_url", "https://proxy.example/v1beta/openai/")
    assert get_runner("aistudio").chat("m", chat_cls=Rec).kw["base_url"] == "https://proxy.example/v1beta/openai"


def test_aistudio_embeddings_use_its_embedding_model_and_skip_the_tiktoken_check():
    e = get_runner("aistudio").embeddings(settings.embedding_model, embeddings_cls=Rec).kw
    assert e["model"] == "gemini-embedding-001" and e["check_embedding_ctx_length"] is False and e["api_key"] == "AIza-secret-key"
    assert get_runner("aistudio").embeddings("text-embedding-004", embeddings_cls=Rec).kw["model"] == "text-embedding-004"


def test_the_openai_package_is_imported_lazily_for_the_new_runners(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    monkeypatch.setattr(settings, "openai_compat_base_url", "http://localhost:8000/v1")
    for name in ("aistudio", "openai"):
        with pytest.raises(RunnerConfigError, match="langchain-openai") as ei:
            get_runner(name).chat("m")
        assert f"'{name}' runner" in str(ei.value)


# ------------------------------------------------------------------ aistudio: model mapping table
@pytest.mark.parametrize("given,expected", [
    ("gemma4:31b-cloud", G31), ("gemma4:31b", G31), ("Gemma4:31B-it-q4", G31), ("gemma-4:31b", G31),
    ("gemma4:26b", G26), ("gemma4:26b-a4b-cloud", G26),
    ("qwen2.5-coder:7b", G31), ("llama3:8b", G31), ("gemma4:e4b", G31),       # any other Ollama tag -> AISTUDIO_LLM_MODEL
    ("", G31), ("   ", G31),
    (G31, G31), (G26, G26), ("gemini-2.5-flash", "gemini-2.5-flash"), ("models/gemini-2.5-pro", "gemini-2.5-pro"),
])
def test_aistudio_model_mapping(given, expected):
    assert get_runner("aistudio").map_model(given) == expected


def test_mapping_follows_the_settings_and_the_table_is_extendable(monkeypatch):
    monkeypatch.setattr(settings, "aistudio_llm_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "aistudio_vision_model", "gemini-2.5-pro")
    r = get_runner("aistudio")
    assert r.map_model("qwen2.5-coder:7b") == "gemini-2.5-flash" and r.map_model("gemma4:31b-cloud") == G31   # the table wins over the default
    assert r.map_model("qwen2.5-coder:7b", vision=True) == "gemini-2.5-pro" and r.map_model("", vision=True) == "gemini-2.5-pro"
    assert r.map_model(G26, vision=True) == G26                                  # an explicit AI Studio id is never remapped
    monkeypatch.setattr(asr, "MODEL_MAP", asr.MODEL_MAP + [(r"^gemma4:e4b", "gemma-4-e4b-it")])
    assert r.map_model("gemma4:e4b-cloud") == "gemma-4-e4b-it"


def test_resolve_model_follows_the_aistudio_scope():
    assert resolve_model() == "gemma4:31b-cloud"
    with runner_scope("aistudio"):
        assert resolve_model() == G31 and resolve_model("gemma4:26b") == G26


# ------------------------------------------------------------------ aistudio: catalogue
BODY = {"object": "list", "data": [
    {"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-2.5-pro"}, {"id": "models/gemma-3-12b-it"}, {"id": G31},
    {"id": "models/gemini-embedding-001"}, {"id": "models/text-embedding-004"}, {"id": "models/imagen-4.0-generate-001"},
    {"id": "models/veo-3.0-generate-001"}, {"id": "models/gemini-2.5-flash-preview-tts"}, {"id": "models/gemini-2.5-flash-native-audio-dialog"},
    {"id": "models/gemini-live-2.5-flash"}, {"id": "models/aqa"}, {"id": "models/gemini-2.0-flash-preview-image-generation"},
    {"id": "models/gemini-2.5-flash-image"}, {"id": "models/chat-bison-001"}]}


def test_catalogue_keeps_chat_models_strips_prefix_and_lists_pinned_first(monkeypatch):
    calls = []
    catalogue(monkeypatch, BODY, calls)
    got = run(get_runner("aistudio").list_models())
    assert got == [G31, G26, "gemini-2.5-flash", "gemini-2.5-pro", "gemma-3-12b-it"]       # pinned first (in AISTUDIO_MODELS order), no duplicates
    assert calls == [("https://generativelanguage.googleapis.com/v1beta/openai/models", {"Authorization": "Bearer AIza-secret-key"})]


def test_catalogue_is_cached_across_runner_instances(monkeypatch):
    calls = []
    catalogue(monkeypatch, BODY, calls)
    first = run(get_runner("aistudio").list_models())
    assert run(get_runner("aistudio").list_models()) == first and len(calls) == 1


def test_catalogue_failure_falls_back_to_the_pinned_list(monkeypatch):
    catalogue(monkeypatch, RuntimeError("503"))
    assert run(get_runner("aistudio").list_models()) == [G31, G26]
    catalogue(monkeypatch, {"error": "x"})                                                # unexpected shape: still no crash
    assert run(get_runner("aistudio").list_models()) == [G31, G26]


def test_catalogue_off_or_without_key_lists_only_the_pinned_models(monkeypatch):
    calls = []
    catalogue(monkeypatch, BODY, calls)
    monkeypatch.setattr(settings, "aistudio_list_all_models", False)
    assert run(get_runner("aistudio").list_models()) == [G31, G26]
    monkeypatch.setattr(settings, "aistudio_list_all_models", True)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert run(get_runner("aistudio").list_models()) == [G31, G26] and calls == []


def test_a_stale_cache_survives_a_failing_refresh(monkeypatch):
    catalogue(monkeypatch, BODY)
    first = run(get_runner("aistudio").list_models())
    monkeypatch.setattr(settings, "compat_models_cache_s", 0)
    catalogue(monkeypatch, RuntimeError("down"))
    assert run(get_runner("aistudio").list_models()) == first


# ------------------------------------------------------------------ aistudio: system prompt folding
def test_fold_merges_system_text_into_the_first_human_message():
    out = fold_system_messages([SystemMessage(content="RULES"), SystemMessage(content="MORE"), HumanMessage(content="q1"),
                                AIMessage(content="a"), HumanMessage(content="q2")])
    assert [(m.type, m.content) for m in out] == [("human", "RULES\n\nMORE\n\nq1"), ("ai", "a"), ("human", "q2")]


def test_fold_edge_cases():
    plain = [HumanMessage(content="q"), AIMessage(content="a")]
    assert fold_system_messages(plain) == plain                                        # nothing to fold: untouched
    only = fold_system_messages([SystemMessage(content="RULES")])
    assert [(m.type, m.content) for m in only] == [("human", "RULES")]               # no human message: the system text becomes one
    blocks = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    out = fold_system_messages([SystemMessage(content="RULES"), HumanMessage(content=blocks)])
    assert out[0].content[0] == {"type": "text", "text": "RULES\n\n"} and out[0].content[1:] == blocks      # images survive
    orig = HumanMessage(content="q")
    fold_system_messages([SystemMessage(content="S"), orig])
    assert orig.content == "q"                                                           # the caller's message is not mutated


class FakeChatBase:
    """Shape of ChatOpenAI that matters here: _convert_input -> PromptValue.to_messages(), _get_request_payload(input_, *, stop)."""
    def __init__(self, **kw):
        self.kw = kw
        self.payloads = []

    def _convert_input(self, x):
        msgs = [HumanMessage(content=x)] if isinstance(x, str) else list(x)
        return types.SimpleNamespace(to_messages=lambda: msgs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        self.payloads.append((list(input_), stop, kwargs))
        return {"messages": [{"role": m.type, "content": m.content} for m in input_], "stop": stop, **kwargs}


def test_the_folding_class_rewrites_the_payload_for_every_input_shape():
    cls = folding_chat_cls(FakeChatBase)
    assert cls is folding_chat_cls(FakeChatBase) and issubclass(cls, FakeChatBase)           # one class per base, cached
    llm = cls(model="x")
    p = llm._get_request_payload([SystemMessage(content="S"), HumanMessage(content="q")], stop=["END"], n=1)
    assert p == {"messages": [{"role": "human", "content": "S\n\nq"}], "stop": ["END"], "n": 1}
    assert llm._get_request_payload("just text")["messages"] == [{"role": "human", "content": "just text"}]
    assert is_folding(llm) and not is_folding(FakeChatBase())


@pytest.mark.parametrize("model,expected", [(G31, True), (G26, True), ("gemma-3-12b-it", True), ("gemma4:31b-cloud", True),   # maps to gemma
                                            ("gemini-2.5-flash", False), ("models/gemini-2.5-pro", False)])
def test_auto_folds_for_gemma_and_not_for_gemini(model, expected):
    llm = get_runner("aistudio").chat(model, chat_cls=FakeChatBase)
    assert is_folding(llm) is expected and llm.kw["model"] == get_runner("aistudio").map_model(model)


def test_fold_setting_overrides_auto():
    r = get_runner("aistudio")
    settings.aistudio_fold_system = "on"
    assert is_folding(r.chat("gemini-2.5-flash", chat_cls=FakeChatBase)) and is_folding(r.chat(G31, chat_cls=FakeChatBase))
    settings.aistudio_fold_system = "off"
    assert not is_folding(r.chat(G31, chat_cls=FakeChatBase)) and not is_folding(r.chat("gemini-2.5-flash", chat_cls=FakeChatBase))
    settings.aistudio_fold_system = "bogus"                                             # invalid value = auto
    assert is_folding(r.chat(G31, chat_cls=FakeChatBase)) and not is_folding(r.chat("gemini-2.5-flash", chat_cls=FakeChatBase))


def test_folding_keeps_the_client_kwargs_and_the_provider_seam():
    llm = get_runner("aistudio").chat(G31, temperature=0.2, json_mode=True, callbacks=["CB"], chat_cls=FakeChatBase)
    assert isinstance(llm, FakeChatBase) and llm.kw["temperature"] == 0.2 and llm.kw["callbacks"] == ["CB"] and llm.kw["model"] == G31


# ------------------------------------------------------------------ aistudio: json mode
JSON = {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize("model,sent", [("gemini-2.5-flash", True), ("models/gemini-2.5-pro", True), (G31, False), (G26, False),
                                        ("gemma4:31b-cloud", False)])
def test_json_mode_auto_is_for_gemini_only(model, sent):
    kw = get_runner("aistudio").chat(model, json_mode=True, chat_cls=Rec).kw
    assert (kw.get("model_kwargs") == JSON) is sent and ("model_kwargs" in kw) is sent


def test_json_mode_setting_and_request_flag():
    r = get_runner("aistudio")
    assert "model_kwargs" not in r.chat("gemini-2.5-flash", chat_cls=Rec).kw                 # not asked for: never sent
    settings.aistudio_json_mode = "on"
    assert r.chat(G31, json_mode=True, chat_cls=Rec).kw["model_kwargs"] == JSON
    settings.aistudio_json_mode = "off"
    assert "model_kwargs" not in r.chat("gemini-2.5-flash", json_mode=True, chat_cls=Rec).kw


# ------------------------------------------------------------------ configuration, labels, payloads
def test_missing_key_means_not_set_up_with_a_reason(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "  ")
    row = {r["name"]: r for r in runners_payload()["runners"]}["aistudio"]
    assert row["configured"] is False and row["label"] == "Google AI Studio" and row["remote"] is True
    assert "GEMINI_API_KEY" in row["error"] and "aistudio.google.com/apikey" in row["error"]
    with pytest.raises(RunnerConfigError, match="GEMINI_API_KEY"):
        get_runner("aistudio").chat("m", chat_cls=Rec)
    with pytest.raises(RunnerConfigError, match="GEMINI_API_KEY"):
        get_runner("aistudio").embeddings(embeddings_cls=Rec)


def test_a_chat_request_with_an_unconfigured_runner_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with pytest.raises(RunnerConfigError, match="GEMINI_API_KEY"):
        check_chat_choice("aistudio", "lean")
    with pytest.raises(RunnerConfigError):
        require_runner("aistudio")
    with pytest.raises(RunnerConfigError, match="GEMINI_API_KEY"):
        run(models_payload("aistudio"))


def test_google_api_key_is_accepted_as_an_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
    assert type(settings)().gemini_api_key == "from-google"
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
    assert type(settings)().gemini_api_key == "from-gemini"


def test_new_settings_read_the_documented_env_names(monkeypatch):
    for k, v in {"AISTUDIO_BASE_URL": "https://x/v1", "AISTUDIO_LLM_MODEL": "gemini-2.5-flash", "AISTUDIO_MODELS": "a, b,",
                 "AISTUDIO_FOLD_SYSTEM": "off", "OPENAI_COMPAT_BASE_URL": "http://localhost:8000/v1", "OPENAI_COMPAT_MODELS": "m1,m2",
                 "OPENAI_COMPAT_REMOTE": "false"}.items():
        monkeypatch.setenv(k, v)
    s = type(settings)()
    assert (s.aistudio_base_url, s.aistudio_llm_model, s.aistudio_fold_system) == ("https://x/v1", "gemini-2.5-flash", "off")
    assert s.aistudio_model_list == ["a", "b"] and s.openai_compat_model_list == ["m1", "m2"]
    assert s.openai_compat_base_url == "http://localhost:8000/v1" and s.openai_compat_remote is False
    assert settings.aistudio_embedding_model == "gemini-embedding-001" and settings.aistudio_list_all_models is True


def test_labels_are_in_the_runners_payload():
    by = {r["name"]: r for r in runners_payload()["runners"]}
    assert {n: by[n]["label"] for n in ("ollama", "cloudflare", "aistudio", "openai")} == {
        "ollama": "Ollama (local)", "cloudflare": "Cloudflare Workers AI", "aistudio": "Google AI Studio", "openai": "OpenAI-compatible"}
    assert by["ollama"]["remote"] is False and by["cloudflare"]["remote"] is True and by["aistudio"]["configured"] is True


def test_a_runner_without_a_label_falls_back_to_its_name():
    class Bare:
        name, remote, supports, max_input_tokens = "bare", False, {}, 0
        def check_config(self): return None
    register_runner("bare", Bare)
    try:
        assert {r["name"]: r for r in runner_status()}["bare"]["label"] == "bare"
    finally:
        _REGISTRY.pop("bare", None)


# ------------------------------------------------------------------ generic "openai" runner
def test_openai_runner_is_not_set_up_without_a_base_url():
    row = {r["name"]: r for r in runners_payload()["runners"]}["openai"]
    assert row["configured"] is False and "OPENAI_COMPAT_BASE_URL" in row["error"] and row["label"] == "OpenAI-compatible"
    with pytest.raises(RunnerConfigError, match="OPENAI_COMPAT_BASE_URL"):
        check_chat_choice("openai", "")
    with pytest.raises(RunnerConfigError):
        get_runner("openai").chat("m", chat_cls=Rec)


def test_openai_runner_configured_by_settings_alone(monkeypatch):
    monkeypatch.setattr(settings, "openai_compat_base_url", "https://openrouter.ai/api/v1/")
    monkeypatch.setattr(settings, "openai_compat_api_key", " sk-or-1 ")
    monkeypatch.setattr(settings, "openai_compat_models", "meta-llama/llama-3.3-70b-instruct:free, qwen/qwen3-coder")
    monkeypatch.setattr(settings, "openai_compat_llm_model", "qwen/qwen3-coder")
    r = get_runner("openai")
    r.check_config()
    kw = r.chat("gemma4:31b-cloud", max_tokens=300, json_mode=True, chat_cls=Rec).kw
    assert kw["base_url"] == "https://openrouter.ai/api/v1" and kw["api_key"] == "sk-or-1" and kw["model"] == "qwen/qwen3-coder"
    assert kw["model_kwargs"] == JSON and kw["max_tokens"] == 300 and "default_headers" not in kw
    assert r.map_model("meta-llama/llama-3.3-70b-instruct:free") == "meta-llama/llama-3.3-70b-instruct:free"   # ':' inside a real id is kept
    assert not is_folding(r.chat("gemma-3-27b-it", chat_cls=FakeChatBase))                                  # folding is AI Studio only
    assert run(r.list_models()) == ["meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen3-coder"]


def test_openai_runner_needs_no_key_for_a_local_server(monkeypatch):
    monkeypatch.setattr(settings, "openai_compat_base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(settings, "openai_compat_llm_model", "Qwen/Qwen2.5-Coder-7B-Instruct")
    r = get_runner("openai")
    r.check_config()
    assert r.chat("qwen2.5-coder:7b", chat_cls=Rec).kw["api_key"] == "not-needed"                          # the client refuses ''
    assert r.chat("qwen2.5-coder:7b", chat_cls=Rec).kw["model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_openai_runner_remote_flag_drives_token_mode(monkeypatch):
    monkeypatch.setattr(settings, "openai_compat_base_url", "http://localhost:8000/v1")
    with runner_scope("openai"):
        assert get_runner().remote is True and effective_mode() == "lean"
        monkeypatch.setattr(settings, "openai_compat_remote", False)
        assert get_runner().remote is False and effective_mode() == "normal"
    assert {r["name"]: r for r in runners_payload()["runners"]}["openai"]["remote"] is False


def test_openai_runner_catalogue_is_optional_and_cached(monkeypatch):
    monkeypatch.setattr(settings, "openai_compat_base_url", "http://localhost:8000/v1")
    monkeypatch.setattr(settings, "openai_compat_models", "pinned-1")
    calls = []
    catalogue(monkeypatch, {"data": [{"id": "z-model"}, {"id": "pinned-1"}, {"id": "a-model"}]}, calls)
    assert run(get_runner("openai").list_models()) == ["pinned-1"] and calls == []                          # off by default
    monkeypatch.setattr(settings, "openai_compat_list_all_models", True)
    assert run(get_runner("openai").list_models()) == ["pinned-1", "a-model", "z-model"]
    assert calls[0] == ("http://localhost:8000/v1/models", {}) and run(get_runner("openai").list_models()) and len(calls) == 1


def test_openai_runner_embeddings_need_a_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "openai_compat_base_url", "http://localhost:8000/v1")
    with pytest.raises(RunnerConfigError, match="OPENAI_COMPAT_EMBEDDING_MODEL"):
        get_runner("openai").embeddings(settings.embedding_model, embeddings_cls=Rec)
    monkeypatch.setattr(settings, "openai_compat_embedding_model", "bge-m3")
    assert get_runner("openai").embeddings(settings.embedding_model, embeddings_cls=Rec).kw["model"] == "bge-m3"


# ------------------------------------------------------------------ startup check: only the configured default runners
def test_startup_check_ignores_unconfigured_runners():
    assert {r["name"]: r for r in runner_status()}["aistudio"]["configured"] is True              # sanity: configured here ...
    settings.gemini_api_key = ""
    check_runner_config()                                                                       # ollama defaults: aistudio / openai / cf are not looked at


def test_startup_check_does_not_call_other_runners(monkeypatch):
    seen = []

    class Boom:
        name, remote, supports, max_input_tokens = "boom", False, {}, 0
        def __init__(self): seen.append("built")
        def check_config(self): raise RuntimeError("must not be called at startup")
    register_runner("boom", Boom)
    try:
        check_runner_config()
        assert seen == []
    finally:
        _REGISTRY.pop("boom", None)


def test_startup_check_validates_the_configured_chat_and_embedding_runner(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "llm_runner", "aistudio")
    with pytest.raises(RunnerConfigError, match="GEMINI_API_KEY"):
        check_runner_config()
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    monkeypatch.setattr(settings, "embedding_runner", "openai")
    with pytest.raises(RunnerConfigError, match="OPENAI_COMPAT_BASE_URL"):
        check_runner_config()
    with runner_scope("cloudflare"):                      # a request scope never changes what the startup check validates
        monkeypatch.setattr(settings, "embedding_runner", "ollama")
        check_runner_config()


def test_embeddings_ignore_the_per_request_runner(monkeypatch):
    monkeypatch.setattr(settings, "embedding_runner", "aistudio")
    with runner_scope("cloudflare"):
        assert get_embedding_runner().name == "aistudio"
    monkeypatch.setattr(settings, "embedding_runner", "ollama")
    with runner_scope("aistudio"):
        assert get_embedding_runner().name == "ollama" and get_runner().name == "aistudio"


# ------------------------------------------------------------------ error mapping (Google shapes)
RESOURCE_EXHAUSTED = ('{"error": {"code": 429, "message": "You exceeded your current quota, please check your plan and billing details.", '
                      '"status": "RESOURCE_EXHAUSTED"}}')


def test_google_429_is_a_quota_error_and_the_message_links_the_docs():
    exc = APIStatusError("Resource has been exhausted (e.g. check quota).", 429, body={"error": {"status": "RESOURCE_EXHAUSTED"}})
    assert quota_status(exc) == 429 and quota_status(APIStatusError(RESOURCE_EXHAUSTED, 429)) == 429
    assert quota_status(APIStatusError("RESOURCE_EXHAUSTED", 429)) == 429
    assert quota_status(APIStatusError("Too many requests, slow down", 429)) is None             # unchanged: a plain rate limit is not a quota
    with runner_scope("aistudio"):
        q = as_quota_error(exc, G31)
    assert isinstance(q, ModelQuotaError) and q.runner == "aistudio" and "Google AI Studio" in str(q) and "Ollama" not in str(q)
    assert "https://aistudio.google.com/apikey" in str(q) and "https://ai.google.dev/gemini-api/docs/rate-limits" in str(q) and "429" in str(q)


def test_google_auth_failures_are_readable_for_400_401_403(monkeypatch):
    bad_key = APIStatusError("API key not valid. Please pass a valid API key.", 400, body=[{"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}])
    assert auth_status(bad_key) == 400 and auth_status(APIStatusError("denied", 403)) == 403 and auth_status(APIStatusError("x", 401)) == 401
    assert auth_status(APIStatusError("Unable to parse request: bad json", 400)) is None            # an ordinary 400 is not an auth error
    for exc, status in ((bad_key, 400), (APIStatusError("PERMISSION_DENIED: unregistered callers", 403), 403)):
        calls = []

        async def call(m, exc=exc):
            calls.append(m)
            raise exc
        monkeypatch.setattr(settings, "llm_fallback_model", G26)
        with runner_scope("aistudio"), pytest.raises(ModelAuthError) as ei:
            run(with_model_fallback(call, G31))
        assert ei.value.status == status and calls == [G31]                                         # no fallback model for a bad key
        assert "GEMINI_API_KEY" in str(ei.value) and "aistudio.google.com/apikey" in str(ei.value) and "Cloudflare" not in str(ei.value)


def test_google_quota_triggers_the_configured_fallback_like_other_remote_runners(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", G26)
    seen = []

    async def call(m):
        seen.append(m)
        if m == G31:
            raise APIStatusError(RESOURCE_EXHAUSTED, 429)
        return "ok"
    with runner_scope("aistudio"):
        assert run(with_model_fallback(call, G31)) == "ok"
    assert seen == [G31, G26]


def test_other_runners_keep_their_messages_and_openai_gets_its_own(monkeypatch):
    assert "Ollama Cloud" in str(ModelQuotaError("gemma4:31b-cloud", 402, "x"))
    assert "Cloudflare Workers AI" in str(ModelAuthError("m", 401, runner="cloudflare")) and "CF_API_TOKEN" in str(ModelAuthError("m", 401, runner="cloudflare"))
    assert "OPENAI_COMPAT_API_KEY" in str(ModelAuthError("m", 401, runner="openai"))
    assert "OpenAI-compatible" in str(ModelQuotaError("m", 429, runner="openai"))


# ------------------------------------------------------------------ the chain: UI pick -> API -> runner_scope -> client -> telemetry
def test_ui_pick_to_client_chain_for_aistudio(monkeypatch):
    monkeypatch.setattr(GoogleAIStudioRunner, "_import", lambda self, name: FakeChatBase)
    catalogue(monkeypatch, BODY)
    # 1. GET /api/v1/runners lists it, usable and remote
    row = {r["name"]: r for r in runners_payload()["runners"]}["aistudio"]
    assert (row["configured"], row["remote"], row["label"]) == (True, True, "Google AI Studio")
    # 2. GET /api/v1/models?runner=aistudio -> its list, default gemma-4-31b-it (settings.llm_model is an Ollama tag)
    models = run(models_payload("aistudio"))
    assert models["runner"] == "aistudio" and models["default"] == G31 and models["models"][:2] == [G31, G26] and "gemini-2.5-flash" in models["models"]
    # 3. POST chat with runner=aistudio and token_mode='' -> validated name
    assert check_chat_choice("AIStudio", "") == ("aistudio", "")
    # 4. runner_scope -> make_chat builds the AI Studio client with the mapped model; token mode auto => lean
    with runner_scope("aistudio"), ledger_scope() as led:
        assert get_runner().name == "aistudio" and effective_mode() == "lean" and max_input_tokens() == 60000
        llm = make_chat("gemma4:31b-cloud", temperature=0, max_tokens=2048, json_mode=True)
        assert llm.kw["model"] == G31 and llm.kw["base_url"].startswith("https://generativelanguage.googleapis.com") and is_folding(llm)
        assert "model_kwargs" not in llm.kw                                                           # gemma: no response_format
        built = model_calls._build("gemma4:26b", json_mode=True, num_predict=512, num_ctx=8192, repo_id="r", phase="ASK")
        assert built.kw["model"] == G26 and built.kw["max_tokens"] == 512
        # 5. telemetry rows are tagged with the runner
        sent = []
        cb = StdoutTelemetryCallback("gemma-4-31b-it", "r1", phase="ASK_AGENT", emitter=sent.append)
        cb.on_chat_model_start({}, [[HumanMessage(content="x")]])
        gen = types.SimpleNamespace(message=types.SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 5}, response_metadata={}),
                                    generation_info={}, text="hi")
        cb.on_llm_end(types.SimpleNamespace(generations=[[gen]], llm_output={}))
    row = json.loads(sent[0])
    assert row["runner"] == "aistudio" and row["token_mode"] == "lean" and row["model"] == "gemma-4-31b-it"
    assert led.totals()["runner"] == "aistudio"
    # outside the scope the server default is back
    assert get_runner().name == "ollama" and effective_mode() == "normal"


def test_per_request_runner_never_changes_embeddings_and_gemini_gets_json_mode(monkeypatch):
    monkeypatch.setattr(GoogleAIStudioRunner, "_import", lambda self, name: Rec)
    with runner_scope("aistudio"):
        llm = make_chat("gemini-2.5-flash", json_mode=True)
        assert llm.kw["model_kwargs"] == JSON and not is_folding(llm) and get_embedding_runner().name == "ollama"


def test_the_ui_and_api_layers_carry_labels_and_the_runner_choice():
    """FastAPI is stubbed in this environment: check the wiring in the source of the thin layers."""
    root = pathlib.Path(__file__).resolve().parent.parent
    js = (root / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "x.label || x.name" in js and "S.runnerLabels" in js and "function runnerLabel" in js
    assert "' ☁'" in js and "(not set up)" in js and "x.error || 'not configured'" in js           # cloud marker, unconfigured marker + reason tooltip
    assert "Runner: ${runnerLabel(runner)} (${runner})" in js and "runner: S.runner" in js
    assert "(t.runner ? ` · ${t.runner}`" in js and "· ${t.runner} · mode ${t.token_mode}" in js        # telemetry / footer show the runner name
    api = (root / "views" / "api_view.py").read_text(encoding="utf-8")
    assert "check_runner_config()" in api and "runners_payload()" in api
    ctl = (root / "controllers" / "engine_controller.py").read_text(encoding="utf-8")
    assert "runner_scope(runner)" in ctl
