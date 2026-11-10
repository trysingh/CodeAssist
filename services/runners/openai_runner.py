"""Generic OpenAI-compatible runner: any server that speaks /chat/completions (OpenRouter, Groq, vLLM, LM Studio, llama.cpp ...).

Configured purely by settings (no base URL = "not set up" in the UI):
    OPENAI_COMPAT_BASE_URL         e.g. https://openrouter.ai/api/v1, https://api.groq.com/openai/v1, http://localhost:8000/v1
    OPENAI_COMPAT_API_KEY          optional (local servers); sent as Authorization: Bearer
    OPENAI_COMPAT_MODELS           comma list shown in the picker (pinned)
    OPENAI_COMPAT_LLM_MODEL        what an empty name / Ollama-style name (gemma4:31b-cloud) is mapped to
    OPENAI_COMPAT_TIMEOUT_S        seconds per attempt (default COMPAT_TIMEOUT_S = 120); OPENAI_COMPAT_MAX_RETRIES (default COMPAT_MAX_RETRIES = 2)
    OPENAI_COMPAT_EMBEDDING_MODEL  needed only when EMBEDDING_RUNNER=openai
    OPENAI_COMPAT_REMOTE           true (default) = billed service -> TOKEN_MODE=auto runs lean; false for a local server
    OPENAI_COMPAT_LIST_ALL_MODELS  also list GET {base}/models (cached)
"""
from __future__ import annotations

from typing import List

from config.settings import settings
from services.runners.base import RunnerConfigError
from services.runners.openai_compat import OpenAICompatRunner, is_ollama_style


class OpenAIRunner(OpenAICompatRunner):
    name = "openai"
    label = "OpenAI-compatible"
    supports = {"vision": True, "json": True, "reasoning": False, "num_ctx": False, "embeddings": True}

    @property
    def remote(self) -> bool:               # type: ignore[override]
        return bool(settings.openai_compat_remote)

    @property
    def max_input_tokens(self) -> int:      # type: ignore[override]
        return max(0, int(settings.openai_compat_max_input_tokens))

    timeout_setting = "OPENAI_COMPAT_TIMEOUT_S"

    def timeout_s(self) -> float:
        t = settings.openai_compat_timeout_s
        return settings.compat_timeout_s if t is None else t

    def max_retries(self) -> int:
        r = settings.openai_compat_max_retries
        return settings.compat_max_retries if r is None else max(0, int(r))

    def check_config(self) -> None:
        if not (settings.openai_compat_base_url or "").strip():
            raise RunnerConfigError("The 'openai' runner is not set up: set OPENAI_COMPAT_BASE_URL (e.g. https://openrouter.ai/api/v1 or "
                                    "http://localhost:8000/v1) and optionally OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_LLM_MODEL in .env.")

    def base_url(self) -> str:
        return (settings.openai_compat_base_url or "").strip().rstrip("/")

    def api_key(self) -> str:
        return (settings.openai_compat_api_key or "").strip()

    def sdk_key(self) -> str:
        return self.api_key() or "not-needed"       # the OpenAI client refuses an empty key; local servers ignore it

    def llm_default(self, vision: bool = False) -> str:
        name = (settings.openai_compat_llm_model or "").strip() or (self.pinned_models() or [""])[0]
        if not name:
            raise RunnerConfigError("The 'openai' runner has no model to use: set OPENAI_COMPAT_LLM_MODEL (or OPENAI_COMPAT_MODELS).")
        return name

    def embedding_default(self) -> str:
        name = (settings.openai_compat_embedding_model or "").strip()
        if not name:
            raise RunnerConfigError("EMBEDDING_RUNNER=openai needs OPENAI_COMPAT_EMBEDDING_MODEL (and a re-index, see README).")
        return name

    def is_foreign_name(self, model: str) -> bool:
        # OpenRouter ids such as `meta-llama/llama-3.3-70b-instruct:free` have ':' too: only a slash-less `name:tag` is an Ollama tag,
        # and a model the user pinned is always theirs.
        return is_ollama_style(model) and "/" not in model and model not in self.pinned_models()

    def pinned_models(self) -> List[str]:
        return list(settings.openai_compat_model_list)

    def list_all_models(self) -> bool:
        return bool(settings.openai_compat_list_all_models)

    def can_list_remote(self) -> bool:
        return bool(self.base_url_or_empty())

    def cache_key(self) -> str:
        return f"{self.base_url_or_empty()}|{self.api_key()[-6:]}"
