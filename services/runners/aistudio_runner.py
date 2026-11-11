"""Google AI Studio (Gemini API) runner, through its OpenAI-compatible endpoint (shared logic: openai_compat.OpenAICompatRunner).

    base URL  https://generativelanguage.googleapis.com/v1beta/openai/   (AISTUDIO_BASE_URL)
    auth      Authorization: Bearer {GEMINI_API_KEY}   (GOOGLE_API_KEY is accepted too)
    models    gemma-4-31b-it, gemma-4-26b-a4b-it, gemini-* ... (GET {base}/models lists them, ids may carry a "models/" prefix)

AI Studio specifics this class absorbs:
  * an empty name or the Ollama-style global default (ENGINE_LLM_MODEL = gemma4:31b-cloud, what a request carries when the UI picker is on
    "Default") means AISTUDIO_LLM_MODEL (vision calls: AISTUDIO_VISION_MODEL); MODEL_MAP (gemma4:26b -> gemma-4-26b-a4b-it ...) only
    applies to OTHER explicit Ollama-style names (e.g. a fallback model); any other `x:y` -> AISTUDIO_LLM_MODEL;
  * older Gemma on the Gemini API rejects system instructions: AISTUDIO_FOLD_SYSTEM (auto = on for gemma*) merges the system
    prompt into the first user message (see openai_compat.folding_chat_cls);
  * json mode (response_format=json_object) is sent for gemini-* only (AISTUDIO_JSON_MODE auto|on|off): Gemma gets plain text
    and the app's loose JSON parser does the rest;
  * no session-affinity header (the API caches implicitly).
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

from config.settings import settings
from services.runners.base import RunnerConfigError
from services.runners.openai_compat import OpenAICompatRunner, folding_chat_cls

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY_URL = "https://aistudio.google.com/apikey"
RATE_LIMITS_URL = "https://ai.google.dev/gemini-api/docs/rate-limits"

# Ollama-style name (regex on the lower-cased name) -> AI Studio id. First match wins; extend here. Anything else with a ':'
# goes to settings.aistudio_llm_model. Applies only to an EXPLICIT Ollama-style name that is not settings.llm_model (the global
# default always means AISTUDIO_LLM_MODEL, see OpenAICompatRunner.map_model) and never to a vision call.
MODEL_MAP: List[Tuple[str, str]] = [
    (r"^gemma-?4:31b", "gemma-4-31b-it"),
    (r"^gemma-?4:26b", "gemma-4-26b-a4b-it"),
]
CHAT_PREFIXES = ("gemini", "gemma")
# Not chat models (or not usable through chat/completions): embeddings, image / video / speech generation, live, AQA.
_NOT_CHAT = re.compile(r"embed|imagen|veo|tts|audio|live|aqa|image", re.IGNORECASE)
_MODES = ("auto", "on", "off")
_EFFORTS = ("none", "minimal", "low", "medium", "high")


def strip_models_prefix(model_id: str) -> str:
    m = (model_id or "").strip()
    return m[len("models/"):] if m.startswith("models/") else m


def _mode(value: str) -> str:
    v = (value or "auto").strip().lower()
    return v if v in _MODES else "auto"


class GoogleAIStudioRunner(OpenAICompatRunner):
    name = "aistudio"
    label = "Google AI Studio"
    remote = True
    supports = {"vision": True, "json": True, "reasoning": False, "num_ctx": False, "embeddings": True, "web_search": False}

    @property
    def max_input_tokens(self) -> int:      # type: ignore[override]
        return max(0, int(settings.aistudio_max_input_tokens))

    # ------------------------------------------------------------------ config
    def check_config(self) -> None:
        if not (settings.gemini_api_key or "").strip():
            raise RunnerConfigError(f"The 'aistudio' runner needs GEMINI_API_KEY (or GOOGLE_API_KEY): create one at {API_KEY_URL} and put it "
                                    "in .env (see docs/GOOGLE_AI_STUDIO_SETUP.md), or set LLM_RUNNER=ollama.")

    def base_url(self) -> str:
        return (settings.aistudio_base_url or "").strip().rstrip("/") or DEFAULT_BASE_URL.rstrip("/")

    def api_key(self) -> str:
        return settings.gemini_api_key.strip()

    # ------------------------------------------------------------------ models
    def llm_default(self, vision: bool = False) -> str:
        return (settings.aistudio_vision_model if vision else settings.aistudio_llm_model).strip() or settings.aistudio_llm_model

    def embedding_default(self) -> str:
        return settings.aistudio_embedding_model

    timeout_setting = "AISTUDIO_TIMEOUT_S"

    def timeout_s(self) -> float:
        return settings.aistudio_timeout_s

    def max_retries(self) -> int:
        return max(0, int(settings.aistudio_max_retries))

    def map_model(self, model: str, *, vision: bool = False) -> str:
        return super().map_model(strip_models_prefix(model), vision=vision)

    def map_foreign(self, model: str, vision: bool = False) -> str:
        if vision:
            return self.llm_default(True)
        low = model.lower()
        for pattern, target in MODEL_MAP:
            if re.match(pattern, low):
                return target
        return self.llm_default()

    def reasoning_effort(self, reasoning: Any = None) -> str:
        """The reasoning_effort to send, or '' (not sent). AISTUDIO_REASONING_EFFORT: '' never; none|minimal|low|medium|high always;
        auto = follow the call's `reasoning` argument (ASK_REASONING: off -> none, low / medium / high -> that level)."""
        v = (settings.aistudio_reasoning_effort or "").strip().lower()
        if v == "auto":
            if reasoning is False:
                return "none"
            return reasoning if isinstance(reasoning, str) and reasoning in _EFFORTS else ""
        return v if v in _EFFORTS else ""

    def chat_extra(self, model: str, reasoning: Any = None) -> dict:
        effort = self.reasoning_effort(reasoning)
        return {"reasoning_effort": effort} if effort else {}

    def fold_system(self, model: str) -> bool:
        mode = _mode(settings.aistudio_fold_system)
        return mode == "on" or (mode == "auto" and strip_models_prefix(model).lower().startswith("gemma"))

    def json_enabled(self, model: str) -> bool:
        mode = _mode(settings.aistudio_json_mode)
        return mode == "on" or (mode == "auto" and strip_models_prefix(model).lower().startswith("gemini"))

    def chat_class(self, cls: Any, model: str) -> Any:
        return folding_chat_cls(cls) if self.fold_system(model) else cls

    # ------------------------------------------------------------------ model picker
    def pinned_models(self) -> List[str]:
        return list(settings.aistudio_model_list)

    def list_all_models(self) -> bool:
        return bool(settings.aistudio_list_all_models)

    def can_list_remote(self) -> bool:
        return bool(settings.gemini_api_key.strip())

    def cache_key(self) -> str:
        return f"{self.base_url()}|{self.api_key()[-6:]}"

    @staticmethod
    def catalog_id(raw: str) -> str:
        return strip_models_prefix(raw)

    def catalog_keep(self, model_id: str) -> bool:
        return model_id.lower().startswith(CHAT_PREFIXES) and not _NOT_CHAT.search(model_id)
