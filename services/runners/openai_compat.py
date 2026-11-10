"""Shared base of every runner that talks to an OpenAI-style HTTP API (Cloudflare Workers AI, Google AI Studio, generic).

    client    langchain_openai.ChatOpenAI / OpenAIEmbeddings, imported lazily so Ollama-only installs do not need the package
    chat      base_url + api_key + temperature + timeout/retries (+ callbacks, max_tokens, json mode, extra headers)

A subclass only says WHAT differs; everything below is shared:

    name / label / remote / supports / max_input_tokens
    base_url(), api_key(), check_config()          where to call and whether it is set up
    llm_default(vision), embedding_default()       the model ids that Ollama-style names (`gemma4:31b-cloud`) are mapped to
    map_model(model, vision=)                      the mapping (see "Default-model semantics" below), map_foreign() is its hook
    timeout_s(), max_retries(), timeout_setting     per-attempt timeout, client retries, and the env name that raises the timeout (error messages)
    chat_extra(model, reasoning)                    extra ChatOpenAI kwargs (AI Studio: reasoning_effort)
    json_enabled(model)                            send response_format=json_object for this model?
    extra_headers()                                default_headers hook (Cloudflare: session affinity)
    chat_class(cls, model)                         last chance to wrap the client class (AI Studio: system-prompt folding)
    pinned_models() / list_all_models() / cache_ttl_s() / cache()   the model picker
    _fetch_catalog()                                the remote model catalogue (default: GET {base_url}/models, OpenAI list shape)

To add another OpenAI-style backend: subclass OpenAICompatRunner, fill those in, `register_runner("name", Cls)`.

Default-model semantics (every runner here; Ollama keeps its own ids):
    * an empty name, or an Ollama-style name that equals settings.llm_model (the Ollama-side global default that a request carries when
      the UI picker is on "Default"), means "this runner's configured default": AISTUDIO_LLM_MODEL / CF_LLM_MODEL / OPENAI_COMPAT_LLM_MODEL
      (vision calls: the runner's vision default);
    * any OTHER Ollama-style name (e.g. a fallback model) goes through map_foreign(): the base class returns the default again, AI Studio
      first tries its MODEL_MAP table (gemma4:26b -> gemma-4-26b-a4b-it);
    * a native id (gemini-2.5-flash, @cf/..., meta-llama/...) is used as is.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import settings
from services.runners.base import RunnerConfigError

log = logging.getLogger(__name__)


def is_ollama_style(model: str) -> bool:
    """`gemma4:31b-cloud`, `qwen2.5-coder:7b`: an Ollama tag (has ':' and no '@'), never a valid id on an OpenAI-style endpoint."""
    return ":" in model and "@" not in model


# ---------------------------------------------------------------- system prompt folding (models that reject system instructions)
def _text_of_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type", "text") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)


def _with_content(msg: Any, content: Any) -> Any:
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update={"content": content})
    clone = copy.copy(msg)
    clone.content = content
    return clone


def fold_system_messages(messages: List[Any]) -> List[Any]:
    """Messages without any system message: their text is prefixed (plus a blank line) to the FIRST human message.

    No human message -> the system text becomes one. Content blocks (vision) keep their images: the text goes in front as a block.
    A list without system messages is returned unchanged (same objects)."""
    msgs = list(messages)
    system = [m for m in msgs if getattr(m, "type", "") == "system"]
    if not system:
        return msgs
    prefix = "\n\n".join(t for t in (_text_of_content(m.content).strip() for m in system) if t)
    rest = [m for m in msgs if getattr(m, "type", "") != "system"]
    if not prefix:
        return rest
    for i, m in enumerate(rest):
        if getattr(m, "type", "") == "human":
            if isinstance(m.content, list):
                rest[i] = _with_content(m, [{"type": "text", "text": prefix + "\n\n"}] + list(m.content))
            else:
                rest[i] = _with_content(m, f"{prefix}\n\n{m.content}")
            return rest
    from langchain_core.messages import HumanMessage
    return [HumanMessage(content=prefix)] + rest


_FOLD_CLASSES: Dict[Any, Any] = {}


def folding_chat_cls(base: Any) -> Any:
    """A subclass of the chat class `base` (ChatOpenAI) that folds system messages into the first human message before the
    request payload is built. `_get_request_payload` is the one place sync / async / streaming calls all go through, and it
    leaves ainvoke / invoke / bind / with_retry / callbacks untouched. Cached per base class."""
    cls = _FOLD_CLASSES.get(base)
    if cls is None:
        class FoldingChat(base):                    # type: ignore[misc, valid-type]
            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                msgs = self._convert_input(input_).to_messages()
                return super()._get_request_payload(fold_system_messages(msgs), stop=stop, **kwargs)
        FoldingChat.__name__ = FoldingChat.__qualname__ = f"Folding{getattr(base, '__name__', 'Chat')}"
        cls = _FOLD_CLASSES[base] = FoldingChat
    return cls


def is_folding(llm: Any) -> bool:
    return type(llm) in _FOLD_CLASSES.values()


# ---------------------------------------------------------------- the runner
class OpenAICompatRunner:
    name = "openai"
    label = "OpenAI-compatible"
    remote = True
    supports = {"vision": True, "json": True, "reasoning": False, "num_ctx": False, "embeddings": True}
    max_input_tokens = 0
    _CACHE: Dict[str, Any] = {}                     # per subclass (see cache()); module level state: get_runner() builds a new runner per call

    # ------------------------------------------------------------------ hooks (override)
    def base_url(self) -> str:
        raise NotImplementedError

    def api_key(self) -> str:
        return ""

    def sdk_key(self) -> str:
        """The key handed to the OpenAI client (it refuses an empty one); api_key() stays the real value for the catalogue request."""
        return self.api_key()

    def check_config(self) -> None:
        raise NotImplementedError

    def llm_default(self, vision: bool = False) -> str:
        raise NotImplementedError

    def embedding_default(self) -> str:
        raise NotImplementedError

    def is_foreign_name(self, model: str) -> bool:
        """Is `model` an id of another backend (an Ollama tag) that must be mapped?"""
        return is_ollama_style(model)

    def is_default_alias(self, model: str) -> bool:
        """An empty name, or the Ollama-style global default (settings.llm_model): both mean "my configured default"."""
        m = (model or "").strip()
        return not m or (self.is_foreign_name(m) and m == (settings.llm_model or "").strip())

    def map_foreign(self, model: str, vision: bool = False) -> str:
        """An Ollama-style name that is not the global default (a fallback / role model). Base: this runner's default."""
        return self.llm_default(vision)

    def map_model(self, model: str, *, vision: bool = False) -> str:
        m = (model or "").strip()
        if self.is_default_alias(m):
            return self.llm_default(vision)
        if self.is_foreign_name(m):
            return self.map_foreign(m, vision)
        return m

    def json_enabled(self, model: str) -> bool:
        return True

    def extra_headers(self) -> Dict[str, str]:
        return {}

    def chat_class(self, cls: Any, model: str) -> Any:
        return cls

    timeout_setting = "COMPAT_TIMEOUT_S"            # the env name that raises timeout_s() (named in the timeout error message)

    def timeout_s(self) -> float:
        return settings.compat_timeout_s

    def max_retries(self) -> int:
        return settings.compat_max_retries

    def chat_extra(self, model: str, reasoning: Any = None) -> Dict[str, Any]:
        """Extra ChatOpenAI keyword arguments for this model (default none; AI Studio: reasoning_effort)."""
        return {}

    # model picker
    def pinned_models(self) -> List[str]:
        return []

    def list_all_models(self) -> bool:
        return False

    def can_list_remote(self) -> bool:
        return bool(self.base_url_or_empty())

    def base_url_or_empty(self) -> str:
        try:
            return self.base_url()
        except Exception:  # noqa: BLE001
            return ""

    def cache(self) -> Dict[str, Any]:
        cls = type(self)
        if "_CACHE" not in cls.__dict__:
            cls._CACHE = {"names": [], "at": 0.0, "key": ""}
        return cls._CACHE

    def cache_key(self) -> str:
        return f"{self.base_url_or_empty()}|{self.api_key()[-6:]}"

    def cache_ttl_s(self) -> float:
        return settings.compat_models_cache_s

    def catalog_keep(self, model_id: str) -> bool:
        return True

    @staticmethod
    def catalog_id(raw: str) -> str:
        return (raw or "").strip()

    async def _get_json(self, url: str, headers: Dict[str, str]) -> Any:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url, headers=headers)
            r.raise_for_status()
            return r.json()

    async def _fetch_catalog(self) -> List[str]:
        """GET {base_url}/models (OpenAI list shape {"data": [{"id": ...}]}) -> sorted chat-capable ids."""
        headers = {"Authorization": f"Bearer {self.api_key()}"} if self.api_key() else {}
        body = await self._get_json(self.base_url().rstrip("/") + "/models", headers)
        rows = body.get("data") if isinstance(body, dict) else body
        names = set()
        for row in rows or []:
            if isinstance(row, dict):
                raw = row.get("id") or row.get("name") or ""
            else:
                raw = row if isinstance(row, str) else ""
            mid = self.catalog_id(raw)
            if mid and self.catalog_keep(mid):
                names.add(mid)
        return sorted(names)

    # ------------------------------------------------------------------ clients
    def _import(self, name: str):
        try:
            import langchain_openai
            return getattr(langchain_openai, name)
        except ImportError as exc:
            raise RunnerConfigError(f"The '{self.name}' runner needs the langchain-openai package: pip install langchain-openai "
                                    "(or set LLM_RUNNER=ollama).") from exc

    def chat(self, model: str, *, temperature: float = 0, max_tokens: Optional[int] = None, json_mode: bool = False,
             reasoning: Any = None, num_ctx: Optional[int] = None, callbacks: Optional[list] = None,
             vision: bool = False, chat_cls: Any = None) -> Any:
        self.check_config()
        cls = chat_cls or self._import("ChatOpenAI")
        mapped = self.map_model(model, vision=vision)
        kw = dict(model=mapped, base_url=self.base_url(), api_key=self.sdk_key(),
                  temperature=temperature, timeout=self.timeout_s(), max_retries=self.max_retries())
        if callbacks is not None:
            kw["callbacks"] = callbacks
        if max_tokens:
            kw["max_tokens"] = max_tokens
        if json_mode and self.json_enabled(mapped):
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        headers = self.extra_headers()
        if headers:
            kw["default_headers"] = headers
        kw.update(self.chat_extra(mapped, reasoning))
        return self.chat_class(cls, mapped)(**kw)

    def embeddings(self, model: Optional[str] = None, *, embeddings_cls: Any = None) -> Any:
        self.check_config()
        cls = embeddings_cls or self._import("OpenAIEmbeddings")
        name = (model or "").strip()
        if not name or is_ollama_style(name):       # settings.embedding_model is an Ollama name by default
            name = self.embedding_default()
        # check_embedding_ctx_length=False: otherwise the client sends tiktoken token ids, which a non-OpenAI endpoint rejects
        return cls(model=name, base_url=self.base_url(), api_key=self.sdk_key(),
                   check_embedding_ctx_length=False, timeout=self.timeout_s(), max_retries=self.max_retries())

    # ------------------------------------------------------------------ model picker
    async def list_models(self) -> List[str]:
        """Pinned models first, then (when enabled and configured) the remote catalogue, cached. Never raises."""
        pinned: List[str] = []
        try:
            pinned = list(self.pinned_models())
            if not self.list_all_models() or not self.can_list_remote():
                return pinned
            cache, key = self.cache(), self.cache_key()
            if cache["key"] != key or not cache["names"] or time.monotonic() - cache["at"] > self.cache_ttl_s():
                try:
                    names = await self._fetch_catalog()
                    cache.update(names=names, at=time.monotonic(), key=key)
                except Exception as exc:  # noqa: BLE001  - the picker must still work: fall back to the pinned list
                    log.warning("%s model list unavailable, showing the pinned models only: %s", self.label, exc)
                    if cache["key"] != key or not cache["names"]:
                        return pinned
            return pinned + [m for m in cache["names"] if m not in pinned]
        except Exception:  # noqa: BLE001
            return pinned
