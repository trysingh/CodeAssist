"""Cloudflare Workers AI runner, through its OpenAI-compatible endpoint.

    base URL  https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1   (or settings.cf_ai_gateway_url)
    auth      Authorization: Bearer {CF_API_TOKEN}
    client    langchain_openai.ChatOpenAI / OpenAIEmbeddings, imported lazily so Ollama-only installs do not need the package

Differences from Ollama that this class absorbs:
  * model ids look like `@cf/google/gemma-3-12b-it`. An Ollama-style name (has ':' and no '@', e.g. the default `gemma4:31b-cloud`
    or a `qwen2.5-coder:7b` fallback) is mapped to settings.cf_llm_model (settings.cf_vision_model for image calls);
  * `num_ctx` does not exist (the context window is the model's): ignored;
  * json mode is `response_format={"type": "json_object"}`;
  * `reasoning` has no portable equivalent on this endpoint: ignored (best effort);
  * the output cap is `max_tokens`.
"""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from config.settings import settings
from services.runners.base import RunnerConfigError

ACCOUNT_BASE = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
MODELS_SEARCH = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search"
CHAT_TASK = "Text Generation"
# Only Workers AI hosted models: the catalogue also lists proxied third-party models, which are billed and called differently.
HOSTED_PREFIXES = ("@cf/", "@hf/")
_MAX_PAGES = 10
_CACHE = {"names": [], "at": 0.0, "key": ""}   # module level: get_runner() builds a new runner per call
log = logging.getLogger(__name__)


def is_ollama_style(model: str) -> bool:
    return ":" in model and "@" not in model


class CloudflareRunner:
    name = "cloudflare"
    supports = {"vision": True, "json": True, "reasoning": False, "num_ctx": False, "embeddings": True}

    # ------------------------------------------------------------------ config
    def check_config(self) -> None:
        missing = [n for n, v in (("CF_ACCOUNT_ID", settings.cf_account_id), ("CF_API_TOKEN", settings.cf_api_token))
                   if not (v or "").strip()]
        if settings.cf_ai_gateway_url.strip() and "CF_ACCOUNT_ID" in missing:
            missing.remove("CF_ACCOUNT_ID")         # a gateway URL already carries the account
        if missing:
            raise RunnerConfigError(
                f"The 'cloudflare' runner needs {' and '.join(missing)} (set in .env, see docs/CLOUDFLARE_SETUP.md), "
                "or set LLM_RUNNER=ollama.")

    def base_url(self) -> str:
        gw = (settings.cf_ai_gateway_url or "").strip().rstrip("/")
        return gw or ACCOUNT_BASE.format(account=settings.cf_account_id.strip())

    def map_model(self, model: str, *, vision: bool = False) -> str:
        m = (model or "").strip()
        if not m or is_ollama_style(m):
            return (settings.cf_vision_model if vision else settings.cf_llm_model).strip() or settings.cf_llm_model
        return m

    # ------------------------------------------------------------------ clients
    @staticmethod
    def _import(name: str):
        try:
            import langchain_openai
            return getattr(langchain_openai, name)
        except ImportError as exc:
            raise RunnerConfigError("The 'cloudflare' runner needs the langchain-openai package: pip install langchain-openai "
                                    "(or set LLM_RUNNER=ollama).") from exc

    def chat(self, model: str, *, temperature: float = 0, max_tokens: Optional[int] = None, json_mode: bool = False,
             reasoning: Any = None, num_ctx: Optional[int] = None, callbacks: Optional[list] = None,
             vision: bool = False, chat_cls: Any = None) -> Any:
        self.check_config()
        cls = chat_cls or self._import("ChatOpenAI")
        kw = dict(model=self.map_model(model, vision=vision), base_url=self.base_url(), api_key=settings.cf_api_token.strip(),
                  temperature=temperature, timeout=settings.cf_timeout_s, max_retries=settings.cf_max_retries)
        if callbacks is not None:
            kw["callbacks"] = callbacks
        if max_tokens:
            kw["max_tokens"] = max_tokens
        if json_mode:
            kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return cls(**kw)

    def embeddings(self, model: Optional[str] = None, *, embeddings_cls: Any = None) -> Any:
        self.check_config()
        cls = embeddings_cls or self._import("OpenAIEmbeddings")
        name = (model or "").strip()
        if not name or is_ollama_style(name):       # settings.embedding_model is an Ollama name by default
            name = settings.cf_embedding_model
        # check_embedding_ctx_length=False: otherwise the client sends tiktoken token ids, which a non-OpenAI endpoint rejects
        return cls(model=name, base_url=self.base_url(), api_key=settings.cf_api_token.strip(),
                   check_embedding_ctx_length=False, timeout=settings.cf_timeout_s, max_retries=settings.cf_max_retries)

    async def _search_page(self, page: int) -> dict:
        """One page of GET /accounts/{id}/ai/models/search?task=Text Generation (needs a Workers AI Read token)."""
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(MODELS_SEARCH.format(account=settings.cf_account_id.strip()),
                            params={"task": CHAT_TASK, "page": page, "per_page": 100},
                            headers={"Authorization": f"Bearer {settings.cf_api_token.strip()}"})
            r.raise_for_status()
            return r.json()

    async def _fetch_catalog(self) -> List[str]:
        names: List[str] = []
        for page in range(1, _MAX_PAGES + 1):
            body = await self._search_page(page)
            if body.get("success") is False:
                raise RuntimeError(f"models/search failed: {body.get('errors')}")
            rows = body.get("result") or []
            for row in rows:
                name = (row.get("name") or "") if isinstance(row, dict) else ""
                task = ((row.get("task") or {}).get("name") or CHAT_TASK) if isinstance(row, dict) else ""
                if name.startswith(HOSTED_PREFIXES) and task == CHAT_TASK:
                    names.append(name)
            info = body.get("result_info") or {}
            total = info.get("total_count") or info.get("count")
            if not rows or (isinstance(total, int) and page * 100 >= total):
                break
        return sorted(set(names))

    async def list_models(self) -> List[str]:
        """CF_MODELS first, then (CF_LIST_ALL_MODELS) every hosted text-generation model on the account. Never raises."""
        pinned = list(settings.cf_model_list)
        if not settings.cf_list_all_models or not settings.cf_account_id.strip() or not settings.cf_api_token.strip():
            return pinned
        key = settings.cf_account_id.strip()
        if _CACHE["key"] != key or not _CACHE["names"] or time.monotonic() - _CACHE["at"] > settings.cf_models_cache_s:
            try:
                names = await self._fetch_catalog()
                _CACHE.update(names=names, at=time.monotonic(), key=key)
            except Exception as exc:  # noqa: BLE001  - the picker must still work: fall back to the pinned list
                log.warning("Cloudflare model list unavailable, showing CF_MODELS only: %s", exc)
                if _CACHE["key"] != key or not _CACHE["names"]:
                    return pinned
        return pinned + [m for m in _CACHE["names"] if m not in pinned]
