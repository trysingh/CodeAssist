"""Cloudflare Workers AI runner, through its OpenAI-compatible endpoint (shared logic: openai_compat.OpenAICompatRunner).

    base URL  https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1   (or settings.cf_ai_gateway_url)
    auth      Authorization: Bearer {CF_API_TOKEN}
    client    langchain_openai.ChatOpenAI / OpenAIEmbeddings, imported lazily so Ollama-only installs do not need the package

Differences from Ollama that this class absorbs:
  * model ids look like `@cf/google/gemma-3-12b-it`. An Ollama-style name (has ':' and no '@', e.g. the default `gemma4:31b-cloud`
    or a `qwen2.5-coder:7b` fallback) is mapped to settings.cf_llm_model (settings.cf_vision_model for image calls);
  * `num_ctx` does not exist (the context window is the model's): ignored;
  * json mode is `response_format={"type": "json_object"}`;
  * `reasoning` has no portable equivalent on this endpoint: ignored (best effort);
  * the output cap is `max_tokens`;
  * prompt caching: the `x-session-affinity` header carries a stable id per repo + chat thread (settings.cf_session_affinity).
This subclass only supplies the Cloudflare specifics: URL, credential check, headers, the paged models/search catalogue.
"""
from __future__ import annotations

from typing import Any, Dict, List

from config.settings import settings
from services.runners.base import RunnerConfigError
from services.runners.openai_compat import OpenAICompatRunner, is_ollama_style
from services.runners.scope import current_session_key

ACCOUNT_BASE = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
MODELS_SEARCH = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search"
SESSION_AFFINITY_HEADER = "x-session-affinity"
CHAT_TASK = "Text Generation"
# Only Workers AI hosted models: the catalogue also lists proxied third-party models, which are billed and called differently.
HOSTED_PREFIXES = ("@cf/", "@hf/")
_MAX_PAGES = 10
_CACHE = {"names": [], "at": 0.0, "key": ""}   # module level: get_runner() builds a new runner per call

__all__ = ["CloudflareRunner", "is_ollama_style", "ACCOUNT_BASE", "MODELS_SEARCH", "SESSION_AFFINITY_HEADER"]


class CloudflareRunner(OpenAICompatRunner):
    name = "cloudflare"
    label = "Cloudflare Workers AI"
    remote = True
    supports = {"vision": True, "json": True, "reasoning": False, "num_ctx": False, "embeddings": True, "web_search": False}

    @property
    def max_input_tokens(self) -> int:      # type: ignore[override]
        return max(0, int(settings.cf_max_input_tokens))

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

    def api_key(self) -> str:
        return settings.cf_api_token.strip()

    def llm_default(self, vision: bool = False) -> str:
        return (settings.cf_vision_model if vision else settings.cf_llm_model).strip() or settings.cf_llm_model

    def embedding_default(self) -> str:
        return settings.cf_embedding_model

    timeout_setting = "CF_TIMEOUT_S"

    def timeout_s(self) -> float:
        return settings.cf_timeout_s

    def max_retries(self) -> int:
        return settings.cf_max_retries

    def extra_headers(self) -> Dict[str, str]:
        sid = current_session_key()
        if settings.cf_session_affinity and sid:       # same id for every call of one chat thread -> prompt-prefix cache hits
            return {SESSION_AFFINITY_HEADER: sid}
        return {}

    # ------------------------------------------------------------------ model picker
    def pinned_models(self) -> List[str]:
        return list(settings.cf_model_list)

    def list_all_models(self) -> bool:
        return bool(settings.cf_list_all_models)

    def can_list_remote(self) -> bool:
        return bool(settings.cf_account_id.strip() and settings.cf_api_token.strip())

    def cache(self) -> Dict[str, Any]:
        return _CACHE

    def cache_key(self) -> str:
        return settings.cf_account_id.strip()

    def cache_ttl_s(self) -> float:
        return settings.cf_models_cache_s

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
