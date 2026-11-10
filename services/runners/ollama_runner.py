"""Ollama runner: exactly what the app did before runners existed (ChatOllama / OllamaEmbeddings / GET /api/tags)."""
from __future__ import annotations

import os
from typing import Any, List, Optional

from config.settings import settings


class OllamaRunner:
    name = "ollama"
    supports = {"vision": True, "json": True, "reasoning": True, "num_ctx": True, "embeddings": True}

    def check_config(self) -> None:
        return None

    def map_model(self, model: str, *, vision: bool = False) -> str:
        return model

    def chat(self, model: str, *, temperature: float = 0, max_tokens: Optional[int] = None, json_mode: bool = False,
             reasoning: Any = None, num_ctx: Optional[int] = None, callbacks: Optional[list] = None,
             vision: bool = False, chat_cls: Any = None) -> Any:
        if chat_cls is None:
            from langchain_ollama import ChatOllama as chat_cls
        from services.model_calls import chat_base_url      # lazy: model_calls imports this package
        kw = dict(model=model, base_url=chat_base_url(), temperature=temperature)
        if num_ctx is not None:
            kw["num_ctx"] = num_ctx
        if callbacks is not None:
            kw["callbacks"] = callbacks
        if max_tokens:
            kw["num_predict"] = max_tokens      # never -1: cloud endpoints reject non-positive values
        if json_mode:
            kw["format"] = "json"
        if reasoning is not None:               # only sent when asked for: models without a thinking mode can reject the flag
            kw["reasoning"] = reasoning
        return chat_cls(**kw)

    def embeddings(self, model: Optional[str] = None, *, embeddings_cls: Any = None) -> Any:
        if embeddings_cls is None:
            from langchain_ollama import OllamaEmbeddings as embeddings_cls
        # Force the Ollama Python SDK to honor the remote base URL
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        return embeddings_cls(model=model or settings.embedding_model, base_url=settings.ollama_base_url)

    async def list_models(self) -> List[str]:
        """Models on the Ollama endpoint (embedding models hidden). Never raises."""
        from services.model_calls import chat_base_url
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(chat_base_url() + "/api/tags")
                r.raise_for_status()
                names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            names = []
        return sorted({n for n in names if n and "embed" not in n.lower()})
