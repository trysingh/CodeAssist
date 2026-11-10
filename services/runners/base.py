"""Runner contract: what the app needs from an inference backend.

A runner builds LangChain chat / embedding clients for one backend (Ollama, Cloudflare Workers AI, ...). Call sites never
import a provider package; they ask `services.runners` for a client, so the backend is a config choice (settings.llm_runner /
settings.embedding_runner). To add a runner: implement this protocol in a new module and `register_runner("name", Factory)`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Same shape model_calls.model_scope and llm_errors accept. '@' is there for Cloudflare ids such as @cf/google/gemma-3-12b-it.
MODEL_NAME = re.compile(r"[@\w.:/\-]{1,100}")


class RunnerConfigError(RuntimeError):
    """The selected runner cannot work as configured (missing credentials, missing package, unknown runner name)."""


@runtime_checkable
class Runner(Protocol):
    name: str
    # What the backend can do: {"vision": bool, "json": bool, "reasoning": bool, "num_ctx": bool, "embeddings": bool}
    supports: Dict[str, bool]

    def check_config(self) -> None:
        """Raise RunnerConfigError when the runner cannot be used as configured. Never touches the network."""

    def map_model(self, model: str, *, vision: bool = False) -> str:
        """The model id this backend should receive for a requested name (identity for Ollama)."""

    def chat(self, model: str, *, temperature: float = 0, max_tokens: Optional[int] = None, json_mode: bool = False,
             reasoning: Any = None, num_ctx: Optional[int] = None, callbacks: Optional[list] = None,
             vision: bool = False, chat_cls: Any = None) -> Any:
        """A LangChain chat model. `chat_cls` replaces the provider class (a patchable seam for the call sites / tests)."""

    def embeddings(self, model: Optional[str] = None, *, embeddings_cls: Any = None) -> Any:
        """A LangChain Embeddings object (model None = the configured embedding model of this runner)."""

    async def list_models(self) -> List[str]:
        """Chat models to offer in the UI picker. Never raises."""
