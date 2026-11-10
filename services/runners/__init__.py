"""Pluggable inference runners. `get_runner()` is the only thing call sites need.

    from services.runners import get_runner, make_chat
    llm = make_chat(model, temperature=0, max_tokens=2048, json_mode=True, callbacks=[...])

Chat and embeddings are chosen separately (settings.llm_runner / settings.embedding_runner) because vectors from different
embedding models are not comparable: changing the embedding runner needs a re-index (reset_repo_index.py).
Add a backend: write a class following base.Runner and call register_runner("name", factory).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config.settings import settings
from services.runners.base import MODEL_NAME, Runner, RunnerConfigError
from services.runners.cloudflare_runner import CloudflareRunner
from services.runners.ollama_runner import OllamaRunner

_REGISTRY: Dict[str, Callable[[], Runner]] = {"ollama": OllamaRunner, "cloudflare": CloudflareRunner}


def register_runner(name: str, factory: Callable[[], Runner]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def available_runners() -> List[str]:
    return sorted(_REGISTRY)


def get_runner(name: Optional[str] = None) -> Runner:
    """The runner called `name`; None = settings.llm_runner. Unknown name -> RunnerConfigError."""
    key = (settings.llm_runner if name is None else name or "ollama").strip().lower() or "ollama"
    try:
        return _REGISTRY[key]()
    except KeyError:
        raise RunnerConfigError(f"Unknown runner '{key}'. Available: {', '.join(available_runners())}.") from None


def get_embedding_runner() -> Runner:
    return get_runner(settings.embedding_runner)


def make_chat(model: str, *, ollama_cls: Any = None, **kw: Any) -> Any:
    """Chat client from the configured LLM runner (kwargs: see base.Runner.chat).

    `ollama_cls`: the ChatOllama class the CALLING module holds (rag_graph.ChatOllama, patched by tests). Only the Ollama runner uses it."""
    runner = get_runner()
    if ollama_cls is not None and runner.name == "ollama":
        kw["chat_cls"] = ollama_cls
    return runner.chat(model, **kw)


def resolve_model(model: str = "", *, vision: bool = False) -> str:
    """The model id the configured runner will actually use for `model` ('' = settings.llm_model)."""
    return get_runner().map_model(model or settings.llm_model, vision=vision)


def check_runner_config() -> None:
    """Fail fast (RunnerConfigError) when a selected runner is unusable. Called at startup; no network."""
    get_runner().check_config()
    get_embedding_runner().check_config()


__all__ = ["MODEL_NAME", "Runner", "RunnerConfigError", "available_runners", "check_runner_config", "get_embedding_runner",
           "get_runner", "make_chat", "register_runner", "resolve_model"]
