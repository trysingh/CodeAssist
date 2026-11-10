"""Pluggable inference runners. `get_runner()` is the only thing call sites need.

    from services.runners import get_runner, make_chat
    llm = make_chat(model, temperature=0, max_tokens=2048, json_mode=True, callbacks=[...])

Chat and embeddings are chosen separately (settings.llm_runner / settings.embedding_runner) because vectors from different
embedding models are not comparable: changing the embedding runner needs a re-index (reset_repo_index.py).
The chat runner can also be picked per request (the UI runner switch): `with runner_scope(name):` makes get_runner() / make_chat() /
resolve_model() with no name use it. Embeddings IGNORE that scope and always use settings.embedding_runner.
Add a backend: write a class following base.Runner and call register_runner("name", factory).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config.settings import settings
from services.runners.base import MODEL_NAME, Runner, RunnerConfigError
from services.runners.cloudflare_runner import CloudflareRunner
from services.runners.ollama_runner import OllamaRunner
from services.runners.scope import current_runner_name, current_session_key, runner_scope, session_key, thread_scope

_REGISTRY: Dict[str, Callable[[], Runner]] = {"ollama": OllamaRunner, "cloudflare": CloudflareRunner}


def register_runner(name: str, factory: Callable[[], Runner]) -> None:
    _REGISTRY[name.strip().lower()] = factory


def available_runners() -> List[str]:
    return sorted(_REGISTRY)


def get_runner(name: Optional[str] = None) -> Runner:
    """The runner called `name`; None = the request's runner_scope, else settings.llm_runner. Unknown name -> RunnerConfigError."""
    key = (current_runner_name() if name is None else name or "ollama").strip().lower() or "ollama"
    try:
        return _REGISTRY[key]()
    except KeyError:
        raise RunnerConfigError(f"Unknown runner '{key}'. Available: {', '.join(available_runners())}.") from None


def get_embedding_runner() -> Runner:
    """Always settings.embedding_runner, never the per-request chat runner (index compatibility)."""
    return get_runner(settings.embedding_runner)


def runner_status() -> List[dict]:
    """Every registered runner: {"name", "remote", "configured", "error"} (check_config only, never touches the network)."""
    out = []
    for key in available_runners():
        remote, err = False, None
        try:
            r = _REGISTRY[key]()
            remote = bool(getattr(r, "remote", False))
            r.check_config()
        except RunnerConfigError as exc:
            err = str(exc)
        except Exception as exc:  # noqa: BLE001 - a broken third-party runner must not break the list
            err = f"{type(exc).__name__}: {exc}"
        out.append({"name": key, "remote": remote, "configured": err is None, "error": err})
    return out


def require_runner(name: Optional[str]) -> str:
    """Normalised runner name for a request, or RunnerConfigError (unknown or not configured). Empty/None = '' (use the default)."""
    key = (name or "").strip().lower()
    if not key:
        return ""
    r = get_runner(key)             # unknown -> RunnerConfigError
    r.check_config()                # not configured -> RunnerConfigError
    return key


def make_chat(model: str, *, ollama_cls: Any = None, **kw: Any) -> Any:
    """Chat client from the LLM runner of this request (runner_scope, else settings.llm_runner; kwargs: see base.Runner.chat).

    `ollama_cls`: the ChatOllama class the CALLING module holds (rag_graph.ChatOllama, patched by tests). Only the Ollama runner uses it."""
    runner = get_runner()
    if ollama_cls is not None and runner.name == "ollama":
        kw["chat_cls"] = ollama_cls
    return runner.chat(model, **kw)


def resolve_model(model: str = "", *, vision: bool = False) -> str:
    """The model id the request's runner will actually use for `model` ('' = settings.llm_model)."""
    return get_runner().map_model(model or settings.llm_model, vision=vision)


def check_runner_config() -> None:
    """Fail fast (RunnerConfigError) when a selected runner is unusable. Called at startup; no network."""
    get_runner().check_config()
    get_embedding_runner().check_config()


__all__ = ["MODEL_NAME", "Runner", "RunnerConfigError", "available_runners", "check_runner_config", "current_runner_name",
           "current_session_key", "get_embedding_runner", "get_runner", "make_chat", "register_runner", "require_runner",
           "resolve_model", "runner_scope", "runner_status", "session_key", "thread_scope"]
