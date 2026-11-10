"""Pluggable inference runners. `get_runner()` is the only thing call sites need.

    from services.runners import get_runner, make_chat
    llm = make_chat(model, temperature=0, max_tokens=2048, json_mode=True, callbacks=[...])

Chat and embeddings are chosen separately (settings.llm_runner / settings.embedding_runner) because vectors from different
embedding models are not comparable: changing the embedding runner needs a re-index (reset_repo_index.py).
The chat runner can also be picked per request (the UI runner switch): `with runner_scope(name):` makes get_runner() / make_chat() /
resolve_model() with no name use it. Embeddings IGNORE that scope and always use settings.embedding_runner.
Add a backend: write a class following base.Runner and call register_runner("name", factory). For any OpenAI-style API subclass
openai_compat.OpenAICompatRunner (see aistudio_runner.py: ~100 lines). Built in: ollama, cloudflare, aistudio, openai.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config.settings import settings
from services.runners.base import MODEL_NAME, Runner, RunnerConfigError
from services.runners.aistudio_runner import GoogleAIStudioRunner
from services.runners.cloudflare_runner import CloudflareRunner
from services.runners.ollama_runner import OllamaRunner
from services.runners.openai_compat import OpenAICompatRunner
from services.runners.openai_runner import OpenAIRunner
from services.runners.scope import current_runner_name, current_session_key, runner_scope, session_key, thread_scope

_REGISTRY: Dict[str, Callable[[], Runner]] = {"ollama": OllamaRunner, "cloudflare": CloudflareRunner, "aistudio": GoogleAIStudioRunner,
                                              "openai": OpenAIRunner}


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
    """Every registered runner: {"name", "label", "remote", "configured", "error", "timeout_s", "max_retries"} (check_config only, never touches
    the network). timeout_s = seconds one request may take per attempt (None = no client-side limit, as for Ollama), max_retries = how often
    the client repeats a timed-out / failed request: a hang costs up to timeout_s x (max_retries + 1)."""
    out = []
    for key in available_runners():
        remote, err, label, timeout_s, retries = False, None, key, None, 0
        try:
            r = _REGISTRY[key]()
            remote = bool(getattr(r, "remote", False))
            label = str(getattr(r, "label", "") or key)
            t, n = getattr(r, "timeout_s", None), getattr(r, "max_retries", None)
            timeout_s = float(t()) if callable(t) and t() else None
            retries = int(n()) if callable(n) else 0
            r.check_config()
        except RunnerConfigError as exc:
            err = str(exc)
        except Exception as exc:  # noqa: BLE001 - a broken third-party runner must not break the list
            err = f"{type(exc).__name__}: {exc}"
        out.append({"name": key, "label": label, "remote": remote, "configured": err is None, "error": err,
                    "timeout_s": timeout_s, "max_retries": retries})
    return out


def require_runner(name: Optional[str]) -> str:
    """Normalised runner name for a request, or RunnerConfigError (unknown or not configured). Empty/None = '' (use the default)."""
    key = (name or "").strip().lower()
    if not key:
        return ""
    r = get_runner(key)             # unknown -> RunnerConfigError
    r.check_config()                # not configured -> RunnerConfigError
    return key


def _telemetry_callbacks(phase: str) -> list:
    """Default telemetry for a chat client built without callbacks, so every call is counted (tokens, cached tokens, ledger)."""
    try:
        from services.action_log import current_repo_id
        from services.model_calls import _callbacks
        return _callbacks("", current_repo_id() or "", phase)
    except Exception:  # noqa: BLE001
        return []


def make_chat(model: str, *, ollama_cls: Any = None, phase: str = "CHAT", **kw: Any) -> Any:
    """Chat client from the LLM runner of this request (runner_scope, else settings.llm_runner; kwargs: see base.Runner.chat).

    `ollama_cls`: the ChatOllama class the CALLING module holds (rag_graph.ChatOllama, patched by tests). Only the Ollama runner uses it.
    `phase`: telemetry label used when the caller passes no callbacks (a telemetry callback is then added, so no call goes uncounted).
    Telemetry callbacks are told the model id the runner really calls (model: gemma-4-31b-it) and the name that was asked for
    (requested_model: gemma4:31b-cloud)."""
    runner = get_runner()
    if ollama_cls is not None and runner.name == "ollama":
        kw["chat_cls"] = ollama_cls
    if kw.get("callbacks") is None:
        kw["callbacks"] = _telemetry_callbacks(phase) or None
    try:
        actual = runner.map_model(model or settings.llm_model, vision=bool(kw.get("vision")))
        for cb in kw.get("callbacks") or []:
            if hasattr(cb, "model_name") and type(cb).__name__ == "StdoutTelemetryCallback":
                if not getattr(cb, "requested_model", ""):
                    cb.requested_model = model or settings.llm_model
                cb.model_name = actual
    except Exception:  # noqa: BLE001  - labelling only
        pass
    return runner.chat(model, **kw)


def resolve_model(model: str = "", *, vision: bool = False) -> str:
    """The model id the request's runner will actually use for `model` ('' = settings.llm_model)."""
    return get_runner().map_model(model or settings.llm_model, vision=vision)


def check_runner_config() -> None:
    """Fail fast (RunnerConfigError) when the CONFIGURED runners (settings.llm_runner, settings.embedding_runner) are unusable.
    Called at startup; no network. Other registered runners (e.g. aistudio without a key) are only checked when picked in the UI."""
    get_runner(settings.llm_runner).check_config()
    get_embedding_runner().check_config()


__all__ = ["MODEL_NAME", "Runner", "OpenAICompatRunner", "RunnerConfigError", "available_runners", "check_runner_config", "current_runner_name",
           "current_session_key", "get_embedding_runner", "get_runner", "make_chat", "register_runner", "require_runner",
           "resolve_model", "runner_scope", "runner_status", "session_key", "thread_scope"]
