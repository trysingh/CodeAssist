"""Chat-model helpers shared by the change agents (text and JSON out).

Decision models (tev1 ...) do NOT go through here: they use /v1/systemone, see
services/decision.py. Callers receive these functions by injection, so tests
and future providers swap them without touching agent code.
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable, Optional

from config.settings import settings
from services.llm_errors import with_model_fallback
from services.runners import MODEL_NAME, get_runner, runner_scope           # noqa: F401  (runner_scope: re-exported next to model_scope)
from services.token_budget import effective_mode, token_mode_scope           # noqa: F401  (same: the per-request token-mode pick)
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

# (model, prompt, *, phase, repo_id, num_predict) -> text
LLMText = Callable[..., Awaitable[str]]
# (model, prompt, *, phase, repo_id) -> parsed dict or None
LLMJson = Callable[..., Awaitable[Optional[dict]]]


_model_override: ContextVar[str] = ContextVar("model_override", default="")

@contextmanager
def model_scope(model: str):
    """User's reasoning-model pick for this request. Empty/invalid = keep the current (default)."""
    model = (model or "").strip()
    if not MODEL_NAME.fullmatch(model):
        yield
        return
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)

def role_model(name: str) -> str:
    """A per-role model setting (PATCHER_MODEL, CHANGE_ESCALATION_MODEL, DIAGNOSER_MODEL ...) as the request's runner understands it.
    On a non-Ollama runner an Ollama-style tag (qwen3-coder:30b, gemma4:31b-cloud) is not a model of that backend: '' = "use this runner's
    default" (the caller then falls back to the patcher / global default, which resolves to the runner's configured model)."""
    n = (name or "").strip()
    if not n:
        return ""
    try:
        foreign = getattr(get_runner(), "is_foreign_name", None)
        if foreign is not None and foreign(n):
            return ""
    except Exception:  # noqa: BLE001 - an unusable runner config is reported by the call itself
        pass
    return n


def reasoning_model(default: str = "") -> str:
    """The model NAME a call asks for: the user's pick, else the role setting `default` (Ollama-style names are ignored on other
    runners, see role_model), else settings.llm_model. The runner maps it to its real id (resolve_model); '' / settings.llm_model
    mean the runner's configured default."""
    return _model_override.get() or role_model(default) or settings.llm_model


_announced: deque = deque(maxlen=256)       # request ids whose "model resolved" line was already written


def announce_model() -> None:
    """Request start: tell the user which model this request really calls ("model resolved: runner=aistudio picked=Default ->
    gemini-2.5-flash"), once per request. An Activity event and a telemetry-list line. Never raises."""
    try:
        from services import action_log as al
        rid = al.current_request_id() or ""
        if rid and rid in _announced:
            return
        if rid:
            _announced.append(rid)
        runner = get_runner()
        pick = _model_override.get()
        actual = runner.map_model(pick or settings.llm_model)
        detail = f"model resolved: runner={runner.name} picked={pick or 'Default'} -> {actual}"
        t = getattr(runner, "timeout_s", None)
        t = t() if callable(t) else None
        al.log_action("LLM", "ModelResolve", "model_resolved", "info", detail, model=actual,
                      data={"runner": runner.name, "picked": pick or "Default", "model": actual, "timeout_s": t})
        from services.notifier import notifier
        notifier.notify("Model resolved", detail, level="INFO")
    except Exception:  # noqa: BLE001
        pass

async def list_chat_models(runner: Optional[str] = None) -> list:
    """Chat models of `runner` (None = the request's runner_scope, else settings.llm_runner; Ollama: GET /api/tags, embedding
    models hidden). Never raises."""
    try:
        return await get_runner(runner).list_models()
    except Exception:  # noqa: BLE001
        return []

def _callbacks(model: str, repo_id: str, phase: str, requested: str = "") -> list:
    """Telemetry callback for one call. `model` = the id the runner really calls, `requested` = the name the caller asked for."""
    try:
        from services.notifier import notifier
        from services.telemetry import StdoutTelemetryCallback
        return [StdoutTelemetryCallback(
            model_name=model, repo_id=repo_id, phase=phase, requested_model=requested,
            emitter=lambda data: notifier.notify("Telemetry Trace", data, level="INFO"))]
    except Exception:
        return []



def chat_base_url() -> str:
    """Base URL for CHAT models (patcher, locator): plain Ollama, e.g. http://localhost:11434.
 
    Never settings.ollama_base_url_for_sys1_model: that one is the decision-model endpoint
    (.../v1/systemone). ChatOllama appends /api/chat to whatever it is given, so handing it the
    systemone URL requests /v1/systemone/api/chat and Ollama answers 404. A stray /v1 or
    /v1/systemone suffix on ollama_base_url is stripped for the same reason."""
    base = (settings.ollama_base_url or "").strip().rstrip("/")
    for suffix in ("/v1/systemone", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")
 
 
def reasoning_arg(value: str = ""):
    """settings.ask_reasoning -> the `reasoning` argument of ChatOllama.

    "" -> None (not sent: the server/model decides), off/false/no/0 -> False, on/true/yes/1 -> True,
    low/medium/high -> that level (models that take an effort level). Anything else -> None."""
    v = (value or "").strip().lower()
    if v in ("off", "false", "no", "0"):
        return False
    if v in ("on", "true", "yes", "1"):
        return True
    if v in ("low", "medium", "high"):
        return v
    return None


def _build(model: str, *, json_mode: bool, num_predict: Optional[int], num_ctx: int,
           repo_id: str, phase: str, reasoning=None):
    runner = get_runner()
    actual = runner.map_model(model)        # an empty / Ollama-style name becomes the runner's configured default (Ollama: unchanged)
    return runner.chat(actual, temperature=0, max_tokens=num_predict, json_mode=json_mode, reasoning=reasoning,
                       num_ctx=num_ctx, callbacks=_callbacks(actual, repo_id, phase, requested=model))


async def ask_text(model: str, prompt: str, *, phase: str = "CHANGE", repo_id: str = "",
                   num_predict: Optional[int] = None, num_ctx: Optional[int] = None) -> str:
    async def _call(m: str):
        llm = _build(m, json_mode=False, num_predict=num_predict or settings.change_max_output_tokens,
                     num_ctx=num_ctx or settings.change_num_ctx, repo_id=repo_id, phase=phase)
        return await llm.ainvoke(prompt)
    resp = await with_model_fallback(_call, model, phase=phase)     # 402 (cloud credits used up) -> settings.llm_fallback_model, or ModelQuotaError
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def parse_json_loose(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```[\w-]*\s*|\s*```$", "", raw)
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                val = json.loads(m.group(0))
                return val if isinstance(val, dict) else None
            except Exception:
                return None
    return None


async def ask_json(model: str, prompt: str, *, phase: str = "DECIDE", repo_id: str = "") -> Optional[dict]:
    """Chat-model call that must answer in JSON. Any failure returns None so callers fall back."""
    try:
        async def _call(m: str):
            llm = _build(m, json_mode=True, num_predict=2048, num_ctx=16384,
                         repo_id=repo_id, phase=phase)
            return await llm.ainvoke(prompt)
        resp = await with_model_fallback(_call, model, phase=phase)
        return parse_json_loose(resp.content if isinstance(resp.content, str) else str(resp.content))
    except Exception as exc:
        try:    # the success path is logged by the telemetry callback; a failed call never reaches it
            from services.action_log import log_action
            log_action("LLM", phase, "llm_call", "fail", f"{type(exc).__name__}: {exc}"[:200], model=model,
                       repo_id=repo_id or None)
        except Exception:
            pass
        return None
