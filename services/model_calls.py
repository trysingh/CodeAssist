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
    if not re.fullmatch(r"[\w.:/\-]{1,100}", model):
        yield
        return
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)

def reasoning_model(default: str = "") -> str:
    return _model_override.get() or default or settings.llm_model

async def list_chat_models() -> list:
    """Models on the Ollama endpoint (embedding models hidden). Never raises."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(chat_base_url() + "/api/tags")
            r.raise_for_status()
            names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        names = []
    return sorted({n for n in names if n and "embed" not in n.lower()})

def _callbacks(model: str, repo_id: str, phase: str) -> list:
    try:
        from services.notifier import notifier
        from services.telemetry import StdoutTelemetryCallback
        return [StdoutTelemetryCallback(
            model_name=model, repo_id=repo_id, phase=phase,
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
 
 
def _build(model: str, *, json_mode: bool, num_predict: Optional[int], num_ctx: int,
           repo_id: str, phase: str):
    from langchain_ollama import ChatOllama
    kw = dict(model=model, base_url=chat_base_url(), temperature=0,
              num_ctx=num_ctx, callbacks=_callbacks(model, repo_id, phase))
    if num_predict:
        kw["num_predict"] = num_predict     # never -1: cloud endpoints reject non-positive values
    if json_mode:
        kw["format"] = "json"
    return ChatOllama(**kw)


async def ask_text(model: str, prompt: str, *, phase: str = "CHANGE", repo_id: str = "",
                   num_predict: Optional[int] = None) -> str:
    llm = _build(model, json_mode=False, num_predict=num_predict or settings.change_max_output_tokens,
                 num_ctx=settings.change_num_ctx, repo_id=repo_id, phase=phase)
    resp = await llm.ainvoke(prompt)
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
        llm = _build(model, json_mode=True, num_predict=2048, num_ctx=16384,
                     repo_id=repo_id, phase=phase)
        resp = await llm.ainvoke(prompt)
        return parse_json_loose(resp.content if isinstance(resp.content, str) else str(resp.content))
    except Exception:
        return None
