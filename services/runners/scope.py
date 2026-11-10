"""Per-request runner state (ContextVars, isolated across concurrent asyncio tasks). A leaf module: no runner imports.

runner_scope(name)        the chat runner the user picked in the UI for this request; None / "" = settings.llm_runner.
                          Only CHAT goes through it. Embeddings always use settings.embedding_runner (see get_embedding_runner):
                          vectors of different embedding models are not comparable, so a per-request switch would break the index.
thread_scope(repo, thread) the chat thread of this request; the Cloudflare runner derives its `x-session-affinity` header from it.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from config.settings import settings

_runner: ContextVar[str] = ContextVar("runner_override", default="")
_session: ContextVar[str] = ContextVar("session_key", default="")


@contextmanager
def runner_scope(name: Optional[str]):
    """Use runner `name` for chat calls made inside the block. Empty = keep the current one. Validate with require_runner first."""
    name = (name or "").strip().lower()
    if not name:
        yield
        return
    token = _runner.set(name)
    try:
        yield
    finally:
        _runner.reset(token)


def current_runner_name() -> str:
    """The chat runner of this request: runner_scope, else settings.llm_runner."""
    return (_runner.get() or settings.llm_runner or "ollama").strip().lower() or "ollama"


@contextmanager
def thread_scope(repo_id: str, thread_id: str):
    """Stable cache-affinity id for one repo + chat thread. Empty ids = no id."""
    key = session_key(repo_id, thread_id)
    token = _session.set(key)
    try:
        yield
    finally:
        _session.reset(token)


def session_key(repo_id: str, thread_id: str) -> str:
    if not (repo_id or "").strip() or not (thread_id or "").strip():
        return ""
    return "ca-" + hashlib.sha256(f"{repo_id}:{thread_id}".encode("utf-8")).hexdigest()[:32]


def current_session_key() -> str:
    return _session.get()
