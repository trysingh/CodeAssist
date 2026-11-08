"""Structured action log (stdlib only). See docs/DESIGN_v2.md section 2.

Every agent step / tool call / LLM call / sandbox run is an ActionEvent. Events go to
  * an in-memory ring buffer (cap 5000), and
  * any registered sinks (e.g. notifier -> SSE/DB). Sinks never raise into the caller.
request_id / thread_id travel in ContextVars (request_scope), so they survive asyncio tasks
and (via contextvars.copy_context / asyncio.to_thread) worker threads.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

PHASES = ("INTAKE", "DECOMPOSE", "BASELINE", "CONTRACT", "IMPLEMENT", "HARDEN", "TEST", "SANDBOX", "FIX",
          "CROSSCHECK", "DELIVER", "RETRIEVE", "INGEST", "VECTORIZE", "LLM")

MAX_DATA_BYTES = 4096
RING_CAPACITY = 5000

_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("action_request_id", default=None)
_thread_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("action_thread_id", default=None)
_repo_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("action_repo_id", default=None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------- context
def current_request_id() -> Optional[str]:
    return _request_id.get()


def current_thread_id() -> Optional[str]:
    return _thread_id.get()


def current_repo_id() -> Optional[str]:
    """Repo from request_scope(repo_id=...) else from notifier.repo_scope (soft lookup)."""
    rid = _repo_id.get()
    if rid:
        return rid
    mod = sys.modules.get("services.notifier")
    if mod is not None:
        try:
            return getattr(mod, "_current_repo").get()
        except Exception:
            return None
    return None


@contextlib.contextmanager
def request_scope(request_id: Optional[str] = None, thread_id: Optional[str] = None, repo_id: Optional[str] = None):
    """Events logged inside carry this request/thread id. request_id None -> a fresh one is generated."""
    rid = request_id or uuid.uuid4().hex[:12]
    t1 = _request_id.set(rid)
    t2 = _thread_id.set(thread_id)
    t3 = _repo_id.set(repo_id) if repo_id is not None else None
    try:
        yield rid
    finally:
        for var, tok in ((_request_id, t1), (_thread_id, t2), (_repo_id, t3)):
            if tok is None:
                continue
            try:
                var.reset(tok)
            except ValueError:      # reset in another context (task hand-off)
                pass


# ---------------------------------------------------------------------- event
def _truncate_data(data: Any, limit: int = MAX_DATA_BYTES) -> Dict[str, Any]:
    """JSON-safe copy of `data`, <= ~limit bytes serialized. Oversize payloads shrink value by value."""
    if not data:
        return {}
    try:
        safe = json.loads(json.dumps(data, default=repr))
    except Exception:
        safe = {"repr": repr(data)}
    if not isinstance(safe, dict):
        safe = {"value": safe}

    def size(o) -> int:
        return len(json.dumps(o, ensure_ascii=False).encode("utf-8"))

    if size(safe) <= limit:
        return safe
    # shrink: cap long strings / lists progressively
    for cap in (1000, 400, 160, 60, 20):
        def shrink(o):
            if isinstance(o, str):
                return o if len(o) <= cap else o[:cap] + "..."
            if isinstance(o, list):
                items = [shrink(x) for x in o[: max(3, cap // 20)]]
                return items + ([f"...(+{len(o) - len(items)} more)"] if len(o) > len(items) else [])
            if isinstance(o, dict):
                return {str(k): shrink(v) for k, v in o.items()}
            return o
        out = shrink(safe)
        if size(out) <= limit:
            out["_truncated"] = True
            return out
    # last resort: keep keys only
    out = {"_truncated": True, "keys": [str(k) for k in list(safe)[:40]]}
    return out


@dataclass
class ActionEvent:
    uid: str
    ts: str
    request_id: Optional[str]
    thread_id: Optional[str]
    repo_id: Optional[str]
    phase: str
    agent: str
    action: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None
    model: Optional[str] = None
    parent_uid: Optional[str] = None

    def to_dict(self) -> dict:
        return {"uid": self.uid, "ts": self.ts, "request_id": self.request_id, "thread_id": self.thread_id,
                "repo_id": self.repo_id, "phase": self.phase, "agent": self.agent, "action": self.action,
                "status": self.status, "detail": self.detail, "data": dict(self.data),
                "duration_ms": self.duration_ms, "model": self.model, "parent_uid": self.parent_uid}


# ---------------------------------------------------------------------- ring buffer
class RingBuffer:
    def __init__(self, capacity: int = RING_CAPACITY):
        self._dq: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, ev: ActionEvent) -> None:
        with self._lock:
            self._dq.append(ev)

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()

    def __len__(self) -> int:
        return len(self._dq)

    def recent(self, request_id: Optional[str] = None, repo_id: Optional[str] = None, limit: int = 500) -> List[dict]:
        """Newest `limit` matching events, returned oldest first."""
        with self._lock:
            items = list(self._dq)
        if request_id is not None:
            items = [e for e in items if e.request_id == request_id]
        if repo_id is not None:
            items = [e for e in items if e.repo_id == repo_id]
        if limit is not None and limit >= 0:
            items = items[-limit:] if limit else []
        return [e.to_dict() for e in items]


ring = RingBuffer()
_sinks: List[Callable[[ActionEvent], None]] = []
_sinks_lock = threading.Lock()


def set_sink(fn: Optional[Callable[[ActionEvent], None]]) -> None:
    """Replace all sinks with `fn` (None = no sinks)."""
    with _sinks_lock:
        _sinks[:] = [fn] if fn else []


def add_sink(fn: Callable[[ActionEvent], None]) -> None:
    with _sinks_lock:
        if fn not in _sinks:
            _sinks.append(fn)


def clear_sinks() -> None:
    with _sinks_lock:
        _sinks.clear()


def make_notifier_sink(notify: Callable[..., Any]) -> Callable[["ActionEvent"], None]:
    """Sink that forwards every ActionEvent to notifier.notify (SSE + DB): kind="action", data=event dict.
    A failed step is shown as ERROR, everything else as INFO. `notify` is injected so this stays testable."""
    def sink(ev: "ActionEvent") -> None:
        notify("Action", ev.detail or f"{ev.agent}: {ev.action}",
               level="ERROR" if ev.status == "fail" else "INFO",
               repo_id=ev.repo_id, kind="action", request_id=ev.request_id,
               thread_id=ev.thread_id, data=ev.to_dict())
    return sink


def events_from_telemetry_rows(rows: List[dict], repo_id: Optional[str] = None) -> List[dict]:
    """ActionEvent dicts out of persisted telemetry rows (kind == "action" carries the event in `data`)."""
    out = []
    for r in rows or []:
        d = r.get("data") if isinstance(r, dict) else None
        if isinstance(r, dict) and r.get("kind") == "action" and isinstance(d, dict) and d.get("uid"):
            if repo_id is None or d.get("repo_id") == repo_id:
                out.append(d)
    return out


def format_trace(events: List[dict]) -> str:
    """Compact one-line-per-step text trace of ActionEvent dicts (CLI --trace). Start events are folded into their closing event."""
    icons = {"ok": "ok  ", "fail": "FAIL", "skip": "skip", "info": "info"}
    lines = []
    for e in events or []:
        if e.get("status") == "start":
            continue
        ms = f" {e['duration_ms']}ms" if e.get("duration_ms") is not None else ""
        model = f" [{e['model']}]" if e.get("model") else ""
        lines.append(f"  {icons.get(e.get('status'), str(e.get('status')))} {str(e.get('phase', '')):<9} {e.get('agent', '')}: "
                     f"{e.get('action', '')}{ms}{model}" + (f" - {e['detail']}" if e.get("detail") else ""))
    return "\n".join(lines)


def recent(request_id: Optional[str] = None, repo_id: Optional[str] = None, limit: int = 500) -> List[dict]:
    return ring.recent(request_id=request_id, repo_id=repo_id, limit=limit)


def log_action(phase: str, agent: str, action: str, status: str = "info", detail: str = "",
               data: Optional[dict] = None, model: Optional[str] = None, duration_ms: Optional[int] = None,
               parent_uid: Optional[str] = None, repo_id: Optional[str] = None) -> ActionEvent:
    """Record one event. Never raises."""
    try:
        ev = ActionEvent(uid=uuid.uuid4().hex, ts=_now_iso(), request_id=current_request_id(),
                         thread_id=current_thread_id(), repo_id=repo_id or current_repo_id(),
                         phase=str(phase), agent=str(agent), action=str(action), status=str(status),
                         detail=str(detail or ""), data=_truncate_data(data), duration_ms=duration_ms,
                         model=model, parent_uid=parent_uid)
    except Exception:       # pragma: no cover - defensive
        ev = ActionEvent(uid=uuid.uuid4().hex, ts=_now_iso(), request_id=None, thread_id=None, repo_id=None,
                         phase="?", agent="?", action="?", status="info")
    try:
        ring.append(ev)
    except Exception:       # pragma: no cover
        pass
    with _sinks_lock:
        sinks = list(_sinks)
    for s in sinks:
        try:
            s(ev)
        except BaseException as exc:    # a sink must never break the caller
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                sys.stderr.write(f"[ACTION_LOG] sink error: {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
    return ev


# ---------------------------------------------------------------------- action()
class ActionHandle:
    """Yielded by action(); .note() enriches the closing event."""

    def __init__(self, start: ActionEvent):
        self.start = start
        self.uid = start.uid
        self.detail: Optional[str] = None
        self.data: Dict[str, Any] = {}
        self.status: Optional[str] = None
        self.model: Optional[str] = None

    def note(self, detail: Optional[str] = None, *, status: Optional[str] = None, model: Optional[str] = None,
             **data: Any) -> "ActionHandle":
        if detail is not None:
            self.detail = detail
        if status is not None:
            self.status = status
        if model is not None:
            self.model = model
        self.data.update(data)
        return self


@contextlib.contextmanager
def action(phase: str, agent: str, action: str, detail: str = "", data: Optional[dict] = None,
           model: Optional[str] = None):
    """Emit start, then ok (with duration) or fail (exception repr, re-raised)."""
    start = log_action(phase, agent, action, "start", detail, data, model)
    h = ActionHandle(start)
    t0 = time.perf_counter()
    try:
        yield h
    except BaseException as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        merged = dict(data or {})
        merged.update(h.data)
        merged["error"] = f"{type(exc).__name__}: {exc}"[:500]
        log_action(phase, agent, action, "fail", h.detail if h.detail is not None else f"{type(exc).__name__}: {exc}"[:300],
                   merged, h.model or model, ms, start.uid)
        raise
    else:
        ms = int((time.perf_counter() - t0) * 1000)
        merged = dict(data or {})
        merged.update(h.data)
        log_action(phase, agent, action, h.status or "ok", h.detail if h.detail is not None else detail,
                   merged, h.model or model, ms, start.uid)


def traced(phase: str, agent: str, action_name: Optional[str] = None):
    """Decorator (sync and async) wrapping the call in action()."""
    def deco(fn):
        name = action_name or getattr(fn, "__name__", "call")
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                with action(phase, agent, name):
                    return await fn(*a, **kw)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with action(phase, agent, name):
                return fn(*a, **kw)
        return wrapper
    return deco
