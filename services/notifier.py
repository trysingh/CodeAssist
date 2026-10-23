import asyncio
import contextlib
import contextvars
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional, Set

from plyer import notification

from config.settings import settings
from models.schemas import GENERAL_REPO_ID, TelemetryEvent

# Repo the current task works for. Set by repo_scope(); lets code that has no repo_id in hand
# (sandbox, code-change agents, web search) still tag its events with the right repository.
_current_repo: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("notifier_repo", default=None)


@contextlib.contextmanager
def repo_scope(repo_id: Optional[str]):
    """Events raised inside this block (same task / threads started from it) belong to `repo_id`."""
    token = _current_repo.set(repo_id)
    try:
        yield
    finally:
        try:
            _current_repo.reset(token)
        except ValueError:      # reset from a different context (task hand-off): the value dies with that context
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _scope_of(repo_id: Optional[str]):
    """('general', None) for no repo / the General placeholder, else ('repo', repo_id)."""
    if not repo_id or repo_id == GENERAL_REPO_ID:
        return "general", None
    return "repo", repo_id


@dataclass(eq=False)
class _Subscriber:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    repo_id: Optional[str] = None           # selected repository of that browser tab; None = none / General

    def wants(self, ev: TelemetryEvent) -> bool:
        """General events go to everyone; repo events only to a subscriber that selected that repo."""
        return ev.scope == "general" or (ev.scope == "repo" and ev.repo_id == self.repo_id)


class NotificationDispatcher:
    """Dispatches notifications to Logs, Windows OS, Web Toasts / Telemetry tab (SSE), and SQLite.

    * Every event is classified general | repo and stored (batched, off the caller's thread).
    * subscribe(repo_id, limit) first replays the last `limit` stored events visible to that repo
      (general + that repo), then streams live ones with the same visibility rule.
    * Thread-safe: notify() may be called from worker threads (watchdog, sandbox, LangChain callbacks).
    """

    def __init__(self):
        self.subscribers: List[_Subscriber] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._db = None
        self._pending: Optional[asyncio.Queue] = None
        self._writer: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ lifecycle
    def bind(self, db=None) -> None:
        """Call once from the app's startup hook (inside the running loop, after db.init_db())."""
        self._loop = asyncio.get_running_loop()
        if db is not None and settings.telemetry_persist and self._writer is None:
            self._db = db
            self._pending = asyncio.Queue()
            self._writer = self._loop.create_task(self._write_loop())

    async def close(self) -> None:
        """Flush what is queued, then stop the writer (shutdown hook)."""
        writer, self._writer = self._writer, None
        if writer is not None and self._pending is not None:
            self._pending.put_nowait(None)
            try:
                await asyncio.wait_for(writer, timeout=5)
            except Exception:
                writer.cancel()
        self._pending, self._db, self._loop = None, None, None

    # ------------------------------------------------------------------ publish
    @staticmethod
    def _resolve_repo(title: str, message: str, repo_id: Optional[str]) -> Optional[str]:
        if repo_id:
            return repo_id
        if title == "Telemetry Trace":          # LLM traces carry their repo in the JSON payload
            try:
                rid = json.loads(message).get("repo_id")
                if rid:
                    return str(rid)
            except Exception:
                pass
        return _current_repo.get()

    def notify(self, title: str, message: str, level: str = "INFO", repo_id: Optional[str] = None):
        # Handle cases where message might be passed as a dictionary (e.g. Telemetry payload)
        if isinstance(message, dict):
            message = json.dumps(message)

        # 1. Logs / Stdout
        timestamp_log = f"\n[NOTIFICATION [{level}]] {title} — {message}\n"
        sys.stdout.write(timestamp_log)
        sys.stdout.flush()

        # 2. Windows OS Level Desktop Notification
        try:
            if settings.notify_at_os_level:
                notification.notify(
                    title=f"CodeRAG Engine: {title}",
                    message=message,
                    app_name="CodeRAG Engine",
                    timeout=5
                )
        except Exception:
            pass  # Fallback gracefully if running on headless server / container

        # 3. Persist + Web App SSE broadcast (never lets a failure here break the caller)
        try:
            scope, rid = _scope_of(self._resolve_repo(title, str(message), repo_id))
            ev = TelemetryEvent(uid=uuid.uuid4().hex, ts=_now_iso(), scope=scope, repo_id=rid,
                                level=str(level), title=title, message=str(message))
            self._publish(ev)
        except Exception as exc:
            sys.stderr.write(f"[NOTIFIER] dropped event '{title}': {type(exc).__name__}: {exc}\n")

    def _publish(self, ev: TelemetryEvent) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                self._deliver(ev)
            else:                                   # worker thread / another loop: hand over to the app loop
                loop.call_soon_threadsafe(self._deliver, ev)
        else:
            self._deliver(ev)                       # not bound (tests, CLI): broadcast only

    def _deliver(self, ev: TelemetryEvent) -> None:
        for sub in list(self.subscribers):
            if sub.wants(ev):
                try:
                    sub.queue.put_nowait(ev)
                except Exception:
                    pass
        if self._pending is not None:
            self._pending.put_nowait(ev)

    async def _write_loop(self) -> None:
        inserted = 0
        while True:
            first = await self._pending.get()
            if first is None:
                return
            batch, stop = [first], False
            while len(batch) < 200:
                try:
                    nxt = self._pending.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
            try:
                await self._db.save_telemetry_events([e.model_dump() for e in batch])
                inserted += len(batch)
                if inserted >= max(1, settings.telemetry_prune_every):
                    inserted = 0
                    await self._db.prune_telemetry_events(settings.telemetry_max_rows)
            except Exception as exc:
                sys.stderr.write(f"[NOTIFIER] could not persist {len(batch)} event(s): {type(exc).__name__}: {exc}\n")
            if stop:
                return

    # ------------------------------------------------------------------ subscribe
    @staticmethod
    def clamp_limit(limit: Optional[int]) -> int:
        n = settings.telemetry_history_default if limit is None else int(limit)
        return max(0, min(n, settings.telemetry_history_max))

    async def subscribe(self, repo_id: Optional[str] = None, limit: Optional[int] = None,
                        history: bool = True) -> AsyncGenerator[dict, None]:
        """SSE feed for one browser tab.

        Event 'history'      : {repo_id, limit, events:[...]} once, oldest first (replaces the console).
        Event 'notification' : one live event (same fields as a history row).
        """
        _, scoped_repo = _scope_of(repo_id)
        sub = _Subscriber(repo_id=scoped_repo)
        self.subscribers.append(sub)            # registered BEFORE the history read: nothing published meanwhile is lost
        try:
            seen: Set[str] = set()
            if history:
                n = self.clamp_limit(limit)
                rows: list = []
                if self._db is not None and settings.telemetry_persist:
                    try:
                        rows = await self._db.get_telemetry_events(scoped_repo, n)
                    except Exception as exc:
                        sys.stderr.write(f"[NOTIFIER] history read failed: {type(exc).__name__}: {exc}\n")
                seen = {r["uid"] for r in rows}
                yield {"event": "history",
                       "data": json.dumps({"repo_id": scoped_repo, "limit": n, "events": rows})}
            while True:
                ev = await sub.queue.get()
                if ev.uid in seen:              # already inside the history we just sent
                    seen.discard(ev.uid)
                    continue
                yield {"event": "notification", "data": ev.model_dump_json()}
        finally:
            if sub in self.subscribers:
                self.subscribers.remove(sub)


notifier = NotificationDispatcher()
