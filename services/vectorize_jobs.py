"""Background vectorize jobs: queue, dedup, cancel, throttled progress notifications, persisted state.

Dependency-light by design: the actual embedding work, the notifier and the DB status setter are injected.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set

try:  # soft import (Agent A's module)
    from services.action_log import log_action as _log_action
except Exception:  # noqa: BLE001
    def _log_action(*_a, **_k):  # type: ignore[misc]
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_STATES = (JobState.QUEUED, JobState.RUNNING)


@dataclass
class Job:
    job_id: str
    repo_id: str
    kind: str = "vectorize"
    state: JobState = JobState.QUEUED
    reason: str = ""
    progress: int = 0
    total: int = 0
    error: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Job":
        return cls(job_id=d["job_id"], repo_id=d["repo_id"], kind=d.get("kind") or "vectorize",
                   state=JobState(d.get("state") or "QUEUED"), reason=d.get("reason") or "",
                   progress=int(d.get("progress") or 0), total=int(d.get("total") or 0), error=d.get("error"),
                   created_at=d.get("created_at") or _now_iso(), updated_at=d.get("updated_at") or _now_iso())

    @property
    def percent(self) -> int:
        return int(100 * self.progress / self.total) if self.total > 0 else 0


class JobStore(Protocol):
    async def upsert(self, job: Job) -> None: ...
    async def get(self, job_id: str) -> Optional[Job]: ...
    async def list(self, repo_id: Optional[str] = None, limit: int = 50) -> List[Job]: ...
    async def latest(self, repo_id: str, kind: str = "vectorize") -> Optional[Job]: ...


class InMemoryJobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []

    async def upsert(self, job: Job) -> None:
        if job.job_id not in self._jobs:
            self._order.append(job.job_id)
        self._jobs[job.job_id] = Job.from_dict(job.to_dict())       # store a copy, like a DB would

    async def get(self, job_id: str) -> Optional[Job]:
        j = self._jobs.get(job_id)
        return Job.from_dict(j.to_dict()) if j else None

    async def list(self, repo_id: Optional[str] = None, limit: int = 50) -> List[Job]:
        rows = [self._jobs[i] for i in reversed(self._order) if repo_id is None or self._jobs[i].repo_id == repo_id]
        return [Job.from_dict(j.to_dict()) for j in rows[: max(0, limit)]]

    async def latest(self, repo_id: str, kind: str = "vectorize") -> Optional[Job]:
        for i in reversed(self._order):
            j = self._jobs[i]
            if j.repo_id == repo_id and j.kind == kind:
                return Job.from_dict(j.to_dict())
        return None


class DbJobStore:
    """JobStore over DatabaseManager (models/database.py): upsert_job / get_job / list_jobs / latest_job."""

    def __init__(self, db):
        self.db = db

    async def upsert(self, job: Job) -> None:
        await self.db.upsert_job(job.to_dict())

    async def get(self, job_id: str) -> Optional[Job]:
        d = await self.db.get_job(job_id)
        return Job.from_dict(d) if d else None

    async def list(self, repo_id: Optional[str] = None, limit: int = 50) -> List[Job]:
        return [Job.from_dict(d) for d in await self.db.list_jobs(repo_id, limit)]

    async def latest(self, repo_id: str, kind: str = "vectorize") -> Optional[Job]:
        d = await self.db.latest_job(repo_id, kind)
        return Job.from_dict(d) if d else None


_STATUS_BY_STATE = {
    JobState.QUEUED: "queued", JobState.RUNNING: "running", JobState.COMPLETED: "ready",
    JobState.FAILED: "failed", JobState.CANCELLED: "none",
}


class VectorizeJobManager:
    def __init__(self, store: JobStore,
                 work: Callable[[str, Callable[[int, int], None]], Awaitable[None]],
                 notify: Callable[..., Any],
                 set_vector_status: Optional[Callable[[str, str], Awaitable[None]]] = None,
                 *, clock: Callable[[], float] = time.monotonic, progress_step: int = 10,
                 progress_interval: float = 15.0, max_concurrent: int = 1):
        self.store = store
        self.work = work
        self._notify = notify
        self.set_vector_status = set_vector_status
        self.clock = clock
        self.progress_step = progress_step
        self.progress_interval = progress_interval
        self.max_concurrent = max(1, max_concurrent)
        self._sem: Optional[asyncio.Semaphore] = None
        self._active: Dict[str, Job] = {}               # repo_id -> live job (QUEUED/RUNNING)
        self._tasks: Dict[str, asyncio.Task] = {}       # job_id -> task
        self._bg: Set[asyncio.Task] = set()
        self._status: Dict[str, str] = {}               # repo_id -> status cache
        self._done: Dict[str, asyncio.Event] = {}
        self._finished: Dict[str, Job] = {}

    # ---- public API
    async def enqueue(self, repo_id: str, reason: str) -> Job:
        existing = self._active.get(repo_id)
        if existing and existing.state in ACTIVE_STATES:
            return existing
        job = Job(job_id=uuid.uuid4().hex[:12], repo_id=repo_id, state=JobState.QUEUED, reason=reason)
        self._active[repo_id] = job
        self._done[job.job_id] = asyncio.Event()
        await self._persist(job)
        await self._set_status(repo_id, "queued")
        self._send("Vectorize", f"Vectorizing {repo_id} in background: {reason}. This may take a while; you'll be notified.",
                   "INFO", job)
        self._log("enqueue", "info", f"queued vectorize job {job.job_id}: {reason}", job)
        self._tasks[job.job_id] = asyncio.get_running_loop().create_task(self._run(job))
        return job

    async def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task})
            job = self.get_live(job_id)
            if job is not None and job.state in ACTIVE_STATES:      # cancelled before its first step: _run never ran
                await self._on_cancelled(job)
                self._tasks.pop(job_id, None)
                ev = self._done.get(job_id)
                if ev is not None:
                    ev.set()
            return True
        stored = await self.store.get(job_id)          # a row left by an earlier process: nothing runs, just close it
        if stored and stored.state in ACTIVE_STATES and job_id not in self._tasks:
            stored.state = JobState.CANCELLED
            stored.updated_at = _now_iso()
            await self.store.upsert(stored)
            await self._set_status(stored.repo_id, "none")
            return True
        return False

    def status(self, repo_id: str) -> str:
        """none|queued|running|ready|failed (cached; seed with set_cached_status after a restart)."""
        return self._status.get(repo_id, "none")

    def set_cached_status(self, repo_id: str, status: str) -> None:
        self._status[repo_id] = status

    def get_live(self, job_id: str) -> Optional[Job]:
        for j in self._active.values():
            if j.job_id == job_id:
                return j
        return self._finished.get(job_id)

    async def get(self, job_id: str) -> Optional[Job]:
        return self.get_live(job_id) or await self.store.get(job_id)

    async def wait(self, job_id: str, timeout: Optional[float] = None) -> Optional[Job]:
        ev = self._done.get(job_id)
        if ev is not None:
            await asyncio.wait_for(ev.wait(), timeout)
        return await self.get(job_id)

    # ---- internals
    async def _run(self, job: Job) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        t0 = time.perf_counter()
        try:
            async with self._sem:
                job.state = JobState.RUNNING
                await self._persist(job)
                await self._set_status(job.repo_id, "running")
                self._log("run", "start", "vectorizing", job)
                state = {"pct": 0, "t": self.clock(), "saved_pct": 0, "saved_t": self.clock()}
                loop = asyncio.get_running_loop()

                def on_progress(done: int, total: int) -> None:
                    # may be called from the event loop or from a worker thread (embedding runs in to_thread)
                    try:
                        in_loop = asyncio.get_running_loop() is loop
                    except RuntimeError:
                        in_loop = False
                    if in_loop:
                        self._progress(job, int(done), int(total), state)
                    else:
                        loop.call_soon_threadsafe(self._progress, job, int(done), int(total), state)

                await self.work(job.repo_id, on_progress)
            job.state = JobState.COMPLETED
            job.progress = job.total if job.total else job.progress
            job.error = None
            self._finish(job)
            await self._persist(job)
            await self._set_status(job.repo_id, "ready")
            self._send("Vector index ready", f"Vector index ready — semantic search enabled for {job.repo_id}.", "SUCCESS", job)
            self._log("run", "ok", "vector index ready", job, ms=int((time.perf_counter() - t0) * 1000))
        except asyncio.CancelledError:
            await self._on_cancelled(job)
        except Exception as e:  # noqa: BLE001 - a failing job must never crash the loop
            job.state = JobState.FAILED
            job.error = f"{type(e).__name__}: {e}"
            self._finish(job)
            try:
                await self._persist(job)
                await self._set_status(job.repo_id, "failed")
            except Exception:  # noqa: BLE001
                pass
            self._send("Vectorize failed", f"Vectorizing {job.repo_id} failed: {job.error}", "ERROR", job)
            self._log("run", "fail", job.error, job, ms=int((time.perf_counter() - t0) * 1000))
        finally:
            ev = self._done.get(job.job_id)
            if ev is not None:
                ev.set()
            self._tasks.pop(job.job_id, None)

    async def _on_cancelled(self, job: Job) -> None:
        job.state = JobState.CANCELLED
        self._finish(job)
        await self._shielded(self._persist(job), self._set_status(job.repo_id, "none"))
        self._send("Vectorize cancelled", f"Vectorizing {job.repo_id} was cancelled.", "WARNING", job)
        self._log("run", "skip", "cancelled", job)

    def _finish(self, job: Job) -> None:
        job.updated_at = _now_iso()
        if self._active.get(job.repo_id) is job:
            del self._active[job.repo_id]
        self._finished[job.job_id] = job
        if len(self._finished) > 100:
            self._finished.pop(next(iter(self._finished)))

    async def _shielded(self, *aws) -> None:
        for aw in aws:
            try:
                await asyncio.shield(aw)
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass

    def _progress(self, job: Job, done: int, total: int, st: dict) -> None:
        if job.state != JobState.RUNNING:
            return
        job.progress, job.total = max(0, done), max(0, total)
        job.updated_at = _now_iso()
        pct, now = job.percent, self.clock()
        if pct - st["saved_pct"] >= 2 or now - st["saved_t"] >= 2.0:
            st["saved_pct"], st["saved_t"] = pct, now
            self._spawn(self._persist(job))
        if job.total > 0 and job.progress >= job.total:
            return                                              # the "ready" notification covers 100%
        if pct - st["pct"] >= self.progress_step or now - st["t"] >= self.progress_interval:
            st["pct"], st["t"] = pct, now
            self._send("Vectorize progress", f"Vectorizing {job.repo_id}: {job.progress}/{job.total} ({pct}%)", "INFO", job)
            self._log("progress", "info", f"{job.progress}/{job.total} ({pct}%)", job)

    def _spawn(self, coro) -> None:
        async def guard():
            try:
                await coro
            except Exception:  # noqa: BLE001
                pass
        t = asyncio.get_running_loop().create_task(guard())
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _persist(self, job: Job) -> None:
        job.updated_at = _now_iso()
        await self.store.upsert(job)

    async def _set_status(self, repo_id: str, status: str) -> None:
        self._status[repo_id] = status
        if self.set_vector_status is not None:
            try:
                await self.set_vector_status(repo_id, status)
            except Exception:  # noqa: BLE001 - the cache stays right even if the DB write fails
                pass

    def _send(self, title: str, message: str, level: str, job: Job) -> None:
        try:
            self._notify(title, message, level=level, repo_id=job.repo_id, kind="job", data=job.to_dict())
        except TypeError:
            try:
                self._notify(title, message, level, job.repo_id)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    def _log(self, action: str, status: str, detail: str, job: Job, ms: Optional[int] = None) -> None:
        try:
            _log_action("VECTORIZE", "Vectorizer", action, status=status, detail=detail or "",
                        data={"job_id": job.job_id, "repo_id": job.repo_id, "progress": job.progress, "total": job.total},
                        duration_ms=ms)
        except Exception:  # noqa: BLE001
            pass
