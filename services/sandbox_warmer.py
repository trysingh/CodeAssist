"""Background pre-warming of sandbox dependency images.

Why: the first change request for a repo used to wait for `docker build` (pip install of the
repo's requirements: minutes). Now the image is built ahead of time, in the background:

    app startup          -> the N most recently used repos        (start())
    ingest finished      -> that repo                             (request_prewarm)
    requirements changed -> that repo, new image tag              (request_prewarm, from the file watcher)

Rules
  * never blocks: start() only schedules work; builds run in worker threads (docker is a subprocess)
  * one build at a time, so a first start with many repos does not saturate the machine or Ollama;
    requests for a repo that is already queued are merged
  * request() is safe from any thread (the file watcher runs in its own thread with its own event loop)
  * Docker not up yet (very common on Windows: the app starts before Docker Desktop): ask again every
    _RETRY_S seconds, at most _MAX_RETRIES times, then leave it to the first change request
  * a change request that arrives while its image is still building simply waits for that build
    (DockerRunner holds a lock per image tag), it does not start a second one
  * stop() kills a running `docker build`, so closing the application is never held up by a build
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

from config.settings import settings

_RETRY_S = 30.0
_MAX_RETRIES = 20

ListRepoIds = Callable[[], Awaitable[List[str]]]     # most recently used first


def is_requirements_path(rel_path: str) -> bool:
    """True for the repo-root files the sandbox reads its dependencies from."""
    return rel_path.replace("\\", "/") in settings.sandbox_requirements_files


class SandboxWarmer:
    def __init__(self, sandbox, list_repo_ids: ListRepoIds):
        self.sandbox = sandbox
        self._list_repo_ids = list_repo_ids
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Dict[str, None] = {}              # insertion-ordered set
        self._worker: Optional[asyncio.Task] = None
        self._startup: Optional[asyncio.Task] = None
        self._attempts: Dict[str, int] = {}
        self._retry: Dict[str, asyncio.TimerHandle] = {}
        self._prune_ids: Optional[List[str]] = None
        self._closed = False
        self.results: Dict[str, dict] = {}             # repo_id -> outcome of the last prewarm()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Call from the running event loop (application startup). Returns immediately."""
        if self._closed or self._loop is not None or not settings.sandbox_prewarm:
            return
        self._loop = asyncio.get_running_loop()
        _set_active(self)
        self._startup = self._loop.create_task(self._warm_all())

    async def stop(self) -> None:
        self._closed = True
        self.sandbox.close()                           # kills a running docker build, its thread then returns
        for handle in self._retry.values():
            handle.cancel()
        self._retry.clear()
        tasks = [t for t in (self._startup, self._worker) if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _clear_active(self)

    async def join(self) -> None:
        """Wait until everything queued so far is done (tests; retries that are only scheduled are not awaited)."""
        while True:
            tasks = [t for t in (self._startup, self._worker) if t is not None and not t.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ requests
    def request(self, repo_id: str) -> None:
        """Queue a (re)build check for one repo. Thread-safe; ignored before start(), after stop(),
        and when settings.sandbox_prewarm is off."""
        loop = self._loop
        if loop is None or self._closed or not settings.sandbox_prewarm or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is loop:
                self._enqueue(repo_id)
            else:
                loop.call_soon_threadsafe(self._enqueue, repo_id)
        except RuntimeError:
            pass                                       # the loop closed between the check and the call

    # ------------------------------------------------------------------ internals (event-loop thread only)
    def _enqueue(self, repo_id: str) -> None:
        if self._closed:
            return
        self._queue.setdefault(repo_id, None)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    async def _warm_all(self) -> None:
        try:
            ids = await self._list_repo_ids()
        except Exception as e:  # noqa: BLE001
            self.sandbox._emit(f"Could not list repositories for the sandbox warm-up: {type(e).__name__}: {e}", "WARNING")
            return
        for repo_id in ids[: max(0, settings.sandbox_prewarm_max_repos)]:
            self._enqueue(repo_id)
        if settings.sandbox_prune_images and ids:      # never prune on an empty list: that would delete every image
            self._prune_ids = list(ids)                # all repos, not just the warmed ones: their images are kept
            self._ensure_worker()

    async def _drain(self) -> None:
        # No await between the last emptiness check and the return: a request that arrives while this
        # task is finishing either lands before the check (and is served) or sees done() and starts a new one.
        while not self._closed:
            if self._queue:
                repo_id = next(iter(self._queue))
                del self._queue[repo_id]
                await self._warm_one(repo_id)
            elif self._prune_ids is not None:
                ids, self._prune_ids = self._prune_ids, None
                try:
                    await self.sandbox.prune_images(ids)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    self.sandbox._emit(f"Pruning sandbox images failed: {type(e).__name__}: {e}", "WARNING")
            else:
                break

    async def _warm_one(self, repo_id: str) -> None:
        try:
            res = await self.sandbox.prewarm(repo_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - background work must never take the application down
            res = {"repo_id": repo_id, "status": "failed", "note": f"{type(e).__name__}: {e}"}
            self.sandbox._emit(f"Sandbox warm-up for repo {repo_id} crashed: {res['note']}", "WARNING")
        self.results[repo_id] = res
        if res.get("retry") and not self._closed:
            n = self._attempts.get(repo_id, 0)
            if n < _MAX_RETRIES:
                self._attempts[repo_id] = n + 1
                old = self._retry.pop(repo_id, None)
                if old:
                    old.cancel()
                self._retry[repo_id] = asyncio.get_running_loop().call_later(_RETRY_S, self._enqueue, repo_id)
                return
        self._attempts.pop(repo_id, None)


# --------------------------------------------------------------------------- process-wide handle
_active: Optional[SandboxWarmer] = None


def _set_active(w: SandboxWarmer) -> None:
    global _active
    _active = w


def _clear_active(w: SandboxWarmer) -> None:
    global _active
    if _active is w:
        _active = None


def request_prewarm(repo_id: str) -> None:
    """Ask the running application's warmer to check/build the image for a repo. Safe from any thread
    (the file watcher calls it from its own); does nothing when no warmer is running (CLI runs, tests)."""
    w = _active
    if w is not None:
        w.request(repo_id)
