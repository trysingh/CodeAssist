import fnmatch
import hashlib
import os
import sys
import time
import re
import asyncio
import contextlib
import contextvars
import uuid
import orjson
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from config.settings import settings
from models.database import DatabaseManager
from models.schemas import JobStatus, RepoInfo
from services.chunking.registry import ChunkingRegistry, parallel_chunk_files
from services.vector_store import VectorStoreService
from services.rag_graph import RAGWorkflowManager
from services.attachment_service import AttachmentService
from services.attachment_intent import derive_request
from services.notifier import notifier, repo_scope
from services.repo_store import RepoStore
from services.repo_change import RepoChangeService, ChangeResult
from services.code_agent import CodeAgent
from services.sandbox_warmer import SandboxWarmer, is_requirements_path, request_prewarm
from services.intent_router import COMMANDS_HELP, route_intent
from services.web_search_service import split_web_prefix
from models.schemas import GENERAL_REPO_ID
from services.patch_service import join_diffs, make_diff
from services.apply_service import apply_revision as apply_revision_to_disk
from services.apply_service import disk_conflicts
from services.model_calls import announce_model, model_scope
from services.runners import current_session_key, runner_scope, thread_scope
from services.token_budget import budget, token_mode_scope
from services import action_log as al
from services.agentic_search import AgenticSearch
from services.vectorize_jobs import DbJobStore, JobState, VectorizeJobManager
from services.reasoning import BaselinePolicy, IntakeAgent, IntakeBrief, ReasoningSupervisor, VerifierAgent
from services.reasoning.baseline import CHAT_FACTS
from services.reasoning.crosscheck import facts_from_sandbox
from services.reasoning.stacking import find_repeat, repeat_reply
from services.reasoning.skill_pack import guidance_for

# Set by _chat_impl for the duration of one supervised turn (same asyncio task), read by _route_chat. Keeps _route_chat's
# 4-argument signature (tests and subclasses rely on it) while letting the supervisor see the routing decision and collect facts.
_PRE_DECISION: contextvars.ContextVar = contextvars.ContextVar("chat_pre_decision", default=None)
_CHAT_FACTS = CHAT_FACTS        # shared with services/code_agent.py (the ask agent adds its sandbox result), see reasoning/baseline.py

class EngineController:
    """MVC Controller coordinating Repository Ingestion, Job Resumption, and Conversational RAG."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.vector_service = VectorStoreService()
        self.attachments = AttachmentService(self.vector_service)
        self.rag_manager = RAGWorkflowManager(self.db, self.vector_service, self.attachments)
        self.repo_store = RepoStore()
        self.change_service = RepoChangeService(self.db, self.vector_service, self.repo_store)
        # /ask investigates with find/grep/read tools and checks proposed edits in the SAME sandbox (and pre-warmed image) /change uses
        self.rag_manager.code_agent = CodeAgent(self.repo_store, self.change_service.sandbox)
        # Builds sandbox dependency images in the background. Inert until warmer.start() is called from the
        # running application (views/api_view.py); the file watcher's short-lived controllers never start one.
        self.warmer = SandboxWarmer(self.change_service.sandbox, self._warm_repo_ids)
        self.active_chats: Dict[str, asyncio.Task] = {}
        self._chat_lock = asyncio.Lock()
        # ---- v2: background vectorize jobs, agentic file locator, reasoning supervisor
        self.jobs = VectorizeJobManager(DbJobStore(self.db), work=self.vectorize_repository, notify=notifier.notify,
                                        set_vector_status=self.db.set_vector_status)
        self.rag_manager.repo_store = self.repo_store       # rag_manager is built before repo_store exists
        self.rag_manager.jobs = self.jobs
        self.rag_manager.vector_status_fn = self.db.get_vector_status
        self._agentic: Dict[str, Tuple[object, AgenticSearch]] = {}     # repo_id -> (corpus it was built from, searcher)
        self.change_service.locate_fn = self._locate_files
        self.supervisor = self._build_supervisor()

    # ------------------------------------------------------------------ v2 helpers
    def _build_supervisor(self) -> ReasoningSupervisor:
        llm_json = None
        if settings.reasoning_intake_llm:
            from services.model_calls import ask_json, reasoning_model

            def llm_json(prompt: str) -> dict:
                """Sync callable for IntakeAgent; runs in the event loop's thread pool, so drive the coroutine to completion."""
                try:
                    return asyncio.run(ask_json(reasoning_model(), prompt, phase="INTAKE")) or {}
                except Exception:   # noqa: BLE001 - the heuristics stand on their own
                    return {}
        return ReasoningSupervisor(
            IntakeAgent(llm_json=llm_json, lane_override=settings.lane_override), VerifierAgent(),
            defaults_provider=lambda: {
                "change_gate": settings.change_gate, "change_review": settings.change_review,
                "change_tests_first": settings.change_tests_first, "change_author_tests": settings.change_author_tests,
                "change_max_attempts": settings.change_max_attempts},
            enabled=settings.reasoning_supervisor,
            guidance_provider=lambda brief, _policy: guidance_for(
                brief, enabled=settings.skill_pack_enabled, pack_dir=settings.skill_pack_dir,
                max_chars=budget("skill_pack_max_chars")))

    async def _warm_agentic(self, repo_id: str) -> None:
        """Build (or reuse) the repo's agentic searcher in the async path so the change graph's locator thread only reads it."""
        try:
            corpus = await self.rag_manager._agentic_corpus(repo_id)
            cur = self._agentic.get(repo_id)
            if cur is None or cur[0] is not corpus:
                self._agentic[repo_id] = (corpus, AgenticSearch(corpus))
        except Exception:   # noqa: BLE001 - no raw store yet / bare controller: the locator simply finds nothing
            return

    def _locate_files(self, query: str, k: int) -> List[str]:
        """locate_fn of the change graph (runs in a worker thread): the repo comes from notifier.repo_scope."""
        from services.notifier import _current_repo
        entry = getattr(self, "_agentic", {}).get(_current_repo.get() or "")
        return entry[1].locate_files(query, k) if entry else []

    def invalidate_agentic(self, repo_id: str) -> None:
        try:
            getattr(self, "_agentic", {}).pop(repo_id, None)
            self.rag_manager._corpus_cache.pop(repo_id, None)
        except Exception:   # noqa: BLE001
            pass

    async def _vector_status(self, repo_id: str) -> str:
        getter = getattr(self.db, "get_vector_status", None)
        if getter is None:
            return "ready"      # a database without the column is a pre-v2 one: its chunks were embedded at ingest
        try:
            status = str(await getter(repo_id) or "none")
            if status == "none" and getattr(self.db, "list_file_paths", None) is not None \
                    and await self.db.list_file_paths(repo_id):
                return "ready"      # ingested before vector_status existed: its chunks are a usable index (same rule as rag_graph)
            return status
        except Exception:   # noqa: BLE001
            return "none"

    async def _db_call(self, name: str, *args) -> None:
        fn = getattr(self.db, name, None)
        if fn is not None:
            await fn(*args)

    async def reconcile_jobs(self) -> int:
        """At application start no vectorize job can still be running: close stale rows and fix vector_status."""
        await self.db.init_db()
        n = 0
        for j in await self.db.list_jobs(None, 500):
            if j["state"] in (JobState.QUEUED.value, JobState.RUNNING.value):
                j = {**j, "state": JobState.FAILED.value, "error": "interrupted by restart"}
                await self.db.upsert_job(j)
                n += 1
        for r in await self.db.list_repos():
            if await self.db.get_vector_status(r.repo_id) in ("queued", "running"):
                await self.db.set_vector_status(r.repo_id, "failed")
        if n:
            notifier.notify("Vectorize jobs interrupted",
                            f"{n} vectorize job(s) were interrupted by a restart. They resume from the last finished file "
                            f"the next time a search needs vectors, or via POST /api/v1/repos/<id>/vectorize.", level="WARNING")
        return n

    async def start_vectorize(self, repo_id: str, reason: str = "requested by user"):
        """Enqueue (deduplicated) a background vectorize job for an ingested repo."""
        await self.db.init_db()
        if not await self.db.get_repo(repo_id):
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        return await self.jobs.enqueue(repo_id, reason or "requested by user")

    async def list_repos_enriched(self) -> List[dict]:
        """Repo list for the API/UI: RepoInfo + vector_status + ingest_mode. An agentic repo is searchable as soon as it is
        ingested, so its processed count is the file total until chunks exist (the old counter only counted embedded files)."""
        out = []
        for r in await self.list_ingested_repos():
            d = r.model_dump(mode="json")
            vs = await self._vector_status(r.repo_id)
            mode = "agentic"
            try:
                mode = await self.db.get_ingest_mode(r.repo_id)
            except Exception:   # noqa: BLE001
                pass
            d["vector_status"], d["ingest_mode"] = vs, mode
            if r.status == JobStatus.COMPLETED and vs != "ready":
                d["processed_files"] = r.total_files
            out.append(d)
        return out

    async def search_mode(self, repo_id: str) -> dict:
        await self.db.init_db()
        if not await self.db.get_repo(repo_id):
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        vs = await self._vector_status(repo_id)
        mode = str(settings.retrieval_mode or "agentic").lower()
        if vs == "ready":
            retrieval = "vector" if mode == "vector" else "hybrid"
        else:
            retrieval = "agentic"
        return {"ingest_mode": await self.db.get_ingest_mode(repo_id), "vector_status": vs,
                "retrieval_mode": retrieval, "job": await self.db.latest_job(repo_id, "vectorize")}

    async def _warm_repo_ids(self) -> List[str]:
        await self.db.init_db()
        return await self.db.list_repo_ids_by_activity()

    @staticmethod
    def compute_repo_id(root_path: Path) -> str:
        resolved = str(root_path.resolve())
        return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def rel_key(fpath: Path, root_path: Path) -> str:
        """The ONE repo-relative key used for DB / vector / raw-store lookups: POSIX separators on every OS."""
        return fpath.relative_to(root_path).as_posix()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_excluded(self, fpath: Path, root_path: Path) -> bool:
        """Checks if a file or folder matches Pydantic exclusion rules."""
        rel_parts = fpath.relative_to(root_path).parts

        # Placeholder env templates (.env.example ...) are dot-files but belong to the raw store: the sandbox needs them.
        if (fpath.name.lower() in {n.lower() for n in settings.sandbox_env_files}
                and not any(p in settings.exclude_dirs or p.startswith(".") for p in rel_parts[:-1])):
            return False

        # Check excluded directories
        for part in rel_parts:
            if part in settings.exclude_dirs or part.startswith("."):
                return True
                
        # Check file exclusion glob patterns
        name = fpath.name.lower()      # same result on case-sensitive (Linux) and case-insensitive (Windows/macOS) disks
        for pattern in settings.exclude_patterns:
            if fnmatch.fnmatchcase(name, pattern.lower()) or fpath.match(pattern):
                return True
                
        return False

    def scan_supported_files(self, root_path: Path) -> List[Path]:
        supported_exts = set(ChunkingRegistry.supported_extensions())
        matched_files: List[Path] = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune excluded directories in-place during os.walk
            dirnames[:] = [d for d in dirnames if d not in settings.exclude_dirs and not d.startswith(".")]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in supported_exts and not self.is_excluded(fpath, root_path) and fpath.is_file():   # skips dangling symlinks
                    matched_files.append(fpath)
        return matched_files

    async def restore_all_watchers(self):
        """
        Interactive startup prompt with a non-blocking 5-second timeout.
        Automatically defaults to ALL repositories if no input is given.
        """
        from services.watcher_service import watcher_manager
        from services.notifier import notifier
        
        repos = await self.list_ingested_repos()
        if not repos:
            return

        sys.stdout.write("\n[CodeRAG Engine] Ingested Repositories:\n")
        for idx, r in enumerate(repos, 1):
            sys.stdout.write(f"  [{idx}] {r.title} (ID: {r.repo_id}) — Path: {r.root_path}\n")
        
        sys.stdout.write(
            "\n[?] Would you like to watch ALL repositories or select specific ones?\n"
            "    Enter space-separated numbers (e.g., '1 2') or wait 5s to watch ALL: "
        )
        sys.stdout.flush()

        # Console input reader with 5s timeout (Windows: msvcrt, macOS/Linux: select). Runs in a worker thread so the
        # event loop (sandbox pre-warm etc.) is not frozen meanwhile; no terminal attached (service, Docker, nohup) = no prompt.
        def get_input_with_timeout(timeout: float = 5.0) -> str:
            if not sys.stdin or not sys.stdin.isatty():
                return ""
            start_t = time.time()
            buffer = []
            while time.time() - start_t < timeout:
                if os.name == 'nt':
                    import msvcrt
                    if msvcrt.kbhit():
                        char = msvcrt.getch()
                        if char in (b'\r', b'\n'):
                            sys.stdout.write('\n')
                            sys.stdout.flush()
                            return ''.join(b.decode('utf-8', errors='ignore') for b in buffer).strip()
                        elif char == b'\x08': # Backspace
                            if buffer:
                                buffer.pop()
                                sys.stdout.write('\b \b')
                                sys.stdout.flush()
                        else:
                            buffer.append(char)
                            sys.stdout.write(char.decode('utf-8', errors='ignore'))
                            sys.stdout.flush()
                else:
                    import select
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        return sys.stdin.readline().strip()
                time.sleep(0.05)
            raise TimeoutError()

        selected_repos = repos
        try:
            user_input = await asyncio.to_thread(get_input_with_timeout, 5.0)
            if user_input:
                chosen = []
                for part in user_input.split():
                    if part.isdigit():
                        i = int(part) - 1
                        if 0 <= i < len(repos):
                            chosen.append(repos[i])
                if chosen:
                    selected_repos = chosen
                    sys.stdout.write(f"[INFO] Selected {len(selected_repos)} repository/repositories for monitoring.\n")
                else:
                    sys.stdout.write("[INFO] Invalid selection. Defaulting to ALL repositories.\n")
            else:
                sys.stdout.write("[INFO] No input detected. Defaulting to watching ALL repositories.\n")
        except TimeoutError:
            sys.stdout.write("\n[TIMEOUT] 5-second limit reached. Automatically defaulting to watching ALL repositories.\n")
            sys.stdout.flush()

        # Perform reconciliation and start live watchers
        for repo in selected_repos:
            root_path = Path(repo.root_path)
            if repo.status.value != "COMPLETED" or not root_path.exists():
                continue

            notifier.notify("Syncing Repository", f"Checking for offline updates in '{repo.title}'...", level="INFO", repo_id=repo.repo_id)
            completed_hashes = await self.db.get_completed_file_hashes(repo.repo_id)
            # Without a vector index there is nothing per-file to re-embed: the raw snapshot sync below is the whole catch-up.
            current_files = self.scan_supported_files(root_path) if await self._vector_status(repo.repo_id) == "ready" else []
            
            offline_updates_count = 0
            for fpath in current_files:
                rel_path = self.rel_key(fpath, root_path)
                current_hash = self.compute_file_hash(fpath)
                
                if completed_hashes.get(rel_path) != current_hash:
                    offline_updates_count += 1
                    notifier.notify("Offline Change Detected", f"Catching up on modified file: {rel_path}", repo_id=repo.repo_id)
                    await self.incremental_reingest_file(repo.repo_id, root_path, rel_path)

            if offline_updates_count > 0:
                notifier.notify("Sync Complete", f"Caught up on {offline_updates_count} file(s) changed while offline.", level="SUCCESS", repo_id=repo.repo_id)
            else:
                notifier.notify("Repo Up-to-Date", f"No offline changes detected for '{repo.title}'.", level="INFO", repo_id=repo.repo_id)

            # Non-chunked files (requirements.txt, *.txt ...) are skipped by the scan above, so edits made
            # while the app was off never reached the raw store the sandbox reads. Sync them too.
            raw = await self.snapshot_raw_files(repo.repo_id, root_path)
            if raw["updated"]:
                request_prewarm(repo.repo_id)   # dependencies may have changed -> rebuild the sandbox image
            watcher_manager.start_watching(repo.repo_id, str(root_path))

    async def snapshot_raw_files(self, repo_id: str, root_path: Path) -> dict:
        """Persist verbatim copies of every non-excluded text file (rev 0). Cheap: unchanged files are skipped."""
        kept, written, skipped, shadowed, batch, withheld = set(), 0, 0, {}, [], []

        async def flush():
            nonlocal written, skipped, batch
            if not batch:
                return
            out = await self.repo_store.upsert_base_files(repo_id, batch)
            kept.update(out["written"]); kept.update(out["unchanged"])
            written += len(out["written"]); skipped += len(out["skipped"])
            withheld.extend(p for p, why in out["skipped"] if why.startswith(("secret-like", "looks like")))
            shadowed.update({p: b for p, b in batch if p in out["shadowed"]})
            batch = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in settings.exclude_dirs and not d.startswith(".")]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if self.is_excluded(fpath, root_path):
                    continue
                try:
                    batch.append((fpath.relative_to(root_path).as_posix(), fpath.read_bytes()))
                except OSError:
                    continue
                if len(batch) >= 200:
                    await flush()
        await flush()
        pruned = await self.repo_store.prune_base(repo_id, kept)
        sys.stdout.write(f"[RAW STORE] repo={repo_id} stored={len(kept)} updated={written} skipped={skipped} pruned={pruned}\n")
        if withheld:        # silent drops used to look like "the sandbox cannot find my module"
            notifier.notify("Files kept out of the sandbox",
                            f"{len(withheld)} file(s) look like secrets and are not stored, so the sandbox and the patcher cannot see them: "
                            f"{', '.join(sorted(withheld)[:6])}{' ...' if len(withheld) > 6 else ''}",
                            level="WARNING", repo_id=repo_id)
        await self._record_disk_sync(repo_id, shadowed)
        return {"stored": len(kept), "updated": written, "skipped": skipped}

    async def _record_disk_sync(self, repo_id: str, changed: Dict[str, bytes]) -> None:
        """Files edited on disk that a proposed revision also touches: new proposals must build on the disk version."""
        rev = await self.repo_store.record_disk_sync(repo_id, changed)
        if rev:
            notifier.notify("Disk edit recorded",
                            f"{len(changed)} file(s) changed on disk while proposals touched them (e.g. {sorted(changed)[0]}). "
                            f"Recorded as revision {rev}; new proposals build on your version.",
                            level="WARNING", repo_id=repo_id)

    async def incremental_reingest_file(self, repo_id: str, root_path: Path, rel_path: str):
        rel_path = rel_path.replace("\\", "/")          # callers may still pass a Windows-style key
        full_path = root_path / rel_path
        if not full_path.exists() or self.is_excluded(full_path, root_path):
            return

        if full_path.name.lower() in {n.lower() for n in settings.sandbox_env_files}:
            # env templates live in the raw store only (sandbox config); they are not searchable code
            await self.repo_store.upsert_base_files(repo_id, [(rel_path.replace("\\", "/"), full_path.read_bytes())])
            return

        # The verbatim snapshot is what agentic search, the sandbox and the patcher read: refresh it for every change.
        raw_bytes = full_path.read_bytes()
        raw = await self.repo_store.upsert_base_files(repo_id, [(rel_path, raw_bytes)])
        if raw["shadowed"]:
            await self._record_disk_sync(repo_id, {raw["shadowed"][0]: raw_bytes})
        if is_requirements_path(rel_path):
            request_prewarm(repo_id)  # dependencies changed -> new image tag; build it now, not on the next change request
        self.invalidate_agentic(repo_id)
        al.log_action("INGEST", "Watcher", "refresh raw snapshot", "ok", rel_path, repo_id=repo_id)

        # Embeddings only exist (and so only need refreshing) once the repo has been vectorized.
        if await self._vector_status(repo_id) != "ready":
            return

        fhash = self.compute_file_hash(full_path)
        completed_hashes = await self.db.get_completed_file_hashes(repo_id)

        if completed_hashes.get(rel_path) == fhash:
            return

        notifier.notify("Code Change Detected", f"Re-chunking and refreshing vectors for: {rel_path}", repo_id=repo_id)

        ext = full_path.suffix.lower()
        content = raw_bytes.decode("utf-8", errors="replace")

        strategy = ChunkingRegistry.get_strategy(ext)
        chunk_docs = strategy.chunk_file(repo_id, rel_path, fhash, content, settings.chunk_size, settings.chunk_overlap)

        sqlite_rows = []
        for ch in chunk_docs:
            sqlite_rows.append((
                ch.chunk_id, repo_id, rel_path, ch.metadata.chunk_index,
                ch.metadata.prev_chunk_id, ch.metadata.next_chunk_id,
                ch.metadata.sibling_group_id, ch.metadata.structural_context,
                ch.content, orjson.dumps(ch.metadata.model_dump()).decode("utf-8")
            ))

        if chunk_docs:
            self.vector_service.delete_chunks_by_file(repo_id, rel_path)
            await asyncio.to_thread(self.vector_service.embed_and_persist_chunks, repo_id, chunk_docs)
            await self.db.save_chunks_batch(repo_id, sqlite_rows)

        await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.COMPLETED)
        
        notifier.notify(
            "Vectors Refreshed Successfully", 
            f"'{rel_path}' successfully updated. Old vectors replaced with {len(chunk_docs)} new chunks.", 
            level="SUCCESS",
            repo_id=repo_id
        )

    async def ingest_repository(self, repo_root_str: str, *, vectorize: bool = False, watch: bool = True) -> RepoInfo:
        """FAST agentic ingest: scan, raw snapshot, repo registered and COMPLETED (searchable by grep/glob/read/symbols) with NO
        embeddings. vectorize=True or settings.ingest_mode == "vector" additionally runs the background vectorize job
        (see vectorize_repository) and waits for it before returning, which is the classic blocking behaviour."""
        t0 = time.perf_counter()
        await self.db.init_db()
        root_path = Path(repo_root_str).resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Repository root path does not exist: {root_path}")

        repo_id = self.compute_repo_id(root_path)
        title = root_path.name
        want_vectors = vectorize or str(settings.ingest_mode).lower() == "vector"
        with al.request_scope(None, None, repo_id=repo_id):
            with al.action("INGEST", "Ingest", "scan files", detail=str(root_path)) as h:
                all_files = await asyncio.to_thread(self.scan_supported_files, root_path)
                h.note(f"{len(all_files)} supported file(s)")

            await self.db.upsert_repo(repo_id, title, str(root_path), JobStatus.PROCESSING, len(all_files))
            with al.action("INGEST", "Ingest", "snapshot raw files") as h:
                snap = await self.snapshot_raw_files(repo_id, root_path)
                h.note(f"stored={(snap or {}).get('stored')} updated={(snap or {}).get('updated')}")
            request_prewarm(repo_id)      # build the sandbox image while the repo is being used (no-op without a running app)
            self.invalidate_agentic(repo_id)

            prior = await self._vector_status(repo_id) if hasattr(self.db, "get_vector_status") else "none"
            await self._db_call("set_ingest_mode", repo_id, "vector" if want_vectors else "agentic")
            if prior not in ("ready", "queued", "running"):
                await self._db_call("set_vector_status", repo_id, "none")
            # Searchable now: status COMPLETED and the file total (upsert keeps the counts consistent without touching file_jobs,
            # which belong to vectorize_repository).
            await self.db.upsert_repo(repo_id, title, str(root_path), JobStatus.COMPLETED, len(all_files))
            secs = time.perf_counter() - t0
            notifier.notify("Repo ready (agentic search)",
                            f"Repo ready (agentic search) in {secs:.1f}s: {len(all_files)} file(s) in '{title}'."
                            + (" Building the vector index next." if want_vectors else ""),
                            level="SUCCESS", repo_id=repo_id)
            al.log_action("INGEST", "Ingest", "repo ready", "ok", f"{len(all_files)} file(s) in {secs:.1f}s",
                          data={"files": len(all_files), "mode": "vector" if want_vectors else "agentic"}, duration_ms=int(secs * 1000))

            if watch and getattr(settings, "watch_on_ingest", True):
                try:
                    from services.watcher_service import watcher_manager
                    watcher_manager.start_watching(repo_id, str(root_path))
                except Exception as exc:    # noqa: BLE001 - watchdog missing / inotify limit: ingest itself succeeded
                    notifier.notify("Watcher Not Started", f"{type(exc).__name__}: {exc}", level="WARNING", repo_id=repo_id)

            # An already vectorized repo has stale embeddings after a re-ingest: refresh them too (resumes at the changed files).
            if want_vectors or prior == "ready":
                reason = "requested at ingest" if want_vectors else "refresh after re-ingest"
                if getattr(self, "jobs", None) is not None:
                    job = await self.jobs.enqueue(repo_id, reason)
                    if want_vectors:
                        done = await self.jobs.wait(job.job_id)
                        if done is not None and done.state == JobState.FAILED:
                            raise RuntimeError(f"Vectorizing failed: {done.error}")
                elif want_vectors:      # bare controller (no job manager): do the work inline
                    await self.vectorize_repository(repo_id, root_path=root_path)
                    await self._db_call("set_vector_status", repo_id, "ready")
        return await self.db.get_repo(repo_id)

    async def vectorize_repository(self, repo_id: str, progress_cb: Optional[Callable[[int, int], None]] = None,
                                   root_path: Optional[Path] = None) -> None:
        """Chunk + embed every supported file (resumes: files whose hash is already COMPLETED are skipped).
        progress_cb(done_files, total_files) is called after each file batch. Embedding runs in a worker thread."""
        await self.db.init_db()
        if root_path is None:
            repo = await self.db.get_repo(repo_id)
            if not repo:
                raise ValueError(f"Repository ID '{repo_id}' not found.")
            root_path = Path(repo.root_path)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Repository root path does not exist: {root_path}")
        all_files = await asyncio.to_thread(self.scan_supported_files, root_path)
        total = len(all_files)

        def hashes() -> Dict[str, Tuple[Path, str]]:
            return {self.rel_key(f, root_path): (f, self.compute_file_hash(f)) for f in all_files}
        current = await asyncio.to_thread(hashes)
        completed_hashes = await self.db.get_completed_file_hashes(repo_id)
        pending_tasks: List[Tuple[str, str, str, str, int, int]] = []
        for rel_path, (fpath, fhash) in current.items():
            if completed_hashes.get(rel_path) == fhash:
                continue
            await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.PROCESSING)
            pending_tasks.append((repo_id, str(fpath), rel_path, fhash, settings.chunk_size, settings.chunk_overlap))

        done = total - len(pending_tasks)
        if progress_cb:
            progress_cb(done, total)
        al.log_action("VECTORIZE", "Vectorizer", "plan", "info", f"{len(pending_tasks)} of {total} file(s) to embed",
                      data={"pending": len(pending_tasks), "total": total}, repo_id=repo_id)

        batch_size = max(1, settings.batch_size)
        for i in range(0, len(pending_tasks), batch_size):
            batch_tasks = pending_tasks[i : i + batch_size]
            chunked_results = await asyncio.to_thread(parallel_chunk_files, batch_tasks, max_workers=settings.max_workers)

            batch_chunks_for_vector = []
            sqlite_rows = []

            for rel_path, fhash, chunk_docs, err in chunked_results:
                if err:
                    await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.FAILED, err)
                    continue
                batch_chunks_for_vector.extend(chunk_docs)
                for ch in chunk_docs:
                    sqlite_rows.append((
                        ch.chunk_id, repo_id, rel_path, ch.metadata.chunk_index,
                        ch.metadata.prev_chunk_id, ch.metadata.next_chunk_id,
                        ch.metadata.sibling_group_id, ch.metadata.structural_context,
                        ch.content, orjson.dumps(ch.metadata.model_dump()).decode("utf-8")
                    ))

            if batch_chunks_for_vector:
                await asyncio.to_thread(self.vector_service.embed_and_persist_chunks, repo_id, batch_chunks_for_vector)
                await self.db.save_chunks_batch(repo_id, sqlite_rows)

            for rel_path, fhash, chunk_docs, err in chunked_results:
                if not err:
                    await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.COMPLETED)
            done += len(batch_tasks)
            al.log_action("VECTORIZE", "Vectorizer", "embed batch", "ok", f"{done}/{total} file(s), {len(batch_chunks_for_vector)} chunk(s)",
                          data={"done": done, "total": total, "chunks": len(batch_chunks_for_vector)}, repo_id=repo_id)
            if progress_cb:
                progress_cb(done, total)

        await self.db.update_repo_status(repo_id, JobStatus.COMPLETED)

    async def list_ingested_repos(self) -> List[RepoInfo]:
        await self.db.init_db()
        return await self.db.list_repos()

    async def _chat_impl(self, repo_id: str, thread_id: str, query: str, web: str,
                       attachments: Optional[List[dict]], model: str, request_id: Optional[str] = None,
                       runner: str = "", token_mode: str = "") -> str:
        """Actual chat execution - extracted for task tracking and cancellation support.
        Every step logged below carries `request_id` (the Activity timeline groups by it)."""
        await self.db.init_db()
        repo = await self.db.get_repo(repo_id)
        if not repo:
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        rid = request_id or uuid.uuid4().hex[:12]
        with al.request_scope(rid, thread_id, repo_id=repo_id):
            note = ""
            if attachments:
                found = await derive_request(query, attachments, repo_id=repo_id)
                if found.source == "image":
                    query, note = found.request, found.note()
                elif not query.strip():
                    query = "Please review the attached content."
            # runner / token_mode: the UI's per-request picks (validated by the API layer). Chat only: embeddings stay on settings.embedding_runner.
            with repo_scope(repo_id), model_scope(model), runner_scope(runner), token_mode_scope(token_mode), thread_scope(repo_id, thread_id):
                announce_model()            # "model resolved: runner=... picked=Default -> <real id>" (once per request)
                supervisor = getattr(self, "supervisor", None)
                if supervisor is None or not supervisor.enabled or repo_id == GENERAL_REPO_ID:
                    return note + await self._route_chat(repo_id, thread_id, query, web)
                pre = await self._decide(repo_id, query, web)
                decision = pre[2]
                if decision.intent == "help":           # the command list needs no intake / delivery notes
                    return note + await self._route_chat(repo_id, thread_id, query, web)
                try:
                    repo_files = await self.repo_store.list_paths(repo_id, 0)
                except Exception:   # noqa: BLE001
                    repo_files = []
                facts: dict = {}
                t_pre, t_facts = _PRE_DECISION.set(pre), _CHAT_FACTS.set(facts)

                async def execute(_brief: IntakeBrief, _policy: BaselinePolicy):
                    answer = await self._route_chat(repo_id, thread_id, query, web)
                    return answer, facts

                try:
                    answer = await supervisor.run(query=decision.query, intent=decision.intent,
                                                  repo_files=repo_files, execute=execute)
                finally:
                    _PRE_DECISION.reset(t_pre)
                    _CHAT_FACTS.reset(t_facts)
                return note + answer

    async def chat_with_repo(self, repo_id: str, thread_id: str, query: str, web: str = "auto",
                             attachments: Optional[List[dict]] = None, model: str = "",
                             request_id: Optional[str] = None, runner: str = "", token_mode: str = "") -> str:
        """Attachments = prepared attachments of THIS message. Supports cancellation via stop_chat().
        request_id: the caller may pre-generate it (the API returns it with the answer); default = generated per turn."""
        if not hasattr(self, 'active_chats'):
            self.active_chats = {}
        if not hasattr(self, '_chat_lock'):
            self._chat_lock = asyncio.Lock()

        async with self._chat_lock:
            prev = self.active_chats.get(thread_id)
            if prev and not prev.done():
                prev.cancel()
            task = asyncio.create_task(
                self._chat_impl(repo_id, thread_id, query, web, attachments, model, request_id, runner, token_mode),
                name=f"chat-{thread_id}"
            )
            self.active_chats[thread_id] = task

        try:
            return await task
        except asyncio.CancelledError:
            raise
        finally:
            async with self._chat_lock:
                if self.active_chats.get(thread_id) is task:
                    self.active_chats.pop(thread_id, None)

    async def stop_chat(self, thread_id: str) -> bool:
        """Cancels the asyncio task associated with the given thread ID."""
        if not hasattr(self, 'active_chats'):
            self.active_chats = {}
        if not hasattr(self, '_chat_lock'):
            self._chat_lock = asyncio.Lock()

        async with self._chat_lock:
            task = self.active_chats.pop(thread_id, None)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            return True
        return False


    async def _thread_context(self, thread_id: str, max_msgs: int = 6, per_msg: int = 700, total: int = 3000) -> str:
        """Last few turns of this thread, so '/change ... as discussed above' has something to refer to."""
        try:
            msgs = await self.db.get_thread_messages(thread_id)
        except Exception:
            return ""
        if msgs and msgs[-1]["role"] == "user":
            msgs = msgs[:-1]                        # the /change command itself: the API layer saved it before routing
        lines = [f"{m['role'].upper()}: {' '.join(str(m['content']).split())[:per_msg]}" for m in msgs[-max_msgs:]]
        return "\n".join(lines)[-total:]

    async def _last_plan(self, thread_id: str, request: str) -> str:
        """Reviewed /plan, only if the last assistant reply was a plan for this exact request."""
        try:
            msgs = await self.db.get_thread_messages(thread_id)
        except Exception:
            return ""
        last = next((str(m["content"]) for m in reversed(msgs) if m["role"] == "assistant"), "")
        m = re.search(r"To build it, send: `/change (.*)`\s*$", last, re.S)
        if not last.startswith("**Plan**") or not m or m.group(1).strip() != request.strip():
            return ""
        return last.split("\n\n", 2)[-1].rsplit("\n\nTo build it", 1)[0]

    async def _decide(self, repo_id: str, query: str, web: str):
        """-> (query without a /web prefix, web mode, RouteDecision). The one place that routes a message."""
        query, web = split_web_prefix(query, web)
        if repo_id == GENERAL_REPO_ID:
            # No codebase: only explicit prefixes matter (model routing would misread "write a script..." as a change).
            return query, web, await route_intent(query, model="", repo_id=repo_id)
        return query, web, await route_intent(query, repo_id=repo_id)

    def _note_facts(self, **facts) -> None:
        """Best-effort facts for the cross-check (only collected while a supervised turn is running)."""
        box = _CHAT_FACTS.get()
        if box is not None:
            box.update({k: v for k, v in facts.items() if v is not None})

    def _note_retrieval(self, thread_id: str) -> None:
        try:
            meta = self.rag_manager.last_retrieval_meta(thread_id)
        except Exception:   # noqa: BLE001
            meta = {}
        if meta:
            self._note_facts(retrieval_mode=meta.get("mode"), retrieval_confidence=meta.get("confidence"),
                             vectorize_job=meta.get("job_id"))

    async def _route_chat(self, repo_id: str, thread_id: str, query: str, web: str) -> str:
        pre = _PRE_DECISION.get()
        query, web, decision = pre if pre is not None else await self._decide(repo_id, query, web)
        if repo_id == GENERAL_REPO_ID:
            if decision.intent == "help":
                return decision.reply or COMMANDS_HELP
            if decision.intent in ("change", "plan") and decision.source == "prefix":
                return "Code changes need a repository. Pick one in the selector, or ask your question here."
            return await self.rag_manager.invoke_chat(repo_id, thread_id, decision.query, web=web)
        if decision.intent == "help":
            return decision.reply or COMMANDS_HELP
        if decision.intent == "plan":
            await self._warm_agentic(repo_id)
            return await self.change_service.plan_only(repo_id, decision.query,
                                                       context=await self._thread_context(thread_id))
        if decision.intent == "change":
            ctx = await self._thread_context(thread_id)
            plan = await self._last_plan(thread_id, decision.query)
            hit = await self._repeat_hit(repo_id, decision.query)
            if hit is not None:         # same request as the latest unapplied revision: answer with it, no model call, no new revision
                self._note_facts(skip_notes=True)
                return repeat_reply(hit, repo_id)
            res = await self.propose_change(repo_id, decision.query, context=ctx, plan=plan)
            # tests_* only when real tests ran (count > 0); a boot / import check alone sets boot_checked / import_checked
            try:        # best-effort: gathering facts for the notes must never fail the turn (the revision is already saved)
                facts = facts_from_sandbox(res.tests or {})
                facts.update(achieved_level=res.verification or facts.get("achieved_level"),     # one key, no duplicate kwargs
                             files_changed=[f.get("path") for f in (res.files or []) if f.get("path")] if res.ok else [],
                             retrieval_mode=str(settings.retrieval_mode),
                             static_checked=True if res.ok else None,      # a saved proposal passed apply_check (edits applied + static checks)
                             revision=res.revision if res.ok else None, plan=res.plan, change_files=list(res.files or []),
                             stacked_on=list(res.stacked_on or []))        # for the plain summary at the top of the reply
                self._note_facts(**facts)
            except Exception as e:
                sys.stdout.write(f"[REASONING] facts for the delivery notes skipped: {type(e).__name__}: {e}\n")
            return res.to_markdown()
        answer = await self.rag_manager.invoke_chat(repo_id, thread_id, decision.query, web=web)
        self._note_retrieval(thread_id)
        return answer

    async def _repeat_hit(self, repo_id: str, query: str) -> Optional[dict]:
        """The latest unapplied revision when `query` repeats its request (settings.repeat_request_guard), else None. Never raises."""
        if not settings.repeat_request_guard:
            return None
        try:
            return await find_repeat(self.repo_store, repo_id, query)
        except Exception as e:  # noqa: BLE001 - the guard is an optimisation: on any problem build the change as usual
            sys.stdout.write(f"[REASONING] repeat-request check skipped: {type(e).__name__}: {e}\n")
            return None

    async def propose_change(self, repo_id: str, request: str, base_rev: Optional[int] = None,
                             context: str = "", plan: str = "", model: str = "",
                             runner: str = "", token_mode: str = "") -> ChangeResult:
        await self.db.init_db()
        if not await self.db.get_repo(repo_id):
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        await self._warm_agentic(repo_id)        # the change graph's locator thread only reads this cache
        # called from a chat turn the scopes are already set (and thread_scope must keep that thread); /changes sets its own
        with model_scope(model), runner_scope(runner), token_mode_scope(token_mode), \
                (contextlib.nullcontext() if current_session_key() else thread_scope(repo_id, "change")):
            announce_model()                # no-op when the chat turn that called us already announced this request
            return await self.change_service.run(repo_id, request, base_rev, context=context, plan=plan)

    async def list_revisions(self, repo_id: str) -> list:
        return await self.repo_store.list_revisions(repo_id)

    async def revision_diff(self, repo_id: str, rev: int, from_disk: bool = False) -> Optional[str]:
        """Patch for `git apply`. Default: what this revision changed relative to the revision it was built on.
        from_disk=True: cumulative, ingest snapshot (what is on disk) -> this revision, so it applies to the untouched
        repo even when earlier revisions were never applied."""
        if rev < 1 or not await self.repo_store.get_revision(repo_id, rev):
            return None
        if from_disk:
            base_idx = await self.repo_store.index(repo_id, 0)
            view_idx = await self.repo_store.index(repo_id, rev)
            chunks = []
            for p in sorted(set(base_idx) | set(view_idx)):
                if base_idx.get(p) == view_idx.get(p):
                    continue
                before = await self.repo_store.get_file(repo_id, p, 0)
                after = await self.repo_store.get_file(repo_id, p, rev)
                chunks.append(make_diff(p, before.data if before else None, after.data if after else None))
            return join_diffs(chunks)
        changed = await self.repo_store.changed_files(repo_id, rev)
        return join_diffs([make_diff(p, b, a) for p, (b, a) in changed.items()])

    async def revision_patch(self, repo_id: str, rev: int, from_disk: bool = True) -> Optional[Tuple[str, List[dict]]]:
        """(patch text, conflicts) for a `git apply` download. Default is cumulative against the repo as the engine last saw it
        (the same base the Apply button uses), because that is what a working tree matches; conflicts lists files that changed
        on disk since, whose hunks git will reject. They are also written as `#` lines before the first file diff, which git
        and patch ignore."""
        diff = await self.revision_diff(repo_id, rev, from_disk)
        if diff is None:
            return None
        conflicts: List[dict] = []
        repo = await self.db.get_repo(repo_id)
        if from_disk and repo:
            conflicts = await disk_conflicts(self.repo_store, repo_id, rev, Path(repo.root_path))
        if conflicts:
            head = ["# WARNING: these files differ on disk from the snapshot this revision was built on, so git will reject their hunks.",
                    "# Re-ingest the repo and redo the change, or Apply with force from the app:"]
            head += [f"#   {c['path']}: {c['reason']}" for c in conflicts]
            diff = "\n".join(head) + "\n" + diff
        return diff, conflicts

    async def revision_zip(self, repo_id: str, rev: int) -> Optional[bytes]:
        diff = await self.revision_diff(repo_id, rev, from_disk=True)
        if diff is None:
            return None
        return await self.repo_store.export_zip(repo_id, rev, extra={"changes.diff": diff.encode("utf-8")})

    async def revert_revision(self, repo_id: str, to_rev: int) -> int:
        return await self.repo_store.revert_to(repo_id, to_rev)

    async def apply_revision(self, repo_id: str, rev: int, force: bool = False) -> dict:
        """Write a proposed revision into the repo folder (explicit user action)."""
        repo = await self.db.get_repo(repo_id)
        if not repo:
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        return await apply_revision_to_disk(self.repo_store, repo_id, rev, Path(repo.root_path), force)

    async def prepare_attachments(self, files: List[Tuple[str, str, bytes]]) -> List[dict]:
        """Extract text (OCR for images) from (filename, mime, bytes) tuples. Raises ValueError if unsupported."""
        prepared = []
        for name, mime, data in files:
            kind, text = await self.attachments.extract(name, mime, data)
            prepared.append({"filename": name, "mime": mime, "kind": kind, "size": len(data),
                             "text": text or "[No text could be extracted from this file.]"})
        return prepared

    async def store_attachments(self, thread_id: str, message_id: int, prepared: List[dict]) -> List[dict]:
        """Persist extracted text on the thread and index it so later questions can retrieve it."""
        saved = []
        for p in prepared:
            att_id = await self.db.save_attachment(thread_id, message_id, p["filename"], p["mime"],
                                                   p["kind"], p["size"], p["text"])
            try:
                await self.attachments.index(thread_id, att_id, p["filename"], p["text"])
            except Exception as exc:  # chat still works; large files fall back to head excerpts
                notifier.notify("Attachment not indexed", f"{p['filename']}: {exc}", level="WARNING")
            saved.append({"id": att_id, "filename": p["filename"], "kind": p["kind"], "chars": len(p["text"])})
        return saved

    async def reset_repo_index(self, repo_id: str):
        """Resets both the SQLite metadata and the Chroma vector store for a repo."""
        await self.db.init_db()
        repo = await self.db.get_repo(repo_id)
        if not repo:
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        
        # 1. Clear Chroma collection
        try:
            self.vector_service.delete_collection(repo_id)
        except Exception as e:
            print(f"Warning: Could not delete vector collection {repo_id}: {e}")
        # 2. Clear SQLite chunks and jobs
        try:
            await self.db.reset_repo_indexes(repo_id)
        except Exception as e:
            print(f"Warning: Could not reset repository indexes for {repo_id}: {e}")

        
        await self._db_call("set_vector_status", repo_id, "none")     # no chunks left: back to agentic search
        notifier.notify("Index Reset", f"Successfully cleared all indexes for repository {repo_id}.", level="SUCCESS", repo_id=repo_id)

    async def remove_repo(self, repo_id: str):
        """Permanently removes a repository, its index, and its metadata."""
        # 1. Remove from Vector Store (Chroma)
        # This stops the app from "watching" or searching the repo
        try:
            self.vector_service.delete_collection(repo_id)
        except Exception as e:
            print(f"Warning: Could not delete vector collection {repo_id}: {e}")
        
        # 2. Remove from Metadata Store (models/database.py)
        # This removes it from the "Ingested Repos" list in the UI
        try:
            await self.db.delete_repo(repo_id)
        except Exception as e:
            print(f"Warning: Could not delete metadata for {repo_id}: {e}")
        
        # 3. Remove from File Store (services/repo_store.py)
        # This cleans up the actual file paths and revisions
        if hasattr(self, 'repo_store'):
            try:
                await self.repo_store.delete_repo_data(repo_id)
            except Exception as e:
                print(f"Warning: Could not delete files for {repo_id}: {e}")
        
        notifier.notify(
            "Repository Removed", 
            f"Repository {repo_id} has been permanently deleted.", 
            level="SUCCESS", 
            repo_id=repo_id
        )