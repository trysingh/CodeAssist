import hashlib
import os
import sys
import time
import re
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
from services.sandbox_warmer import SandboxWarmer, is_requirements_path, request_prewarm
from services.intent_router import COMMANDS_HELP, route_intent
from services.web_search_service import split_web_prefix
from models.schemas import GENERAL_REPO_ID
from services.patch_service import make_diff
from services.apply_service import apply_revision as apply_revision_to_disk

class EngineController:
    """MVC Controller coordinating Repository Ingestion, Job Resumption, and Conversational RAG."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.vector_service = VectorStoreService()
        self.attachments = AttachmentService(self.vector_service)
        self.rag_manager = RAGWorkflowManager(self.db, self.vector_service, self.attachments)
        self.repo_store = RepoStore()
        self.change_service = RepoChangeService(self.db, self.vector_service, self.repo_store)
        # Builds sandbox dependency images in the background. Inert until warmer.start() is called from the
        # running application (views/api_view.py); the file watcher's short-lived controllers never start one.
        self.warmer = SandboxWarmer(self.change_service.sandbox, self._warm_repo_ids)

    async def _warm_repo_ids(self) -> List[str]:
        await self.db.init_db()
        return await self.db.list_repo_ids_by_activity()

    @staticmethod
    def compute_repo_id(root_path: Path) -> str:
        resolved = str(root_path.resolve())
        return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]

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
        
        # Check excluded directories
        for part in rel_parts:
            if part in settings.exclude_dirs or part.startswith("."):
                return True
                
        # Check file exclusion glob patterns
        for pattern in settings.exclude_patterns:
            if fpath.match(pattern):
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
                if fpath.suffix.lower() in supported_exts and not self.is_excluded(fpath, root_path):
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

        # Non-blocking Windows/Unix console input reader with 5s timeout
        def get_input_with_timeout(timeout: float = 5.0) -> str:
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
            user_input = get_input_with_timeout(5.0)
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
            current_files = self.scan_supported_files(root_path)
            
            offline_updates_count = 0
            for fpath in current_files:
                rel_path = str(fpath.relative_to(root_path))
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
        kept, written, skipped, shadowed, batch = set(), 0, 0, {}, []

        async def flush():
            nonlocal written, skipped, batch
            if not batch:
                return
            out = await self.repo_store.upsert_base_files(repo_id, batch)
            kept.update(out["written"]); kept.update(out["unchanged"])
            written += len(out["written"]); skipped += len(out["skipped"])
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
        full_path = root_path / rel_path
        if not full_path.exists() or self.is_excluded(full_path, root_path):
            return

        fhash = self.compute_file_hash(full_path)
        completed_hashes = await self.db.get_completed_file_hashes(repo_id)

        if completed_hashes.get(rel_path) == fhash:
            return

        notifier.notify("Code Change Detected", f"Re-chunking and refreshing vectors for: {rel_path}", repo_id=repo_id)

        ext = full_path.suffix.lower()
        content = full_path.read_text(encoding="utf-8", errors="replace")

        raw_bytes = full_path.read_bytes()
        raw = await self.repo_store.upsert_base_files(repo_id, [(rel_path.replace("\\", "/"), raw_bytes)])
        if raw["shadowed"]:
            await self._record_disk_sync(repo_id, {raw["shadowed"][0]: raw_bytes})
        if is_requirements_path(rel_path):
            request_prewarm(repo_id)  # dependencies changed -> new image tag; build it now, not on the next change request

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
            self.vector_service.embed_and_persist_chunks(repo_id, chunk_docs)
            await self.db.save_chunks_batch(repo_id, sqlite_rows)

        await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.COMPLETED)
        
        notifier.notify(
            "Vectors Refreshed Successfully", 
            f"'{rel_path}' successfully updated. Old vectors replaced with {len(chunk_docs)} new chunks.", 
            level="SUCCESS",
            repo_id=repo_id
        )

    async def ingest_repository(self, repo_root_str: str) -> RepoInfo:
        await self.db.init_db()
        root_path = Path(repo_root_str).resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Repository root path does not exist: {root_path}")

        repo_id = self.compute_repo_id(root_path)
        title = root_path.name
        all_files = self.scan_supported_files(root_path)
        
        await self.db.upsert_repo(repo_id, title, str(root_path), JobStatus.PROCESSING, len(all_files))
        await self.snapshot_raw_files(repo_id, root_path)
        request_prewarm(repo_id)      # build the sandbox image while the chunks are being embedded (no-op without a running app)

        completed_hashes = await self.db.get_completed_file_hashes(repo_id)
        pending_tasks: List[Tuple[str, str, str, str, int, int]] = []
        
        for fpath in all_files:
            rel_path = str(fpath.relative_to(root_path))
            fhash = self.compute_file_hash(fpath)
            if completed_hashes.get(rel_path) == fhash:
                continue
            await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.PROCESSING)
            pending_tasks.append((repo_id, str(fpath), rel_path, fhash, settings.chunk_size, settings.chunk_overlap))

        batch_size = settings.batch_size
        for i in range(0, len(pending_tasks), batch_size):
            batch_tasks = pending_tasks[i : i + batch_size]
            chunked_results = parallel_chunk_files(batch_tasks, max_workers=settings.max_workers)
            
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
                self.vector_service.embed_and_persist_chunks(repo_id, batch_chunks_for_vector)
                await self.db.save_chunks_batch(repo_id, sqlite_rows)
                
            for rel_path, fhash, chunk_docs, err in chunked_results:
                if not err:
                    await self.db.record_file_job(repo_id, rel_path, fhash, JobStatus.COMPLETED)

        await self.db.update_repo_status(repo_id, JobStatus.COMPLETED)
        return await self.db.get_repo(repo_id)

    async def list_ingested_repos(self) -> List[RepoInfo]:
        await self.db.init_db()
        return await self.db.list_repos()

    async def chat_with_repo(self, repo_id: str, thread_id: str, query: str, web: str = "auto",
                             attachments: Optional[List[dict]] = None) -> str:
        """attachments = prepared attachments of THIS message. When the typed text is empty or vague, a request
        written inside an uploaded image becomes the user's message (then routed like typed text)."""
        await self.db.init_db()
        repo = await self.db.get_repo(repo_id)
        if not repo:
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        note = ""
        if attachments:
            found = await derive_request(query, attachments, repo_id=repo_id)
            if found.source == "image":
                query, note = found.request, found.note()
            elif not query.strip():
                query = "Please review the attached content."
        with repo_scope(repo_id):       # events raised while answering (web search, sandbox, ...) belong to this repo
            return note + await self._route_chat(repo_id, thread_id, query, web)

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

    async def _route_chat(self, repo_id: str, thread_id: str, query: str, web: str) -> str:
        query, web = split_web_prefix(query, web)
        if repo_id == GENERAL_REPO_ID:
            # No codebase: only explicit prefixes matter (model routing would misread "write a script..." as a change).
            decision = await route_intent(query, model="", repo_id=repo_id)
            if decision.intent == "help":
                return COMMANDS_HELP
            if decision.intent in ("change", "plan") and decision.source == "prefix":
                return "Code changes need a repository. Pick one in the selector, or ask your question here."
            return await self.rag_manager.invoke_chat(repo_id, thread_id, decision.query, web=web)
        decision = await route_intent(query, repo_id=repo_id)
        if decision.intent == "help":
            return COMMANDS_HELP
        if decision.intent == "plan":
            return await self.change_service.plan_only(repo_id, decision.query,
                                                       context=await self._thread_context(thread_id))
        if decision.intent == "change":
            ctx = await self._thread_context(thread_id)
            plan = await self._last_plan(thread_id, decision.query)
            return (await self.propose_change(repo_id, decision.query, context=ctx, plan=plan)).to_markdown()
        return await self.rag_manager.invoke_chat(repo_id, thread_id, decision.query, web=web)

    async def propose_change(self, repo_id: str, request: str, base_rev: Optional[int] = None,
                             context: str = "", plan: str = "") -> ChangeResult:
        await self.db.init_db()
        if not await self.db.get_repo(repo_id):
            raise ValueError(f"Repository ID '{repo_id}' not found.")
        return await self.change_service.run(repo_id, request, base_rev, context=context, plan=plan)

    async def list_revisions(self, repo_id: str) -> list:
        return await self.repo_store.list_revisions(repo_id)

    async def revision_diff(self, repo_id: str, rev: int) -> Optional[str]:
        if rev < 1 or not await self.repo_store.get_revision(repo_id, rev):
            return None
        changed = await self.repo_store.changed_files(repo_id, rev)
        return "\n".join(make_diff(p, b, a) for p, (b, a) in changed.items())

    async def revision_zip(self, repo_id: str, rev: int) -> Optional[bytes]:
        diff = await self.revision_diff(repo_id, rev)
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
        self.vector_service.delete_collection(repo_id)
        # 2. Clear SQLite chunks and jobs
        await self.db.reset_repo_indexes(repo_id)
        
        notifier.notify("Index Reset", f"Successfully cleared all indexes for repository {repo_id}.", level="SUCCESS", repo_id=repo_id)