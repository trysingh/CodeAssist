import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from services.notifier import notifier
import asyncio

class RepoChangeHandler(FileSystemEventHandler):
    DEBOUNCE_S = 1.5

    def __init__(self, repo_id: str, root_path: Path):
        self.repo_id = repo_id
        self.root_path = root_path
        self._timers = {}                       # path -> pending debounce timer
        self._lock = threading.Lock()

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_moved(self, event):
        # Atomic saves (vim, JetBrains safe-write, sed -i, os.replace) land as a rename: macOS/Linux report no
        # "modified" event for the target, only "moved". Windows editors mostly write in place.
        if event.is_directory:
            return
        self._handle_file_change(event.dest_path)

    def _handle_file_change(self, file_path_str: str):
        fpath = Path(file_path_str)
        
        from controllers.engine_controller import EngineController
        controller = EngineController()
        
        # Skip if file matches exclusion rules (or lies outside the watched root, e.g. a resolved symlink / macOS /private path)
        try:
            if controller.is_excluded(fpath, self.root_path):
                return
        except ValueError:
            return

        # Debounce PER FILE, trailing edge: re-index a file 1.5 s after its LAST event. A single shared timer silently dropped
        # every file but the first when several were written together (git apply, copying a folder, a formatter run), which left
        # the stored snapshot stale and made later patches fail against it; it also re-indexed half-written saves.
        key = str(fpath)
        with self._lock:
            pending = self._timers.pop(key, None)
            if pending:
                pending.cancel()
            timer = threading.Timer(self.DEBOUNCE_S, self._reingest, args=(fpath,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _reingest(self, fpath: Path):
        with self._lock:
            self._timers.pop(str(fpath), None)
        try:
            rel_path = fpath.relative_to(self.root_path).as_posix()
            from controllers.engine_controller import EngineController
            asyncio.run(EngineController().incremental_reingest_file(self.repo_id, self.root_path, rel_path))
        except Exception as exc:
            notifier.notify("Incremental Ingestion Error", f"Failed to refresh {fpath.name}: {exc}", level="ERROR", repo_id=self.repo_id)


class RepoWatcherManager:
    """Manages active background filesystem watchers for ingested repositories."""
    def __init__(self):
        self.observers = {}

    def start_watching(self, repo_id: str, root_path_str: str):
        if repo_id in self.observers:
            return
        root_path = Path(root_path_str)
        if not root_path.exists():
            return

        event_handler = RepoChangeHandler(repo_id, root_path)
        observer = Observer()
        try:
            observer.schedule(event_handler, str(root_path), recursive=True)
            observer.start()
        except OSError as exc:      # Linux/inotify: "watch limit reached" (ENOSPC) or "too many open files"
            notifier.notify("Watcher Not Started",
                            f"Could not watch '{root_path.name}': {exc}. On Linux raise the limit: "
                            f"sudo sysctl fs.inotify.max_user_watches=524288", level="ERROR", repo_id=repo_id)
            return
        self.observers[repo_id] = observer
        notifier.notify("Watcher Active", f"Real-time monitoring enabled for repository: {root_path.name}", repo_id=repo_id)

    def stop_all(self):
        for obs in self.observers.values():
            obs.stop()
            obs.join()
        self.observers.clear()

watcher_manager = RepoWatcherManager()