import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from services.notifier import notifier
import asyncio

class RepoChangeHandler(FileSystemEventHandler):
    def __init__(self, repo_id: str, root_path: Path):
        self.repo_id = repo_id
        self.root_path = root_path
        self._last_processed = 0

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file_change(event.src_path)

    def _handle_file_change(self, file_path_str: str):
        fpath = Path(file_path_str)
        
        from controllers.engine_controller import EngineController
        controller = EngineController()
        
        # Skip if file matches exclusion rules
        if controller.is_excluded(fpath, self.root_path):
            return

        now = time.time()
        if now - self._last_processed < 1.5:
            return
        self._last_processed = now

        try:
            rel_path = str(fpath.relative_to(self.root_path))
            asyncio.run(controller.incremental_reingest_file(self.repo_id, self.root_path, rel_path))
        except Exception as exc:
            notifier.notify("Incremental Ingestion Error", f"Failed to refresh {fpath.name}: {exc}", level="ERROR")


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
        observer.schedule(event_handler, str(root_path), recursive=True)
        observer.start()
        self.observers[repo_id] = observer
        notifier.notify("Watcher Active", f"Real-time monitoring enabled for repository: {root_path.name}")

    def stop_all(self):
        for obs in self.observers.values():
            obs.stop()
            obs.join()
        self.observers.clear()

watcher_manager = RepoWatcherManager()