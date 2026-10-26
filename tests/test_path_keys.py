"""Path-key + cross-platform regressions. Windows behaviour is simulated, so this runs the same on every OS.
Run:  pytest tests/test_path_keys.py -q
"""
import asyncio
import os
import sqlite3
import sys
import uuid
from pathlib import Path, PureWindowsPath

import pytest

from config.settings import settings
from controllers.engine_controller import EngineController
from models.database import DatabaseManager
from models.schemas import JobStatus
from services import sandbox_service as sbx
from services.apply_service import apply_revision
from services.notifier import notifier
from services.repo_store import RepoStore
from services.vector_store import VectorStoreService

run = asyncio.run


def test_rel_key_is_posix_even_for_windows_paths():
    key = EngineController.rel_key(PureWindowsPath("C:/r/services/a.py"), PureWindowsPath("C:/r"))
    assert key == "services/a.py"


def test_record_file_job_updates_one_row_for_either_separator():
    async def go():
        db, rid = DatabaseManager(), "r" + uuid.uuid4().hex[:8]
        await db.init_db()
        await db.record_file_job(rid, "sub\\a.py", "h1", JobStatus.PROCESSING)
        await db.record_file_job(rid, "sub/a.py", "h1", JobStatus.COMPLETED)      # used to raise IntegrityError
        assert await db.get_completed_file_hashes(rid) == {"sub/a.py": "h1"}
        with sqlite3.connect(db.db_path) as c:
            assert c.execute("SELECT COUNT(*) FROM file_jobs WHERE repo_id=?", (rid,)).fetchone()[0] == 1
    run(go())


def test_legacy_backslash_rows_are_migrated_once():
    async def go():
        db, rid = DatabaseManager(), "m" + uuid.uuid4().hex[:8]
        await db.init_db()
        with sqlite3.connect(db.db_path) as c:
            c.execute("INSERT INTO file_jobs (repo_id, file_path, file_hash, status) VALUES (?,?,?,?)", (rid, "a\\b.py", "old", "COMPLETED"))
            c.execute("INSERT INTO file_jobs (repo_id, file_path, file_hash, status) VALUES (?,?,?,?)", (rid, "x\\y.py", "h", "COMPLETED"))
            c.execute("INSERT INTO file_jobs (repo_id, file_path, file_hash, status) VALUES (?,?,?,?)", (rid, "a/b.py", "new", "COMPLETED"))   # twin
            c.execute("INSERT INTO code_chunks (chunk_id, repo_id, file_path) VALUES (?,?,?)", ("c" + rid, rid, "x\\y.py"))
            c.execute("PRAGMA user_version = 0")
        await db.init_db()
        assert await db.get_completed_file_hashes(rid) == {"a/b.py": "new", "x/y.py": "h"}      # twin: POSIX row wins
        assert await db.list_file_paths(rid) == ["x/y.py"]
        with sqlite3.connect(db.db_path) as c:
            assert c.execute("PRAGMA user_version").fetchone()[0] == 1
    run(go())


def _controller(monkeypatch):
    c = EngineController()
    monkeypatch.setattr(c.vector_service, "embed_and_persist_chunks", lambda *a, **k: None)
    monkeypatch.setattr(c.vector_service, "delete_chunks_by_file", lambda *a, **k: None)
    return c


def test_restart_does_not_reindex_unchanged_files_even_from_a_legacy_db(tmp_path, monkeypatch):
    """The reported bug: after a restart every file was 'Offline Change Detected'."""
    import services.watcher_service as ws
    monkeypatch.setattr(ws.watcher_manager, "start_watching", lambda *a, **k: None)
    events = []
    monkeypatch.setattr(notifier, "notify", lambda title, *a, **k: events.append((title, a[0] if a else "")))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")

    async def go():
        c = _controller(monkeypatch)
        repo = await c.ingest_repository(str(tmp_path))
        # turn the DB into what a Windows install had before the POSIX change
        with sqlite3.connect(c.db.db_path) as db:
            db.execute("UPDATE file_jobs SET file_path = replace(file_path, '/', char(92)) WHERE repo_id=?", (repo.repo_id,))
            db.execute("UPDATE code_chunks SET file_path = replace(file_path, '/', char(92)) WHERE repo_id=?", (repo.repo_id,))
            db.execute("PRAGMA user_version = 0")
        events.clear()
        await c.restore_all_watchers()                      # pytest has no TTY: prompt skipped, all repos watched
        assert not [e for e in events if e[0] == "Offline Change Detected"], events
        assert any(e[0] == "Repo Up-to-Date" for e in events)

        (tmp_path / "b.py").write_text("x = 2\n", encoding="utf-8")        # a real offline edit is still found, and only that
        events.clear()
        await c.restore_all_watchers()
        changed = [e[1] for e in events if e[0] == "Offline Change Detected"]
        assert len(changed) == 1 and changed[0].endswith("b.py")
    run(go())


def test_vector_delete_matches_both_separator_styles(monkeypatch):
    seen = {}

    class Store:
        def get(self, where):
            seen.update(where)
            return {"ids": []}

    svc = VectorStoreService()
    monkeypatch.setattr(svc, "_get_vectorstore", lambda rid: Store())
    svc.delete_chunks_by_file("r", "sub/a.py")
    assert set(seen["file_path"]["$in"]) == {"sub/a.py", "sub\\a.py"}


@pytest.mark.skipif(os.name == "nt", reason="no executable bit on Windows")
def test_apply_keeps_executable_bit(tmp_path):
    async def go():
        s, r = RepoStore(), "x" + uuid.uuid4().hex[:8]
        f = tmp_path / "run.sh"
        f.write_bytes(b"#!/bin/sh\necho 1\n")
        f.chmod(0o755)
        await s.upsert_base_files(r, [("run.sh", f.read_bytes())])
        rev = await s.create_revision(r, 0, "req", {"run.sh": b"#!/bin/sh\necho 2\n"}, {})
        assert (await apply_revision(s, r, rev, tmp_path))["ok"]
        assert f.read_bytes().endswith(b"echo 2\n") and os.access(f, os.X_OK)
    run(go())


def test_exclusion_patterns_ignore_case(tmp_path):
    c = EngineController.__new__(EngineController)
    assert c.is_excluded(tmp_path / "Build.LOG", tmp_path) and c.is_excluded(tmp_path / "x.ZIP", tmp_path)
    assert c.is_excluded(tmp_path / "__MACOSX" / "a.py", tmp_path)
    assert not c.is_excluded(tmp_path / "app.py", tmp_path)


def test_linux_bridge_network_maps_host_docker_internal(monkeypatch):
    d = sbx.DockerRunner()
    monkeypatch.setattr(settings, "sandbox_network", "bridge")
    monkeypatch.setattr(sys, "platform", "linux")
    assert "host.docker.internal:host-gateway" in d.docker_args("img:1", "n", 60, ["pytest"])
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "--add-host" not in d.docker_args("img:1", "n", 60, ["pytest"])
    monkeypatch.setattr(settings, "sandbox_network", "none")
    monkeypatch.setattr(sys, "platform", "linux")
    assert "--add-host" not in d.docker_args("img:1", "n", 60, ["pytest"])


def test_sqlite_url_is_absolute_on_posix(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    url = settings.sqlite_url
    assert url == f"sqlite+aiosqlite:///{(tmp_path.resolve() / settings.sqlite_db_name).as_posix()}"
    if os.name != "nt":
        assert url.startswith("sqlite+aiosqlite:////")
