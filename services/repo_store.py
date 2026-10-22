"""Verbatim file store with revisions.

Chunks (Chroma + code_chunks) stay the SEARCH index. This store is the source of
truth for EDITS: exact bytes, exact line endings.

Model
-----
rev 0        base snapshot of the repo on disk (written by ingest / watcher)
rev 1..N     proposed edits; a revision stores ONLY the files it changed
resolution   a file at revision R = its newest row with rev <= R (deleted rows hide it)

Paths are always repo-relative with forward slashes, even on Windows. The older
code_chunks table keeps native separators, so callers must normalise when mixing.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import re
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

import aiosqlite

from config.settings import settings


class StaleRevision(Exception):
    """The repo moved on since the caller read it (someone created a newer revision)."""


# --------------------------------------------------------------------------- helpers
def safe_rel_path(p: str) -> str:
    """Normalise to a forward-slash repo-relative path, or raise ValueError."""
    s = (p or "").strip().strip("`'\"*").replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    if not s or "\x00" in s or s.startswith("/") or re.match(r"^[A-Za-z]:", s):
        raise ValueError(f"unsafe path: {p!r}")
    parts = PurePosixPath(s).parts
    if not parts or any(part in ("..", ".", "") for part in parts):
        raise ValueError(f"unsafe path: {p!r}")
    return "/".join(parts)


def is_secret_path(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(fnmatch.fnmatch(name, pat.lower()) for pat in settings.raw_store_exclude_patterns)


def is_text_bytes(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class RawFile:
    path: str
    data: bytes
    sha256: str
    rev: int

    @property
    def text(self) -> str:
        return self.data.decode("utf-8")


# --------------------------------------------------------------------------- store
class RepoStore:
    def __init__(self):
        self.db_path = settings.storage_dir / settings.sqlite_db_name
        self._ready = False

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS repo_files (
                    repo_id   TEXT NOT NULL,
                    rev       INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256    TEXT,
                    size      INTEGER,
                    content   BLOB,
                    deleted   INTEGER DEFAULT 0,
                    PRIMARY KEY (repo_id, rev, file_path)
                )""")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_repo_files_path ON repo_files(repo_id, file_path)")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS repo_revisions (
                    repo_id     TEXT NOT NULL,
                    rev         INTEGER NOT NULL,
                    parent      INTEGER,
                    request     TEXT,
                    report_json TEXT,
                    created_at  TEXT,
                    PRIMARY KEY (repo_id, rev)
                )""")
            await db.commit()
        self._ready = True

    # ------------------------------------------------------------------ base (rev 0)
    async def upsert_base_files(self, repo_id: str, files: Iterable[Tuple[str, bytes]]) -> Dict[str, list]:
        """Write rev-0 rows. Unchanged files are skipped.

        Returns {'written','unchanged','skipped','shadowed'}:
          skipped   -> [(path, reason)] secrets / too large / not UTF-8 text
          shadowed  -> paths whose disk content changed while a pending revision
                       overrides them (the revision no longer reflects disk).
        """
        await self._ensure()
        out: Dict[str, list] = {"written": [], "unchanged": [], "skipped": [], "shadowed": []}
        max_bytes = settings.raw_store_max_file_kb * 1024
        rows = []
        async with aiosqlite.connect(self.db_path) as db:
            for raw_path, data in files:
                try:
                    path = safe_rel_path(raw_path)
                except ValueError:
                    out["skipped"].append((raw_path, "unsafe path"))
                    continue
                if is_secret_path(path):
                    out["skipped"].append((path, "secret-like filename"))
                    continue
                if len(data) > max_bytes:
                    out["skipped"].append((path, f"larger than {settings.raw_store_max_file_kb} KB"))
                    continue
                if not is_text_bytes(data):
                    out["skipped"].append((path, "not UTF-8 text"))
                    continue
                digest = sha256_hex(data)
                async with db.execute(
                    "SELECT sha256 FROM repo_files WHERE repo_id=? AND rev=0 AND file_path=?",
                    (repo_id, path),
                ) as cur:
                    row = await cur.fetchone()
                if row and row[0] == digest:
                    out["unchanged"].append(path)
                    continue
                if row:
                    async with db.execute(
                        "SELECT 1 FROM repo_files WHERE repo_id=? AND file_path=? AND rev>0 LIMIT 1",
                        (repo_id, path),
                    ) as cur:
                        if await cur.fetchone():
                            out["shadowed"].append(path)
                rows.append((repo_id, 0, path, digest, len(data), zlib.compress(data, 6), 0))
                out["written"].append(path)
            if rows:
                await db.executemany(
                    "INSERT OR REPLACE INTO repo_files "
                    "(repo_id, rev, file_path, sha256, size, content, deleted) VALUES (?,?,?,?,?,?,?)",
                    rows)
            await db.commit()
        return out

    async def prune_base(self, repo_id: str, keep_paths: Iterable[str]) -> int:
        """Drop rev-0 rows for files no longer on disk. Call only after a full, successful scan."""
        await self._ensure()
        keep = set(keep_paths)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT file_path FROM repo_files WHERE repo_id=? AND rev=0", (repo_id,)) as cur:
                stale = [r[0] for r in await cur.fetchall() if r[0] not in keep]
            for i in range(0, len(stale), 400):
                part = stale[i:i + 400]
                marks = ",".join("?" for _ in part)
                await db.execute(
                    f"DELETE FROM repo_files WHERE repo_id=? AND rev=0 AND file_path IN ({marks})",
                    [repo_id] + part)
            await db.commit()
        return len(stale)

    # ------------------------------------------------------------------ reads
    async def latest_rev(self, repo_id: str) -> int:
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT COALESCE(MAX(rev),0) FROM repo_revisions WHERE repo_id=?", (repo_id,)) as cur:
                return (await cur.fetchone())[0]

    async def get_file(self, repo_id: str, path: str, rev: Optional[int] = None) -> Optional[RawFile]:
        await self._ensure()
        path = safe_rel_path(path)
        if rev is None:
            rev = await self.latest_rev(repo_id)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT rev, sha256, content, deleted FROM repo_files "
                "WHERE repo_id=? AND file_path=? AND rev<=? ORDER BY rev DESC LIMIT 1",
                (repo_id, path, rev),
            ) as cur:
                row = await cur.fetchone()
        if not row or row[3]:
            return None
        return RawFile(path=path, data=zlib.decompress(row[2]), sha256=row[1], rev=row[0])

    async def _index(self, db, repo_id: str, rev: int) -> Dict[str, str]:
        """path -> sha256 for every live file at `rev`."""
        async with db.execute(
            "SELECT f.file_path, f.sha256, f.deleted FROM repo_files f "
            "JOIN (SELECT file_path, MAX(rev) AS r FROM repo_files "
            "      WHERE repo_id=? AND rev<=? GROUP BY file_path) m "
            "  ON f.file_path=m.file_path AND f.rev=m.r "
            "WHERE f.repo_id=?",
            (repo_id, rev, repo_id),
        ) as cur:
            return {r[0]: r[1] for r in await cur.fetchall() if not r[2]}

    async def list_paths(self, repo_id: str, rev: Optional[int] = None) -> List[str]:
        await self._ensure()
        if rev is None:
            rev = await self.latest_rev(repo_id)
        async with aiosqlite.connect(self.db_path) as db:
            return sorted((await self._index(db, repo_id, rev)).keys())

    # ------------------------------------------------------------------ revisions
    async def create_revision(self, repo_id: str, parent: int, request: str,
                              changes: Dict[str, Optional[bytes]], report: dict) -> int:
        """Append a revision. `changes` maps path -> new bytes (None = delete).

        Raises StaleRevision if `parent` is no longer the latest revision.
        """
        await self._ensure()
        rows = []
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                "SELECT COALESCE(MAX(rev),0) FROM repo_revisions WHERE repo_id=?", (repo_id,)) as cur:
                latest = (await cur.fetchone())[0]
            if parent != latest:
                await db.execute("ROLLBACK")
                raise StaleRevision(f"base revision {parent} is stale (latest is {latest})")
            rev = latest + 1
            for raw_path, data in changes.items():
                path = safe_rel_path(raw_path)
                if data is None:
                    rows.append((repo_id, rev, path, None, 0, None, 1))
                else:
                    rows.append((repo_id, rev, path, sha256_hex(data), len(data), zlib.compress(data, 6), 0))
            await db.executemany(
                "INSERT INTO repo_files (repo_id, rev, file_path, sha256, size, content, deleted) "
                "VALUES (?,?,?,?,?,?,?)", rows)
            await db.execute(
                "INSERT INTO repo_revisions (repo_id, rev, parent, request, report_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (repo_id, rev, parent, request, json.dumps(report, default=str),
                 datetime.now(timezone.utc).isoformat()))
            await db.commit()
        return rev

    async def record_disk_sync(self, repo_id: str, changes: Dict[str, bytes]) -> Optional[int]:
        """The user edited files on disk that a revision also changes. Append a revision holding the
        disk content so new proposals build on what is on disk (history is kept, nothing is deleted)."""
        if not changes:
            return None
        for _ in range(2):
            latest = await self.latest_rev(repo_id)
            try:
                return await self.create_revision(
                    repo_id, latest, "disk sync: file(s) edited on disk", dict(changes),
                    {"kind": "disk-sync", "files": sorted(changes)})
            except StaleRevision:
                continue
        return None

    async def newer_revision_for(self, repo_id: str, path: str, after_rev: int) -> Optional[int]:
        """Newest revision above `after_rev` that changes `path`, if any."""
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT MAX(rev) FROM repo_files WHERE repo_id=? AND file_path=? AND rev>?",
                (repo_id, safe_rel_path(path), after_rev)) as cur:
                return (await cur.fetchone())[0]

    async def list_revisions(self, repo_id: str) -> List[dict]:
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT r.rev, r.parent, r.request, r.created_at, "
                "       (SELECT COUNT(*) FROM repo_files f WHERE f.repo_id=r.repo_id AND f.rev=r.rev) "
                "FROM repo_revisions r WHERE r.repo_id=? ORDER BY r.rev DESC", (repo_id,)) as cur:
                return [{"rev": r[0], "parent": r[1], "request": r[2], "created_at": r[3], "files": r[4]}
                        for r in await cur.fetchall()]

    async def get_revision(self, repo_id: str, rev: int) -> Optional[dict]:
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT rev, parent, request, report_json, created_at FROM repo_revisions "
                "WHERE repo_id=? AND rev=?", (repo_id, rev)) as cur:
                r = await cur.fetchone()
        if not r:
            return None
        return {"rev": r[0], "parent": r[1], "request": r[2],
                "report": json.loads(r[3] or "{}"), "created_at": r[4]}

    async def changed_files(self, repo_id: str, rev: int) -> Dict[str, Tuple[Optional[bytes], Optional[bytes]]]:
        """{path: (before, after)} for the files this revision touched. None = absent."""
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT file_path FROM repo_files WHERE repo_id=? AND rev=? ORDER BY file_path",
                (repo_id, rev)) as cur:
                paths = [r[0] for r in await cur.fetchall()]
        out = {}
        for p in paths:
            before = await self.get_file(repo_id, p, rev - 1) if rev > 0 else None
            after = await self.get_file(repo_id, p, rev)
            out[p] = (before.data if before else None, after.data if after else None)
        return out

    async def revert_to(self, repo_id: str, target_rev: int) -> int:
        """Append a revision that restores the repo to `target_rev`. No-op returns latest."""
        await self._ensure()
        latest = await self.latest_rev(repo_id)
        if target_rev > latest or target_rev < 0:
            raise ValueError(f"unknown revision {target_rev}")
        async with aiosqlite.connect(self.db_path) as db:
            now_idx = await self._index(db, repo_id, latest)
            then_idx = await self._index(db, repo_id, target_rev)
        changes: Dict[str, Optional[bytes]] = {}
        for p in set(now_idx) | set(then_idx):
            if now_idx.get(p) == then_idx.get(p):
                continue
            if p in then_idx:
                changes[p] = (await self.get_file(repo_id, p, target_rev)).data
            else:
                changes[p] = None
        if not changes:
            return latest
        return await self.create_revision(
            repo_id, latest, f"revert to revision {target_rev}", changes,
            {"kind": "revert", "to_rev": target_rev})

    async def index(self, repo_id: str, rev: Optional[int] = None) -> Dict[str, str]:
        """{path: sha256} of every live file at `rev` (default: latest)."""
        await self._ensure()
        if rev is None:
            rev = await self.latest_rev(repo_id)
        async with aiosqlite.connect(self.db_path) as db:
            return await self._index(db, repo_id, rev)

    async def snapshot_bytes(self, repo_id: str, rev: Optional[int] = None) -> Dict[str, bytes]:
        """{path: bytes} of the whole repo view at `rev`, in one query (used to build sandbox workspaces)."""
        await self._ensure()
        if rev is None:
            rev = await self.latest_rev(repo_id)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT f.file_path, f.content, f.deleted FROM repo_files f "
                "JOIN (SELECT file_path, MAX(rev) AS r FROM repo_files "
                "      WHERE repo_id=? AND rev<=? GROUP BY file_path) m "
                "  ON f.file_path=m.file_path AND f.rev=m.r "
                "WHERE f.repo_id=?", (repo_id, rev, repo_id)) as cur:
                return {r[0]: zlib.decompress(r[1]) for r in await cur.fetchall() if not r[2]}

    async def delete_base(self, repo_id: str, paths: Iterable[str]) -> int:
        """Remove rev-0 rows (the file was deleted from disk by an applied revision)."""
        await self._ensure()
        paths = [safe_rel_path(p) for p in paths]
        async with aiosqlite.connect(self.db_path) as db:
            for p in paths:
                await db.execute("DELETE FROM repo_files WHERE repo_id=? AND rev=0 AND file_path=?", (repo_id, p))
            await db.commit()
        return len(paths)

    async def mark_applied(self, repo_id: str, rev: int) -> None:
        await self._ensure()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT report_json FROM repo_revisions WHERE repo_id=? AND rev=?",
                                  (repo_id, rev)) as cur:
                row = await cur.fetchone()
            if not row:
                return
            report = json.loads(row[0] or "{}")
            report["applied_at"] = datetime.now(timezone.utc).isoformat()
            await db.execute("UPDATE repo_revisions SET report_json=? WHERE repo_id=? AND rev=?",
                             (json.dumps(report, default=str), repo_id, rev))
            await db.commit()

    # ------------------------------------------------------------------ extract
    async def materialize(self, repo_id: str, rev: Optional[int], dest: Path,
                          paths: Optional[Iterable[str]] = None) -> int:
        """Write files (exact bytes) under `dest`. Used by the sandbox in Phase 2."""
        await self._ensure()
        if rev is None:
            rev = await self.latest_rev(repo_id)
        wanted = list(paths) if paths is not None else await self.list_paths(repo_id, rev)
        root = Path(dest).resolve()
        root.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in wanted:
            f = await self.get_file(repo_id, p, rev)
            if not f:
                continue
            target = (root / f.path).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"path escapes workspace: {f.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f.data)
            n += 1
        return n

    async def export_zip(self, repo_id: str, rev: int, extra: Optional[Dict[str, bytes]] = None) -> bytes:
        """Zip of the files revision `rev` changed (rev 0 = everything), laid out repo-relative
        so it can be unzipped over the repo root. `extra` adds e.g. a changes.diff."""
        await self._ensure()
        buf = io.BytesIO()
        deleted: List[str] = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            if rev == 0:
                for p in await self.list_paths(repo_id, 0):
                    z.writestr(p, (await self.get_file(repo_id, p, 0)).data)
            else:
                for p, (_, after) in (await self.changed_files(repo_id, rev)).items():
                    if after is None:
                        deleted.append(p)
                    else:
                        z.writestr(p, after)
            if deleted:
                z.writestr("DELETED.txt", "\n".join(deleted) + "\n")
            for name, data in (extra or {}).items():
                z.writestr(name, data)
        return buf.getvalue()
