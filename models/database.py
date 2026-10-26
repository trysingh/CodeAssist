from datetime import datetime, timezone 
import aiosqlite
from typing import List, Dict, Optional, Any
from config.settings import settings
from models.schemas import GENERAL_REPO_ID, RepoInfo, JobStatus 

def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp compatible with Python 3.10-3.13+."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path_str: str) -> str:
    """Normalize file paths across Windows, macOS, and Linux to POSIX style."""
    if not path_str:
        return ""
    return str(path_str).replace("\\", "/")

class DatabaseManager:
    """Manages SQLite persistence for repository metadata, job states, and multi-thread chat history."""
    
    def __init__(self):
        self.db_path = settings.storage_dir / settings.sqlite_db_name

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    repo_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS file_jobs (
                    repo_id TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    status TEXT,
                    error TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (repo_id, file_path)
                )
            """)
            # Ensure backward compatibility for columns if table already existed
            async with db.execute("PRAGMA table_info(file_jobs)") as cursor:
                columns = [row[1] for row in await cursor.fetchall()]
                if "error" not in columns:
                    await db.execute("ALTER TABLE file_jobs ADD COLUMN error TEXT")
                if "updated_at" not in columns:
                    await db.execute("ALTER TABLE file_jobs ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

            # Deduplicate any older rows and ensure a unique index exists on (repo_id, file_path)
            await db.execute("""
                DELETE FROM file_jobs
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM file_jobs
                    GROUP BY repo_id, file_path
                )
            """)
            await db.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_file_jobs_repo_path 
                ON file_jobs (repo_id, file_path)
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (repo_id) REFERENCES repositories(repo_id) ON DELETE CASCADE
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_attachments (
                    attachment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_id INTEGER,
                    filename TEXT NOT NULL,
                    mime TEXT,
                    kind TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    extracted_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_chat_attachments_thread ON chat_attachments(thread_id)")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS code_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo_id TEXT,
                    file_path TEXT,
                    chunk_index INTEGER,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    sibling_group_id TEXT,
                    structural_context TEXT,
                    content TEXT,
                    metadata_json TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT NOT NULL UNIQUE,
                    ts TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    repo_id TEXT,
                    level TEXT NOT NULL DEFAULT 'INFO',
                    title TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_scope_repo ON telemetry_events(scope, repo_id, id)")

            # One-time migration (PRAGMA user_version 0 -> 1): older rows may hold Windows "a\\b.py" paths. Rewrite them to
            # POSIX "a/b.py" so every lookup uses a single key. If a POSIX twin already exists it wins (worst case: one re-index).
            async with db.execute("PRAGMA user_version") as cur:
                ver = (await cur.fetchone())[0]
            if ver < 1:
                await db.execute("""
                    DELETE FROM file_jobs WHERE instr(file_path, char(92)) > 0 AND EXISTS (
                        SELECT 1 FROM file_jobs t WHERE t.repo_id = file_jobs.repo_id
                        AND t.file_path = replace(file_jobs.file_path, char(92), '/'))
                """)
                await db.execute("UPDATE file_jobs SET file_path = replace(file_path, char(92), '/') "
                                 "WHERE instr(file_path, char(92)) > 0")
                await db.execute("UPDATE code_chunks SET file_path = replace(file_path, char(92), '/') "
                                 "WHERE instr(file_path, char(92)) > 0")
                await db.execute("PRAGMA user_version = 1")
            await db.execute(
                "INSERT OR IGNORE INTO repositories (repo_id, title, root_path, status, total_files, created_at) "
                "VALUES (?, 'General (web)', '', 'COMPLETED', 0, ?)", (GENERAL_REPO_ID, datetime.now(timezone.utc).isoformat()))
            await db.commit()

    async def create_thread(self, thread_id: str, repo_id: str, title: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO chat_threads (thread_id, repo_id, title) VALUES (?, ?, ?)",
                (thread_id, repo_id, title)
            )
            await db.commit()

    async def get_threads_for_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT thread_id, title, created_at FROM chat_threads WHERE repo_id = ? ORDER BY created_at DESC",
                (repo_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]

    async def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, role, content FROM chat_messages WHERE thread_id = ? ORDER BY id ASC",
                (thread_id,)
            ) as cursor:
                rows = await cursor.fetchall()
            async with db.execute(
                "SELECT message_id, attachment_id, filename, kind FROM chat_attachments WHERE thread_id = ? ORDER BY attachment_id",
                (thread_id,)
            ) as cursor:
                atts: Dict[int, list] = {}
                for a in await cursor.fetchall():
                    atts.setdefault(a[0], []).append({"id": a[1], "filename": a[2], "kind": a[3]})
        return [{"role": r[1], "content": r[2], "attachments": atts.get(r[0], [])} for r in rows]

    async def save_chat_message(self, thread_id: str, role: str, content: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
                (thread_id, role, content)
            )
            await db.commit()
            return cur.lastrowid

    async def save_attachment(self, thread_id: str, message_id: int, filename: str, mime: str,
                              kind: str, size_bytes: int, text: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "INSERT INTO chat_attachments (thread_id, message_id, filename, mime, kind, size_bytes, extracted_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, message_id, filename, mime, kind, size_bytes, text)
            )
            await db.commit()
            return cur.lastrowid

    async def get_thread_attachments(self, thread_id: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT attachment_id, filename, kind, extracted_text FROM chat_attachments "
                "WHERE thread_id = ? ORDER BY attachment_id", (thread_id,)
            ) as cursor:
                return [{"id": r[0], "filename": r[1], "kind": r[2], "text": r[3] or ""} for r in await cursor.fetchall()]

    async def upsert_repo(self, repo_id: str, title: str, root_path: str, status: JobStatus, total_files: int):
        now_iso = datetime.now(timezone.utc).isoformat()
        norm_root = _normalize_path(root_path)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO repositories (repo_id, title, root_path, status, total_files, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    title=excluded.title,
                    root_path=excluded.root_path,
                    status=excluded.status,
                    total_files=excluded.total_files
            """, (repo_id, title, norm_root, status.value, total_files, now_iso))
            await db.commit()

    async def update_repo_status(self, repo_id: str, status: JobStatus):
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM file_jobs WHERE repo_id = ?", (repo_id,)) as cursor:
                row = await cursor.fetchone()
                total_files = row[0] if row else 0

            await db.execute("""
                UPDATE repositories 
                SET status = ?, total_files = ?
                WHERE repo_id = ?
            """, (status.value, total_files, repo_id))
            await db.commit()

    async def get_repo(self, repo_id: str) -> Optional[RepoInfo]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT repo_id, title, root_path, status, total_files, processed_files, total_chunks, created_at
                FROM repositories WHERE repo_id = ?
            """, (repo_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                created_val = row[7] if row[7] else datetime.now(timezone.utc).isoformat()
                return RepoInfo(
                    repo_id=row[0], title=row[1], root_path=row[2],
                    status=JobStatus(row[3]), total_files=row[4],
                    processed_files=row[5], total_chunks=row[6],
                    created_at=str(created_val)
                )

    async def list_repos(self) -> List[RepoInfo]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT repo_id, title, root_path, status, total_files, processed_files, total_chunks, created_at
                FROM repositories WHERE repo_id != ? ORDER BY created_at DESC
            """, (GENERAL_REPO_ID,)) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    async with db.execute("SELECT COUNT(*) FROM file_jobs WHERE repo_id = ? AND status = 'COMPLETED'", (row[0],)) as count_cursor:
                        count_row = await count_cursor.fetchone()
                        processed = count_row[0] if count_row else 0
                    
                    created_val = row[7] if row[7] else datetime.now(timezone.utc).isoformat()
                    results.append(RepoInfo(
                        repo_id=row[0], title=row[1], root_path=row[2],
                        status=JobStatus(row[3]), total_files=row[4],
                        processed_files=processed, total_chunks=row[6],
                        created_at=str(created_val)
                    ))
                return results

    async def list_repo_ids_by_activity(self) -> List[str]:
        """Ingested (COMPLETED) repos, most recently chatted-with first; a repo nobody chatted with yet is
        ranked by its creation time. Used to decide which sandbox images to pre-build first at startup.
        The General (web) placeholder is not a codebase and is never listed."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT r.repo_id
                FROM repositories r
                LEFT JOIN (
                    SELECT t.repo_id AS repo_id, MAX(m.timestamp) AS last_msg
                    FROM chat_threads t JOIN chat_messages m ON m.thread_id = t.thread_id
                    GROUP BY t.repo_id
                ) a ON a.repo_id = r.repo_id
                WHERE r.repo_id != ? AND r.status = ?
                ORDER BY datetime(COALESCE(a.last_msg, r.created_at)) DESC, r.repo_id
            """, (GENERAL_REPO_ID, JobStatus.COMPLETED.value)) as cursor:
                return [row[0] for row in await cursor.fetchall()]

    async def record_file_job(self, repo_id: str, file_path: str, file_hash: str, status: JobStatus, error: Optional[str] = None):
        now_iso = datetime.now(timezone.utc).isoformat()
        norm_path = _normalize_path(file_path)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                UPDATE file_jobs
                SET file_hash = ?, status = ?, error = ?, updated_at = ?
                WHERE repo_id = ? AND file_path = ?
            """, (file_hash, status.value, error, now_iso, repo_id, norm_path))
            if cursor.rowcount == 0:
                await db.execute("""
                    INSERT INTO file_jobs (repo_id, file_path, file_hash, status, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (repo_id, norm_path, file_hash, status.value, error, now_iso))
            await db.commit()

    async def get_completed_file_hashes(self, repo_id: str) -> Dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT file_path, file_hash FROM file_jobs WHERE repo_id = ? AND status = 'COMPLETED'", (repo_id,)) as cursor:
                rows = await cursor.fetchall()
                return {_normalize_path(row[0]): row[1] for row in rows}

    async def save_chunks_batch(self, repo_id: str, chunk_records: List[tuple]):
        async with aiosqlite.connect(self.db_path) as db:
            if chunk_records:
                normalized_records = [
                    (
                        r[0], r[1], _normalize_path(r[2]), r[3],
                        r[4], r[5], r[6], r[7], r[8], r[9]
                    )
                    for r in chunk_records
                ]
                file_paths = list({r[2] for r in normalized_records})
                placeholders = ",".join("?" for _ in file_paths)
                await db.execute(
                    f"DELETE FROM code_chunks WHERE repo_id = ? AND file_path IN ({placeholders})",
                    [repo_id] + file_paths
                )
                await db.executemany("""
                    INSERT OR REPLACE INTO code_chunks (
                        chunk_id, repo_id, file_path, chunk_index,
                        prev_chunk_id, next_chunk_id, sibling_group_id,
                        structural_context, content, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, normalized_records)

            async with db.execute("SELECT COUNT(*) FROM code_chunks WHERE repo_id = ?", (repo_id,)) as cursor:
                row = await cursor.fetchone()
                total_chunks = row[0] if row else 0

            await db.execute("UPDATE repositories SET total_chunks = ? WHERE repo_id = ?", (total_chunks, repo_id))
            await db.commit()

    async def get_adjacent_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches neighbor chunks by their IDs for relational RAG context expansion."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(f"""
                SELECT chunk_id, file_path, chunk_index, structural_context, content
                FROM code_chunks
                WHERE chunk_id IN ({placeholders})
            """, chunk_ids) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "chunk_id": r[0],
                        "file_path": _normalize_path(r[1]),
                        "chunk_index": r[2],
                        "structural_context": r[3],
                        "content": r[4]
                    }
                    for r in rows
                ]

    async def list_file_paths(self, repo_id: str) -> List[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT DISTINCT file_path FROM code_chunks WHERE repo_id=?", (repo_id,)) as c:
                return [_normalize_path(r[0]) for r in await c.fetchall()]

    async def delete_thread(self, thread_id: str) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chat_attachments WHERE thread_id = ?", (thread_id,))
            await db.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
            cur = await db.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
            await db.commit()
            return cur.rowcount > 0
    
    async def reset_repo_indexes(self, repo_id: str):
        """Deletes all chunks and file job history for a specific repository."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM code_chunks WHERE repo_id = ?", (repo_id,))
            await db.execute("DELETE FROM file_jobs WHERE repo_id = ?", (repo_id,))
            await db.commit()

    
    # ------------------------------------------------------------------ telemetry & logs
    async def save_telemetry_events(self, events: List[Dict[str, Any]]) -> None:
        """Batch insert (one connection, one commit). A repeated uid is ignored, so a retry cannot duplicate a row."""
        if not events:
            return
        rows = [(e["uid"], e["ts"], e["scope"], e.get("repo_id"), e.get("level") or "INFO",
                 e["title"], e["message"]) for e in events]
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO telemetry_events (uid, ts, scope, repo_id, level, title, message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            await db.commit()

    async def get_telemetry_events(self, repo_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Last `limit` events, oldest first. General events always; repo events only for `repo_id`
        (None or the General placeholder = general events only)."""
        limit = max(0, int(limit))
        if limit == 0:
            return []
        cols = "uid, ts, scope, repo_id, level, title, message"
        if repo_id and repo_id != GENERAL_REPO_ID:
            sql = (f"SELECT {cols} FROM telemetry_events WHERE scope = 'general' OR (scope = 'repo' AND repo_id = ?) "
                   f"ORDER BY id DESC LIMIT ?")
            args: tuple = (repo_id, limit)
        else:
            sql = f"SELECT {cols} FROM telemetry_events WHERE scope = 'general' ORDER BY id DESC LIMIT ?"
            args = (limit,)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(sql, args) as cursor:
                rows = await cursor.fetchall()
        keys = ("uid", "ts", "scope", "repo_id", "level", "title", "message")
        return [dict(zip(keys, r)) for r in reversed(rows)]

    async def prune_telemetry_events(self, keep: int) -> int:
        """Keep only the newest `keep` rows. Returns how many were removed."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM telemetry_events WHERE id <= (SELECT COALESCE(MAX(id), 0) FROM telemetry_events) - ?",
                (max(1, int(keep)),))
            await db.commit()
            return cur.rowcount or 0

    async def get_file_chunks(self, repo_id: str, file_path: str) -> List[Dict[str, Any]]:
        norm_path = _normalize_path(file_path)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT file_path, chunk_index, content FROM code_chunks "
                "WHERE repo_id=? AND file_path=? ORDER BY chunk_index", (repo_id, norm_path)) as c:
                return [{"file_path": _normalize_path(r[0]), "chunk_index": r[1], "content": r[2]} for r in await c.fetchall()]