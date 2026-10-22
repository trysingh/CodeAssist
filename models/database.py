from datetime import datetime
import aiosqlite
from typing import List, Dict, Optional, Any
from config.settings import settings
from models.schemas import GENERAL_REPO_ID, RepoInfo, JobStatus 

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
            await db.execute(
                "INSERT OR IGNORE INTO repositories (repo_id, title, root_path, status, total_files, created_at) "
                "VALUES (?, 'General (web)', '', 'COMPLETED', 0, ?)", (GENERAL_REPO_ID, datetime.utcnow().isoformat()))
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
        now_iso = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO repositories (repo_id, title, root_path, status, total_files, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    title=excluded.title,
                    root_path=excluded.root_path,
                    status=excluded.status,
                    total_files=excluded.total_files
            """, (repo_id, title, root_path, status.value, total_files, now_iso))
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
                created_val = row[7] if row[7] else datetime.utcnow().isoformat()
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
                    
                    created_val = row[7] if row[7] else datetime.utcnow().isoformat()
                    results.append(RepoInfo(
                        repo_id=row[0], title=row[1], root_path=row[2],
                        status=JobStatus(row[3]), total_files=row[4],
                        processed_files=processed, total_chunks=row[6],
                        created_at=str(created_val)
                    ))
                return results

    async def record_file_job(self, repo_id: str, file_path: str, file_hash: str, status: JobStatus, error: Optional[str] = None):
        now_iso = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                UPDATE file_jobs
                SET file_hash = ?, status = ?, error = ?, updated_at = ?
                WHERE repo_id = ? AND file_path = ?
            """, (file_hash, status.value, error, now_iso, repo_id, file_path))
            if cursor.rowcount == 0:
                await db.execute("""
                    INSERT INTO file_jobs (repo_id, file_path, file_hash, status, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (repo_id, file_path, file_hash, status.value, error, now_iso))
            await db.commit()

    async def get_completed_file_hashes(self, repo_id: str) -> Dict[str, str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT file_path, file_hash FROM file_jobs WHERE repo_id = ? AND status = 'COMPLETED'", (repo_id,)) as cursor:
                rows = await cursor.fetchall()
                return {row[0]: row[1] for row in rows}

    async def save_chunks_batch(self, repo_id: str, chunk_records: List[tuple]):
        async with aiosqlite.connect(self.db_path) as db:
            if chunk_records:
                file_paths = list({r[2] for r in chunk_records})
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
                """, chunk_records)

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
                        "file_path": r[1],
                        "chunk_index": r[2],
                        "structural_context": r[3],
                        "content": r[4]
                    }
                    for r in rows
                ]

    async def list_file_paths(self, repo_id: str) -> List[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT DISTINCT file_path FROM code_chunks WHERE repo_id=?", (repo_id,)) as c:
                return [r[0] for r in await c.fetchall()]

    async def get_file_chunks(self, repo_id: str, file_path: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT file_path, chunk_index, content FROM code_chunks "
                "WHERE repo_id=? AND file_path=? ORDER BY chunk_index", (repo_id, file_path)) as c:
                return [{"file_path": r[0], "chunk_index": r[1], "content": r[2]} for r in await c.fetchall()]