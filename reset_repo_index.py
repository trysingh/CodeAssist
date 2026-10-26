# reset_repo_index.py
# python reset_repo_index.py list
# python reset_repo_index.py <repo_id>
import sqlite3, sys, chromadb

db = sqlite3.connect(".engine_storage/metadata_and_jobs.db")
repo_id = sys.argv[1]  # pass "list" to see repo ids

if repo_id == "list":
    print(db.execute("SELECT repo_id, title FROM repositories").fetchall())
    sys.exit()

client = chromadb.PersistentClient(path=".engine_storage/chroma_vectors")
try:
    client.delete_collection(f"repo_{repo_id}")
except Exception as e:
    print("chroma:", e)

db.execute("DELETE FROM code_chunks WHERE repo_id=?", (repo_id,))
db.execute("DELETE FROM file_jobs WHERE repo_id=?", (repo_id,))
db.commit()
print("cleared", repo_id)