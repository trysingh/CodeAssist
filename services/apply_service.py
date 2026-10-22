"""Apply a revision to the repository on disk (explicit action, never automatic).

Rules
  * conflicts: every file the revision changes must still match the snapshot taken at
    ingest (revision 0). If someone edited it on disk meanwhile, nothing is written
    unless force=True.
  * all-or-nothing: files are written, and if any write fails the ones already
    replaced are restored from memory.
  * exact bytes: line endings are preserved because content is never re-encoded.
  * afterwards revision 0 is updated to the new disk content, so the file watcher's
    re-index sees "unchanged" and no stale-revision warning is raised.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

from services.repo_store import RepoStore, is_secret_path, safe_rel_path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _target(root: Path, rel: str) -> Path:
    t = (root / rel).resolve()
    if root != t and root not in t.parents:
        raise ValueError(f"path escapes repo root: {rel}")
    return t


async def apply_revision(store: RepoStore, repo_id: str, rev: int, root: Path, force: bool = False) -> dict:
    latest = await store.latest_rev(repo_id)
    if rev < 1 or rev > latest:
        raise ValueError(f"unknown revision {rev}")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"repo root not found on disk: {root}")

    base_idx = await store.index(repo_id, 0)
    view_idx = await store.index(repo_id, rev)
    todo = [p for p in sorted(set(base_idx) | set(view_idx)) if base_idx.get(p) != view_idx.get(p)]
    if not todo:
        return {"ok": True, "written": [], "deleted": [], "conflicts": [], "note": "disk already matches this revision"}

    refused: List[str] = []
    conflicts: List[dict] = []
    plan: Dict[str, Optional[bytes]] = {}
    before: Dict[str, Optional[bytes]] = {}
    for p in todo:
        rel = safe_rel_path(p)
        if is_secret_path(rel) or any(part.startswith(".") for part in rel.split("/")):
            refused.append(rel)
            continue
        t = _target(root, rel)
        disk = t.read_bytes() if t.is_file() else None
        before[rel] = disk
        if disk is None and rel in base_idx:
            conflicts.append({"path": rel, "reason": "file was deleted on disk since ingest"})
        elif disk is not None and rel not in base_idx:
            conflicts.append({"path": rel, "reason": "file exists on disk but is not tracked by the engine"})
        elif disk is not None and _sha(disk) != base_idx.get(rel):
            conflicts.append({"path": rel, "reason": "file changed on disk since ingest"})
        newer = await store.newer_revision_for(repo_id, rel, rev)
        if newer:
            conflicts.append({"path": rel, "reason": f"a newer revision ({newer}) changes this file; apply that one instead"})
        view = await store.get_file(repo_id, rel, rev)
        plan[rel] = view.data if view else None
    if refused:
        return {"ok": False, "written": [], "deleted": [], "conflicts": [],
                "refused": refused, "note": "refusing to write secret-like or hidden paths"}
    if conflicts and not force:
        return {"ok": False, "written": [], "deleted": [], "conflicts": conflicts,
                "note": "disk changed under this revision; review, or apply with force=true to overwrite"}

    done: List[str] = []
    try:
        for rel, data in plan.items():
            t = _target(root, rel)
            if data is None:
                if t.exists():
                    t.unlink()
            else:
                t.parent.mkdir(parents=True, exist_ok=True)
                tmp = t.with_name(t.name + ".coderag-tmp")
                tmp.write_bytes(data)
                os.replace(tmp, t)
            done.append(rel)
    except Exception as e:  # noqa: BLE001 — roll back what was already written
        for rel in done:
            t = _target(root, rel)
            old = before.get(rel)
            try:
                if old is None:
                    t.unlink(missing_ok=True)
                else:
                    t.write_bytes(old)
            except OSError:
                pass
        raise RuntimeError(f"apply failed and was rolled back: {type(e).__name__}: {e}") from e

    written = [r for r, d in plan.items() if d is not None]
    deleted = [r for r, d in plan.items() if d is None]
    if written:
        await store.upsert_base_files(repo_id, [(r, plan[r]) for r in written])
    if deleted:
        await store.delete_base(repo_id, deleted)
    await store.mark_applied(repo_id, rev)
    return {"ok": True, "written": written, "deleted": deleted, "conflicts": conflicts if force else []}
