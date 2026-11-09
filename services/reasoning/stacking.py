"""Stacked revisions, repeated requests and the plain-language change summary (v3.2). stdlib + pydantic only at import time;
the store is passed in (duck-typed: list_revisions / get_revision / index / get_file / changed_files) and the diff helpers are
imported lazily, so the pure parts are testable without aiosqlite.

A revision is "unapplied" the way the rest of the app means it: revision 0 is the disk as the engine last saw it, and
`report["applied_at"]` is set by RepoStore.mark_applied. Revisions pile up on each other, so the cumulative change against
the disk (what the .patch endpoint serves) is what a user has to apply, not the last revision's delta."""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional

from .crosscheck import _keywords

_REDO_RE = re.compile(r"\b(?:redo|again|regenerate|from scratch)\b", re.IGNORECASE)
_MAX_WALK = 25                    # revisions inspected when looking for unapplied ones


# ------------------------------------------------------------------ repeated request
def normalize_request(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/.-]", " ", (text or "").lower())).strip()


def request_similarity(a: str, b: str) -> float:
    na, nb = normalize_request(a), normalize_request(b)
    return difflib.SequenceMatcher(None, na, nb).ratio() if na and nb else 0.0


def wants_redo(text: str) -> bool:
    return bool(_REDO_RE.search(text or ""))


def is_repeat_request(new: str, old: str, threshold: float = 0.9) -> bool:
    """Same request, or nearly (normalised text, difflib ratio >= threshold). Override words are handled by the caller."""
    return request_similarity(new, old) >= threshold


def _fmt(f: dict) -> str:
    return f"`{f['path']}` ({'new file, ' if f.get('new') else ''}+{f.get('added', 0)} −{f.get('removed', 0)})"


def _stats_of(path: str, before: Optional[bytes], after: Optional[bytes]) -> dict:
    from services.patch_service import diff_stats, make_diff      # lazy: patch_service imports the sqlite-backed store
    a, r = diff_stats(make_diff(path, before, after))
    return {"path": path, "added": a, "removed": r, "new": before is None}


async def unapplied_edit_revisions(store: Any, repo_id: str, upto: int) -> List[int]:
    """Edit revisions <= upto that are not applied to disk, oldest first. An applied revision contains everything before it,
    so the walk (newest first) stops there; disk-sync and revert revisions are skipped."""
    out: List[int] = []
    for r in (await store.list_revisions(repo_id))[:_MAX_WALK]:
        if r["rev"] > upto:
            continue
        rep = ((await store.get_revision(repo_id, r["rev"])) or {}).get("report") or {}
        if rep.get("kind") != "edit":
            continue
        if rep.get("applied_at"):
            break
        out.append(r["rev"])
    return sorted(out)


async def cumulative_stats(store: Any, repo_id: str, rev: int) -> List[dict]:
    """Per-file +/- of `rev` against the disk snapshot (rev 0): the same comparison RepoStore-based /patch?from_disk makes."""
    base, view = await store.index(repo_id, 0), await store.index(repo_id, rev)
    out = []
    for p in sorted(set(base) | set(view)):
        if base.get(p) == view.get(p):
            continue
        b, a = await store.get_file(repo_id, p, 0), await store.get_file(repo_id, p, rev)
        out.append(_stats_of(p, b.data if b else None, a.data if a else None))
    return out


async def stack_info(store: Any, repo_id: str, base_rev: int, rev: int) -> tuple:
    """([unapplied revisions this one builds on], [cumulative per-file stats against disk]); ([], []) when it builds on nothing unapplied."""
    if base_rev <= 0:
        return [], []
    chain = [x for x in await unapplied_edit_revisions(store, repo_id, base_rev) if x < rev]
    if not chain:
        return [], []
    cum = await cumulative_stats(store, repo_id, rev)
    return (chain, cum) if cum else ([], [])


def stack_note(stacked_on: List[int], rev: Optional[int], cumulative: List[dict]) -> str:
    if not stacked_on:
        return ""
    revs = ", ".join(str(x) for x in stacked_on)
    together = " and ".join([", ".join(str(x) for x in stacked_on)] + ([str(rev)] if rev else []))
    cum = "; ".join(_fmt(f) for f in cumulative[:8]) + (f" (+{len(cumulative) - 8} more)" if len(cumulative) > 8 else "")
    return (f"**Builds on unapplied revision(s) {revs}.** Shown below: this revision's changes only. "
            f"The cumulative change against your disk: {cum or 'n/a'} — use Download .patch (cumulative) or apply {together} together.")


# ------------------------------------------------------------------ repeat guard
async def find_repeat(store: Any, repo_id: str, query: str, threshold: float = 0.9) -> Optional[dict]:
    """The latest edit revision when it is not applied and was made for the same (or near-identical) request, else None.
    Override words (redo, again, regenerate, from scratch) and a revision without a stored request give None."""
    if not (query or "").strip() or wants_redo(query):
        return None
    for r in (await store.list_revisions(repo_id))[:_MAX_WALK]:
        rep = ((await store.get_revision(repo_id, r["rev"])) or {}).get("report") or {}
        if rep.get("kind") != "edit":
            continue                                    # disk sync / revert
        old = (r.get("request") or "").strip()
        if rep.get("applied_at") or not old or not is_repeat_request(query, old, threshold):
            return None
        files = []
        for p, (b, a) in sorted((await store.changed_files(repo_id, r["rev"])).items()):
            files.append(_stats_of(p, b, a))
        return {"rev": r["rev"], "request": old, "files": files}
    return None


def repeat_reply(hit: dict, repo_id: str) -> str:
    base = f"/api/v1/repos/{repo_id}/revisions/{hit['rev']}"
    files = "; ".join(_fmt(f) for f in hit["files"][:8]) or "no file list stored"
    return (f"This is the same request as revision {hit['rev']} (not applied yet), which already implements it: {files}. "
            f"Apply it, or tell me what to change. Say 'redo' to regenerate from scratch.\n\n"
            f"[Apply to repo]({base}/apply) · [Download changed files]({base}/download) · "
            f"[Full diff]({base}/diff) · [Download .patch]({base}/patch) "
            f"(cumulative against the repo on disk; `git apply --check` it first)")


# ------------------------------------------------------------------ plain summary (deterministic, no model call)
_PLAN_FILE_RE = re.compile(r"^\s*[-*]\s*`?([\w./\\-]+\.\w+)`?\s*(?:[-:–—]\s*(.*))?$")


def plan_file_lines(plan: str) -> Dict[str, str]:
    """{path: what} from the plan's FILES and NEW FILES sections."""
    out: Dict[str, str] = {}
    section = False
    for ln in (plan or "").splitlines():
        head = ln.strip().upper()
        if head.startswith(("FILES", "NEW FILES")):
            section = True
            continue
        if section and head and not head.startswith(("-", "*")):
            section = False
        m = _PLAN_FILE_RE.match(ln) if section else None
        if m:
            out[m.group(1)] = (m.group(2) or "").strip()
    return out


def _short(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def change_summary(brief: Any, facts: dict, notes: Any) -> str:
    """"**Summary**" block (header + 4 lines) for a saved change; "" for any other turn. Built from the plan's FILES lines, the success
    criteria, the facts and the validation summary."""
    rev = facts.get("revision")
    if not rev or getattr(brief, "intent", "") != "change":
        return ""
    stats = {f["path"]: f for f in (facts.get("change_files") or []) if isinstance(f, dict) and f.get("path")}
    plan = plan_file_lines(str(facts.get("plan") or ""))
    paths = list(stats) or list(plan)
    crits = [c[6:] for c in getattr(brief, "success_criteria", []) if c.startswith("Done: ")] or [getattr(brief, "goal", "") or "the request"]
    parts = []
    for c in crits[:3]:
        kws = set(_keywords(c))
        hit = [p for p in paths if kws & set(_keywords(plan.get(p, "")))] or paths
        where = ", ".join(f"`{p}`" + (f" (+{stats[p]['added']} −{stats[p]['removed']})" if p in stats else "") for p in hit[:3])
        parts.append(f"{_short(c, 70)} → {where or 'n/a'}")
    changed = "; ".join(parts) + (f"; +{len(crits) - 3} more" if len(crits) > 3 else "")
    stacked = [int(x) for x in (facts.get("stacked_on") or [])]
    if stacked:
        together = " and ".join([", ".join(map(str, stacked)), str(rev)])
        apply_ = f"revision {rev} builds on unapplied {', '.join(map(str, stacked))}: use the cumulative .patch or apply {together} together."
    else:
        apply_ = f"use the Apply to repo link below (revision {rev}), or the .patch."
    labels = list(getattr(notes, "verification", []) or [])
    verified = (labels[0] if labels else "Unverified (user must run)") + (
        f" — {notes.validation_summary}" if getattr(notes, "validation_summary", "") else "")
    steps = list(getattr(notes, "validation_steps", []) or [])
    nxt = []
    if not facts.get("tests_ran"):
        nxt.append("add a test that fails before and passes after")
    l3 = [s for s in steps if s.startswith("L3") and "skipped" in s]
    if l3:
        nxt.append("check the UI in a browser (L3 isn't available in this version)")
    elif any(s.startswith("L4") for s in steps):
        nxt.append("do the live check yourself (never share credentials in chat)")
    next_ = "; ".join(nxt) if nxt else "review the diff, then apply"
    return "\n".join(["**Summary**", f"- Changed: {changed}", f"- Apply: {apply_}", f"- Verified: {_short(verified, 230)}",
                      f"- Next: {next_}"]) + "\n\n"
