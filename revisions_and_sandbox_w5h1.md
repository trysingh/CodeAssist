# CodeRAG Engine: Revisions, Disk Sync & Sandbox (W5H1)

**Scope:** `repo_bundle.txt` (78 files). Every claim carries an evidence tag:

| Tag | Meaning |
|---|---|
| **[RAN]** | Reproduced by running the repo's own `RepoStore` / `apply_service` code against a temp folder |
| **[READ]** | Read directly in the code (file and function named) |
| **[INFER]** | My reasoning; not proven. Needs a real sandbox log to confirm |

---

## 0. TL;DR

1. **Three copies of your code exist, and they are refreshed by different things.**
   Disk (the real folder), the **raw store** (`repo_files`: rev 0 + revisions), and the **chunk index** (`code_chunks` + Chroma). Revisions live only in the raw store. They are never chunked or embedded.
2. **Nothing "refreshes" a revision.** Only rev 0 follows disk. A revision is a layer on top of rev 0; for each file the newest row wins. "Latest" means the highest revision number, **applied or not**. The next `/change` always builds on it.
3. **There is exactly one case where a revision is reconciled with disk:** a disk edit to a file that rev 0 already held *and* a pending revision also touches. The engine then appends a "disk sync" revision. Every other case relies on "newest row wins".
4. **"Applied" is a weak flag.** It is a timestamp on one revision, set only by the Apply button. It is not shown in the revision list or the UI.
5. **Six scenarios give wrong or confusing state** (section 4): cumulative apply (C), external apply (G), disk edit under a pending revision (H), new-file collision (J), ghost chunks after delete/move (K, L, M), and no undo after apply (N).
6. **On the sandbox: partly yes, but revisions are not the main cause.** Revision issues make the *base* wrong in multi-step sessions. The bigger causes are environment gaps: no network, files silently missing from the workspace, a single requirements file, and "not run" verdicts (section 7).
7. **Fix first:** secret-name filter (drops `token_service.py`), show applied state, smarter disk-sync, watcher delete/move + start after ingest, merge requirements files (section 8).

---

## 1. Mental model

### 1.1 The three stores

| Store | Tables / location | Holds | Refreshed by | Read by |
|---|---|---|---|---|
| **Disk** | your repo folder | the real files | you, `git`, the Apply button | watcher, startup scan |
| **Raw store** | `repo_files`, `repo_revisions` (SQLite, zlib bytes) | rev 0 = last known disk. rev 1..N = proposals, disk-syncs, reverts. Each revision stores **only the files it changed** | ingest, watcher, startup scan, Apply | `/change` pipeline, sandbox, apply, patch download |
| **Chunk index** | `code_chunks` (SQLite) + Chroma vectors + `file_jobs` (hash per file) | chunked text of **disk only** | ingest, watcher, startup scan | Q&A chat, semantic file locate |

### 1.2 Layering and the resolution rule [READ: `repo_store.py` header, `get_file`, `_index`]

```
rev 3   [ b.py ]                      <- proposal (changes only b.py)
rev 2   [ a.py ]                      <- proposal
rev 1   [ a.py, new.py ]              <- proposal
rev 0   [ a.py  b.py  c.py ]          <- last known disk
-----------------------------------------------------------------
view(R) = for each file, newest row with rev <= R  (deleted rows hide the file)
view(3) = a.py@2, b.py@3, c.py@0, new.py@1
```

- **Latest** = `MAX(rev)` in `repo_revisions`. No notion of "applied" is involved.
- **Revision kinds** (all share one linear chain): `edit` (a proposal), `disk-sync`, `revert`.
- **No branches.** `base_rev` must equal latest or the request is refused (`_run`: "Revision N is not the latest").

### 1.3 What "applied" means [READ: `apply_service.py`, `mark_applied`]

- Apply is **cumulative against rev 0**. It writes every file where `view(R) != rev 0`, not just R's own files.
- After writing, it **overwrites rev 0** with the new disk bytes (`upsert_base_files`, `INSERT OR REPLACE`). So rev 0 means "last known disk", **not** "original ingest snapshot".
- It then stamps `applied_at` into `report_json` of **that one revision only**.

---

## 2. W5H1: Revisions

| | |
|---|---|
| **What** | A numbered, append-only set of full-file replacements (`repo_files` rows) plus metadata (`repo_revisions`: parent, request, report JSON). A proposal is *not* a patch; it is the new bytes of each changed file. |
| **Why** | Chunks are lossy and tuned for search. Edits need exact bytes and line endings. Revisions let the engine propose, verify in a sandbox, show a diff, and write to disk **only on an explicit click**. |
| **Who** | Created by `RepoChangeService._persist` (edit), `RepoStore.record_disk_sync` (disk-sync), `RepoStore.revert_to` (revert). Marked by `apply_service` via `mark_applied`. Read by almost everything in the change pipeline. |
| **When** | A proposal is saved only after patch, static checks and sandbox finish. A disk-sync is saved when the watcher / startup scan sees a disk change to a file a pending revision also touches. A revert is saved on `POST /revisions/revert`. |
| **Where** | `.engine_storage/metadata_and_jobs.db`. `repo_files(repo_id, rev, file_path, sha256, content, deleted)`, `repo_revisions(repo_id, rev, parent, request, report_json, created_at)`. API: `/api/v1/repos/{id}/revisions[...]`. |
| **How** | `create_revision(parent, changes)` runs inside `BEGIN IMMEDIATE`. If `parent != latest` it raises `StaleRevision`. Otherwise it writes the changed files and one metadata row. |

### 2.1 How the next proposal picks its base [READ: `repo_change._run`, `_locate`, `_current`, `_patch`]

1. `base_rev = latest` (caller may pass a value, but it must equal latest).
2. **Locate** builds the file list from `raw store @ base_rev`. Files are found by name (raw store) and by semantic search (**Chroma = disk only**). Hits not in the raw-store view are dropped into `missing`.
3. **Patch / outline / contract** read file bytes from `raw store @ base_rev` (`_current`).
4. **Sandbox** tree = `snapshot_bytes(base_rev)` + `working`.
5. **Persist** calls `create_revision(parent=base_rev)`. If another revision landed meanwhile, it fails with `StaleRevision` *after* all the model and sandbox work was done.

So "the right instance" is decided by one rule, **newest row ≤ latest**, not by any per-case logic.

---

## 3. W5H1: Apply, watcher and refresh

### 3.1 Apply button

| | |
|---|---|
| **What** | Writes `view(R)` for every file that differs from rev 0 into the repo folder. All-or-nothing with rollback. Exact bytes. |
| **Why** | Keeps "write to my repo" an explicit human action. |
| **Who** | UI link `…/revisions/{rev}/apply` → `app.js` click handler → `POST` (confirmation phrase header) → `apply_service.apply_revision`. |
| **When** | Only on click. Never automatic. |
| **Where** | Disk, then rev 0 rows, then `report_json.applied_at` of that revision. |
| **How** | Conflict checks: disk differs from rev 0 (edited / deleted / untracked), or a **newer revision** touches the same file. Conflicts return 409; `force=true` overwrites. Secret-like and dot-paths are refused. |

### 3.2 Watcher, startup scan, ingest

| | |
|---|---|
| **What** | The only code that moves **disk changes** into the raw store and chunk index. |
| **Why** | So the next proposal and the Q&A see what you edited by hand. |
| **Who** | `RepoChangeHandler` (watchdog), `restore_all_watchers` (startup), `ingest_repository`, all calling `incremental_reingest_file` / `snapshot_raw_files`. |
| **When** | Watcher: 1.5 s after the **last** event for a file, on **created / modified** only. Startup: reconciles offline edits, then starts watchers. Ingest: full scan. |
| **Where** | rev 0 rows, `code_chunks`, Chroma, `file_jobs`. A disk-sync revision when a pending revision is shadowed. |
| **How** | `incremental_reingest_file`: (1) skip if chunk hash equals disk hash; (2) `upsert_base_files` (rev 0); (3) if shadowed, `record_disk_sync`; (4) re-chunk, delete old vectors, embed, save, record hash. |

**"Shadowed" means** [READ: `upsert_base_files`]: rev 0 already had the file, the new disk hash differs, **and** some revision > 0 also holds that path.

---

## 4. Scenario matrix

Verdict key: **OK** = behaves sensibly, **RISK** = surprising or incomplete, **BUG** = wrong result or silent loss.

| # | Situation | What the engine does | Next revision builds on | Verdict | Evidence |
|---|---|---|---|---|---|
| A | R1 proposed, **not applied** | R1 stored. Disk, rev 0, chunks unchanged | R1 (latest) | OK. But Q&A and semantic locate still see disk | [READ] |
| B | R1 → R2 → R3 proposed, none applied | Each stores only its own files | R3 | OK | [RAN S1] |
| C | **Apply R2 while R1 is unapplied** | Writes R1's and R2's changes (cumulative). Only R2 gets `applied_at` | R2 view (same bytes as disk) | **RISK**: R1 still shows "not applied" though its content is on disk | [RAN S2] |
| D | Apply older R1 after newer R2 touches the same file | 409 "a newer revision changes this file". `force` would **downgrade** the file and set rev 0 to the older bytes | n/a | OK; `force` is a footgun | block: [RAN S4], `tests/test_phase2.py:491`; downgrade via `force`: [READ] |
| E | Applied by button, **watcher running** | rev 0 already updated, so watcher sees "unchanged": no sync revision. Chunk hash differs, so chunks and vectors refresh within ~1.5 s per file | latest | OK | [READ], `test_phase2.py:417` |
| F | Applied by button, **watcher not running** | rev 0 is right (apply writes it). Chunks and Chroma stay old until the next restart. `start_watching` is called **only** from `restore_all_watchers`, never after `ingest` | latest | **RISK**: Q&A and locate see old code | [READ] |
| G | **Applied outside the button** (`git apply`, unzip, copy) | Watcher refreshes rev 0, flags shadowed, appends an extra disk-sync revision with the same bytes. R1 is never marked applied | the sync revision (same bytes as R1) | **RISK**: duplicate revision, wrong status; content still correct | [RAN S3] |
| H | **You edit a file a pending revision also changes** | Disk-sync revision holds your disk bytes. Proposal's change to *that file* becomes invisible to all later proposals; its other files stay. Apply R1 is blocked until `force` | latest = your file + proposal's other files | **BUG**: silent loss of part of the proposal, mixed state | [RAN S4] |
| I | You edit a file **no** revision touches | rev 0 refreshed; latest view picks it up | latest (sees your edit) | OK (up to ~1.5 s debounce window) | [RAN S5] |
| J | Proposal **adds** `new.py`; you create `new.py` yourself | Not flagged (shadow needs an existing rev-0 row). Apply passes with no conflict and **overwrites your file** | proposal's `new.py` | **BUG** | [RAN S6] |
| K | File **deleted on disk** | No `on_deleted` handler. rev 0 row and chunks stay. rev 0 pruned only by next startup / ingest scan; chunks never | ghost file still visible | **RISK**: proposals may edit a ghost; apply later says "deleted on disk" (`force` recreates it) | [READ] |
| L | File **renamed / moved** | No `on_moved`. New path appears only at next startup. Old path becomes a ghost (as K) | old path until restart | **RISK** | [READ] |
| M | **Apply deletes** a file | rev 0 row removed. Chunks and vectors stay | latest | **RISK**: Q&A cites deleted code; locate reports "not in raw store, re-run ingest" | [READ] |
| N | **Undo after apply** (`revert` to 0) | `revert_to(0)` restores rev 0, which is now the *applied* content. Original bytes are gone from the engine | n/a | **BUG** by design: no undo; also rewrites history (see 4.1) | [RAN S7, S2] |
| O | Two proposals started on the same base | Both run the full model + sandbox pipeline. The second fails at save: "repo changed while I was working" | n/a | RISK: wasted work | [RAN S8], [READ `_persist`] |
| P | Want an alternative to R1 from the same base | Not allowed (`base_rev` must be latest). Revert first, which appends a revision | n/a | OK (linear by design) | [READ] |
| Q | Restart or re-ingest with pending revisions | `snapshot_raw_files` runs the same shadow / disk-sync logic. `reset_repo_index` clears **chunks + file jobs only**; revisions survive | latest | OK | [READ] |

### 4.1 The four findings that matter most

**C. Cumulative apply hides R1.** Apply R2 and R1's file also lands on disk, but only R2 is stamped. Anyone reading the revision list sees R1 as pending [RAN S2: `applied_at` flags `{2: True, 1: False}`].

**H. Disk edit under a pending revision.** Example from the run: R1 proposes `a.py` and `b.py`. You hand-edit `a.py`. The sync revision makes the latest view `a.py` = your edit and `b.py` = proposal. If the two proposal files depended on each other, the mixed state may not run. The user is told via a WARNING notification only.

**J. New-file collision.** The shadow check requires an existing rev-0 row. A brand-new path has none, so your own file is treated as ordinary and Apply overwrites it with no conflict.

**N. Apply rewrites history.** After applying R2, R1's "before" bytes for `a.py` read `A1` instead of `A0` [RAN S2], because the "before" of R1 is rev 0, which Apply overwrote. Consequences: the default diff of older revisions is wrong, `revert_to(0)` cannot restore the pre-apply state, and baselines drift. Today only `git` can undo an applied change.

---

## 5. What each part actually sees ("split brain")

| Consumer | Reads from | Reflects |
|---|---|---|
| `/change` file list, named-file locate | raw store @ latest | disk **+ all revisions** |
| `/change` semantic locate | Chroma | **disk only** |
| Patch prompt, outline, signatures, contract | raw store @ latest | disk + all revisions |
| Sandbox workspace | raw store @ latest (+ working) | disk + all revisions, **minus skipped files** |
| Sandbox baseline ("pre-existing failures") | raw store @ `base_rev` = latest | includes unapplied revisions |
| fail→pass proof ("old code") | same | includes unapplied revisions |
| Q&A chat (named file full text, semantic) | `code_chunks` + Chroma | **disk only** |
| Apply conflict check | rev 0 vs disk | last known disk |
| `.patch` download (`from_disk=true`) | rev 0 → R | cumulative vs rev 0 |
| Diff (`from_disk=false`) | R−1 → R | wrong "before" for rev 1 once applied (N) |

**Practical effect:** after `/change` creates R1, asking chat "does endpoint X exist?" answers **no** (disk), while the next `/change` assumes **yes** (latest). Both are internally consistent, but they disagree with each other, and the user sees both.

---

## 6. W5H1: Runtime sandbox

| | |
|---|---|
| **What** | Runs proposed code in a Docker container: `--network none`, read-only root, tmpfs `/work`, 1 GB RAM, 2 CPUs, user 1000. Tests = repo's pytest files chosen by import graph, plus model-generated tests. If no tests exist: a smoke import of changed modules. |
| **Why** | To turn "looks right" into a verification level: L0 static, L1 import smoke, L2 repo tests, L3 generated tests that fail on old code and pass on new. |
| **Who** | `SandboxService` + `DockerRunner` (or `LocalRunner`, no isolation), `SandboxWarmer` (image pre-build), `RepoChangeService._sandbox`, a small decision model for failure triage. |
| **When** | Red checks (tests-first), per-task green checks, and the final run after patch + static checks. Image pre-warm at startup, after ingest, when a requirements file changes. |
| **Where** | Image tag = hash of the requirements text + Dockerfile. Workspace = in-memory tar of `snapshot_bytes(base_rev) + working`. Logs: `.engine_storage/sandbox_logs/<repo>/`. |
| **How** | `select_tests` (affected / all / limited / smoke) → run → if failures, rerun the **base tree** to subtract pre-existing failures → triage (code wrong / test outdated / environment) → fail→pass proof on old code. |

### 6.1 What the sandbox workspace does **not** contain [READ + RAN S9]

| Missing because | Examples |
|---|---|
| **Secret-name filter** (`*token*`, `*secret*`, `*credential*`, `*cookie*`, `*.key`, `*.crt`, `*.pem`, `.env*`, `*.patch`, `*.diff`) | **`tokenizer.py`, `auth/token_service.py`, `secrets_manager.py`, `credentials.py`, `cookie_utils.py`, `.env.example`** are all skipped [RAN S9]. `config.py`, `keys.py`, `api_key.py` are kept |
| Excluded dirs | `data/`, `build/`, `dist/`, `target/`, `workspaces/`, `venv/`, `node_modules/`, any dot-folder |
| Excluded patterns | `*.db`, `*.sqlite*`, `*.lock`, `*.zip`, `*.tar.gz`, `*.log`, `* copy.*` |
| Size / type | files > 512 KB, non-UTF-8 / binary (models, images, parquet, xlsx, pickles) |
| Runtime reality | network, env vars, running services (Ollama, SearXNG, DBs) |

The same exclusion lists serve both the chunk index and the raw store. So a file the sandbox lacks is also invisible to the patcher's outline.

---

## 7. Are these the reasons dynamic code quality is weak?

**Short answer:** the revision scenarios contribute, but they are **probably not the main cause**. They corrupt the *base* a proposal builds on. The sandbox gaps decide whether a proposal can be *verified at all*. I can't rank them for certain without a failing sandbox log (see section 9).

Ranked by my judgement of impact on "dynamic" code (code that touches HTTP, files, env, DB, LLMs):

| # | Cause | Why it hurts | Evidence |
|---|---|---|---|
| 1 | **No network, no env, no running services in the run** | Any test that calls an API, Ollama, SearXNG, a DB or reads env fails. Network errors are recognised and hinted, but the proposal is then "unverified" or blamed on environment, not fixed | [READ `docker_args`, `_NET_ERR`] |
| 2 | **Files silently missing from the workspace** | Imports of `token_*`, `*secret*`, `credentials` modules fail with `ModuleNotFoundError`; data fixtures and `.env.example` are gone | [RAN S9], [READ] |
| 3 | **Single requirements file** | `_requirements` returns the **first** of `requirements-test.txt`, `requirements-dev.txt`, `requirements.txt`. If `requirements-test.txt` has only `pytest` or contains `-r requirements.txt`, the build gets one file and `-r` has nothing to read. No `pyproject.toml` support. Image is Linux `python:3.12-slim`; your machine is Windows | [READ `_build`, `_requirements`], [INFER for your repos] |
| 4 | **"Not run" verdicts** | No test names the change and suite > 12 files → not run. Timeouts 180 s (60 s per task). No tests → import-only. Jinja templates and JS are **never executed** (Python only) | [READ `select_tests`, `verify`, settings] |
| 5 | **Verification sits on top of unapplied revisions** | Baseline and fail→pass use `latest`, so breakage from R1 is called "pre-existing" for R2. A green result may not hold on disk | [READ `verify`, `_baseline`] |
| 6 | **State bugs H, J, C, G** | Next proposal builds on a mixed or duplicated base | [RAN] |
| 7 | **Prompt budget** | 6 files, 60 000 chars per prompt, whole files. A large module can fail with "do not fit the context budget" | [READ `_patch`, settings] |

---

## 8. Recommendations

Ordered by value per effort. Snippets show only the changed lines.

| P | Change | Fixes | Effort |
|---|---|---|---|
| **P0** | Narrow the secret filter (below) | 2 | S |
| **P0** | Expose applied state in `list_revisions` + a badge | C, G | S |
| **P0** | Smarter shadow check (below) | G, H, J | S |
| **P1** | Back up "before" bytes on Apply, add undo | N | M |
| **P1** | Watcher: `on_deleted`, `on_moved`, start after ingest, purge chunks | F, K, L, M | M |
| **P1** | Merge requirements files; strip `-r` / `-c` lines | 3 | S |
| **P1** | Chat notice when revisions are pending | A, section 5 | S |
| **P2** | Separate exclusion lists: chunking vs raw store vs sandbox | 2 | M |
| **P2** | Sandbox: dummy env file, mock-first test-author prompt for network code | 1 | M |
| **P2** | Show "verified on top of unapplied revisions: 1, 2" in the verdict | 5 | S |

### 8.1 Narrow the secret filter (`repo_store.is_secret_path`) [P0]

Only treat name patterns as secrets for non-source files. Trade-off: a `.py` file could hard-code a key, and `gemma4:31b-cloud` is the default model, so content may leave your machine. Add a cheap content scan (see below) if that matters.

```python
_SRC = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".html", ".css", ".md"}

def is_secret_path(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name in (".env.example", ".env.sample", ".env.template"):
        return False
    if PurePosixPath(name).suffix in _SRC:          # source files: judge by content, not by name
        return False
    return any(fnmatch.fnmatch(name, p.lower()) for p in settings.raw_store_exclude_patterns)
```

### 8.2 Smarter shadow check (`repo_store.upsert_base_files`) [P0]

Replace the `if row:` block. It (a) drops the "rev 0 must exist" requirement (fixes J) and (b) separates "disk equals the newest revision" (applied outside the button) from a real conflict. Verified on five cases [RAN: exp2].

```python
async with db.execute(
    "SELECT rev, sha256 FROM repo_files WHERE repo_id=? AND file_path=? AND rev>0 "
    "ORDER BY rev DESC LIMIT 1", (repo_id, path)) as cur:
    newest = await cur.fetchone()
if newest:
    if newest[1] == digest:
        out.setdefault("same_as_revision", []).append((path, newest[0]))   # applied outside the button
    else:
        out["shadowed"].append(path)                                       # real conflict
```

Then, in `snapshot_raw_files` and `incremental_reingest_file`:
- for each `same_as_revision` → call `mark_applied(repo_id, rev)`; **do not** create a disk-sync revision (removes duplicate G).
- for `shadowed` → keep the sync revision, but put the proposal's affected file list in the notification, so H is visible.

### 8.3 Show applied state [P0]

In `list_revisions`, also select `r.report_json`, then return:

```python
rep = json.loads(r[5] or "{}")
{"rev": r[0], ..., "kind": rep.get("kind", "edit"), "applied": bool(rep.get("applied_at"))}
```

In `mark_applied`, also stamp earlier unapplied `edit` revisions whose files are now on disk: `report["applied_via"] = rev`. This fixes C.

### 8.4 Keep the pre-apply bytes (undo) [P1]

`apply_revision` already holds them in `before`. Persist them before `upsert_base_files`:

```python
# new table: repo_apply_backup(repo_id, rev, file_path, content BLOB)   -- NULL content = file did not exist
await store.save_backup(repo_id, rev, before)      # before: {path: bytes|None}
```

An `undo_apply(rev)` then writes them back and refreshes rev 0. Also fixes the wrong "before" in section 4.1 N if you diff against the backup.

### 8.5 Watcher gaps [P1]

```python
# watcher_service.RepoChangeHandler
def on_deleted(self, e):
    if not e.is_directory: self._handle_file_change(e.src_path)
def on_moved(self, e):
    if not e.is_directory:
        self._handle_file_change(e.src_path); self._handle_file_change(e.dest_path)
```

- `incremental_reingest_file`: when the file no longer exists, delete its Chroma vectors (`delete_chunks_by_file`), its `code_chunks` and `file_jobs` rows, and its rev-0 row (`delete_base`), instead of returning early. Needs one small `db.delete_file_chunks` helper.
- `ingest_repository`: call `watcher_manager.start_watching(repo_id, str(root_path))` at the end (fixes F).
- `is_excluded` is called on a path that no longer exists, which is fine because it only uses path parts.

### 8.6 Merge requirements files [P1]

```python
@staticmethod
def _requirements(tree):
    seen, out = set(), []
    for name in settings.sandbox_requirements_files:
        if name in tree:
            for ln in tree[name].decode("utf-8", "replace").splitlines():
                s = ln.strip()
                if s and not s.startswith(("-r", "-c", "--requirement")) and s not in seen:
                    seen.add(s); out.append(ln)
    return "\n".join(out)
```

`requirements_for` (the pre-warm path) must use the same logic, or the warm image tag and the verify tag will differ. This is an existing invariant, covered by `tests/test_sandbox_prewarm.py:251`.

### 8.7 Chat notice for pending revisions [P1]

In `_route_chat`, for the Q&A branch: if `latest_rev > 0`, prepend one line: "Answers reflect files on disk. Unapplied revision(s) N touch: a.py, b.py." This removes the confusing "no / yes" disagreement from section 5 without touching the chunk index.

### 8.8 Quick diagnostic: what never reaches the sandbox

Run from the engine folder: `python skipped_files.py <your_repo_root>`

```python
import sys, os
from pathlib import Path
sys.path.insert(0, ".")
from config.settings import settings
from services.repo_store import is_secret_path, is_text_bytes

root = Path(sys.argv[1]).resolve()
for dp, _, fn in os.walk(root):
    rel_dir = Path(dp).relative_to(root).parts
    for name in fn:
        p = Path(dp) / name; rel = p.relative_to(root).as_posix(); why = None
        if any(x in settings.exclude_dirs or x.startswith(".") for x in (*rel_dir, name)): why = "hidden / excluded dir"
        elif any(p.match(pat) for pat in settings.exclude_patterns): why = "exclude pattern"
        elif is_secret_path(rel): why = "secret-like name"
        elif p.stat().st_size > settings.raw_store_max_file_kb * 1024: why = "too large"
        elif not is_text_bytes(p.read_bytes()[:8192]): why = "not UTF-8 text"
        if why: print(f"{why:24} {rel}")
```

Tested on a demo folder: it flagged `.env.example`, `auth/token_service.py` and `data/f.csv`.

---

## 9. What I could not verify, and what I need

| Gap | Why it matters |
|---|---|
| **No real sandbox logs in the bundle** (`.engine_storage/sandbox_logs/257142fce140` is an empty directory entry) | I can't tell which cause in section 7 dominates your failures |
| Docker / Windows behaviour | The sandbox needs Docker Desktop and a Linux image; I ran only the store and apply logic, not the sandbox |
| UI behaviour | Read `app.js` only. One visible quirk: the success label reads `res.body.count`, which the API doesn't return (it returns `written` / `deleted`), so it likely shows "Applied 0 files" |
| Tests | I read names and the apply / disk-edit tests. I found no tests for scenarios G, J, K, L, M or N |

**Most useful next step:** share the latest failing log from `.engine_storage/sandbox_logs/<repo_id>/`, plus the output of `skipped_files.py` for your target repo. Then I can map each failure to the signatures below.

| Signature in the log | Likely cause |
|---|---|
| `ModuleNotFoundError` for a **local** module | File skipped by the secret filter or an excluded dir (section 6.1) |
| `ModuleNotFoundError` for a **package** | Requirements file choice or `-r` line (section 7, #3) |
| `NameResolutionError`, `Connection refused` | `--network none` (section 7, #1) |
| `KeyError` / settings validation error at import | No env vars; `.env.example` skipped |
| `FileNotFoundError` for fixtures | `data/` or binary files excluded |
| "no test produced a result", "limited" | Not-run path (section 7, #4) |
| Passes in sandbox, fails on your machine | Base includes unapplied revisions (#5), or Linux vs Windows / Python version |

---

## Appendix A. Code map

| Concern | File → function |
|---|---|
| Layering and resolution | `services/repo_store.py` → `get_file`, `_index`, `snapshot_bytes` |
| Create revision, stale guard | `repo_store.create_revision` |
| Disk sync | `repo_store.upsert_base_files` (shadowed), `record_disk_sync` |
| Revert | `repo_store.revert_to` |
| Apply | `services/apply_service.py` → `apply_revision`, `disk_conflicts` |
| Applied flag | `repo_store.mark_applied` (`report_json.applied_at`) |
| Watcher | `services/watcher_service.py` → `RepoChangeHandler`, `RepoWatcherManager` |
| Refresh paths | `controllers/engine_controller.py` → `incremental_reingest_file`, `snapshot_raw_files`, `restore_all_watchers`, `ingest_repository` |
| Next-proposal base | `services/repo_change.py` → `_run`, `_locate`, `_current`, `_patch`, `_persist` |
| Sandbox | `services/sandbox_service.py` → `verify`, `select_tests`, `DockerRunner.docker_args`, `_requirements`, `_baseline` |
| Exclusions | `config/settings.py` → `raw_store_exclude_patterns`, `exclude_dirs`, `exclude_patterns` |
| Revision endpoints | `views/api_view.py` → `/revisions`, `/diff`, `/patch`, `/download`, `/apply`, `/revert` |

## Appendix B. How the [RAN] results were produced

A throwaway script created a temp SQLite store and a temp repo folder, then called `RepoStore.upsert_base_files`, `create_revision`, `record_disk_sync`, `revert_to`, `changed_files` and `apply_service.apply_revision` directly. Scenario labels S1–S9 map to the matrix as: S1→B, S2→C/N, S3→G, S4→D/H, S5→I, S6→J, S7→N, S8→O, S9→section 6.1. Nothing in your repo was modified.
