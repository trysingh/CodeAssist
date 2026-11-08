
## Virtual Env to be used
- Use aCodeAssist


## Core Commands
```
python engine.py serve

python engine.py ingest --rep0 "C:\Users\IN010M80866\Apps\Code_Assistant"      # Windows
python engine.py ingest --rep0 ~/apps/Code_Assistant                                # macOS / Linux

python engine.py ingest --repo ~/apps/Code_Assistant --vectorize                 # also build the vector index now and wait for it
python engine.py ingest jobs [repo-id]                                             # vectorize jobs and their progress

python engine.py ingest list

python engine.py ingest chat <repo-id>
    python engine.py ingest chat 91cfd341a2ea
python engine.py ingest chat <repo-id> --trace                                     # print the step trace after every answer

python engine.py serve --host 127.0.0.1 --port 8000

```

## macOS / Linux notes
- Create the venv with `python3 -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt`.
- Sandbox needs Docker (Docker Desktop on macOS; on Linux add your user to the `docker` group: `sudo usermod -aG docker $USER`).
- Linux file watching uses inotify; for big repos raise the limit: `sudo sysctl fs.inotify.max_user_watches=524288`.
- Run the app from a terminal. With no terminal attached (service, container) the startup watcher prompt is skipped and ALL repos are watched.


## v2: agentic search, background vectors, reasoning loop, Activity log
Design and contracts: `docs/DESIGN_v2.md`. Changes: `CHANGELOG.md`.

**Agentic is the default.** Ingest scans the repo and stores a verbatim snapshot (no embeddings), so a repo is searchable in seconds
("Repo ready (agentic search) in Xs"). Questions are answered by grep / glob / read / symbol search over that snapshot.
When the agentic result is weak (conceptual or "find similar code" questions, confidence below
`ENGINE_VECTORIZE_CONFIDENCE_THRESHOLD`), the answer says so and a **background vectorize job** is queued
(`ENGINE_VECTORIZE_FALLBACK=auto`). You are notified at start, on progress and when "Vector index ready" arrives; from then on
retrieval is hybrid (agentic blocks + semantic hits). A job interrupted by a restart is marked FAILED ("interrupted by restart")
and resumes at the last finished file the next time it is queued.

**Reasoning loop.** Each chat turn goes through intake (intent, risk, ambiguity, success criteria) -> baseline policy (higher risk
raises the change gate / review / tests-first, never lowers your settings) -> the existing ask/plan/change path -> cross-check ->
"Delivery notes" appended to the answer (assumptions, limitations, what was and was not verified). A high-risk, very ambiguous
change request gets clarifying questions instead of an edit. If the reasoning layer itself fails, you still get the plain answer.

**Activity log.** Every agent step, tool call, LLM call, sandbox run and ingest/vectorize step is an event grouped by `request_id`
(UI: Activity tab / "Show steps"; CLI: `ingest chat --trace`).

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ingest?repo_root=` | `{message, repo_root, repo_id}`; fast agentic ingest in the background |
| `POST /api/v1/repos/{id}/vectorize` | enqueue the vector index (optional body `{"reason": "..."}`) -> Job |
| `GET /api/v1/repos/{id}/search-mode` | `{ingest_mode, vector_status, retrieval_mode, job}` |
| `GET /api/v1/jobs?repo_id=&limit=`, `GET /api/v1/jobs/{job_id}` | vectorize jobs (`{"jobs": [...]}` / one Job) |
| `GET /api/v1/actions?request_id=&repo_id=&limit=` | `{"events": [...]}` (memory first, stored telemetry after a restart) |
| `POST /api/v1/chat`, `/chat/attach` | responses now include `request_id` |
| `GET /api/v1/repos` | repo list also carries `vector_status` and `ingest_mode` |

SSE notifications gain `kind` (`log`, `action`, `job`), `request_id`, `thread_id` and `data` (the ActionEvent / Job dict).

| Setting (env prefix `ENGINE_`) | Default | Meaning |
|---|---|---|
| `INGEST_MODE` | `agentic` | `vector` = also embed right after ingest (and wait for it) |
| `RETRIEVAL_MODE` | `agentic` | `agentic` / `vector` / `hybrid`; agentic upgrades itself to hybrid once vectors are ready; `vector` = legacy path |
| `VECTORIZE_FALLBACK` | `auto` | `auto` = a weak agentic result queues a background job; `off` = never |
| `VECTORIZE_CONFIDENCE_THRESHOLD` | `0.45` | agentic confidence below this counts as weak |
| `REASONING_SUPERVISOR` | `true` | intake / baseline / cross-check / delivery notes around every turn |
| `REASONING_INTAKE_LLM` | `false` | let the chat model refine the intake brief (one extra call) |
| `ACTION_LOG_ENABLED` | `true` | stream every step to the Activity log (SSE + telemetry table) |
| `WATCH_ON_INGEST` | `true` | start the file watcher as soon as a repo is ingested |

Existing repos ingested before v2 keep working: their chunks count as a ready vector index, so they stay on vector/hybrid retrieval.
