
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
| `LLM_FALLBACK_MODEL` | *(empty)* | a LOCAL chat model (see `ollama list`) used when the chosen model is refused for credits / usage (Ollama Cloud HTTP 402). Empty = no fallback: the chat shows the reason |
| `LLM_QUOTA_COOLDOWN_S` | `300` | after such a refusal, calls for that model go straight to the fallback for this many seconds (`0` = try the cloud model every time) |

Existing repos ingested before v2 keep working: their chunks count as a ready vector index, so they stay on vector/hybrid retrieval.


## v3: skill pack & lanes
Design: `docs/DESIGN_v3.md`. Changes: `CHANGELOG.md`.

Each change / plan request is put in a **lane** by the intake step: **Fast** (presentational CSS, copy, docs, one-file renames, disposable scripts),
**Standard** (the default) or **Strict** (High risk, auth, payments, migrations, IaC / CI-CD, front-end security triggers such as `innerHTML`,
`eval`, tokens in `localStorage`, cookies, CSP, third-party scripts). The highest trigger wins and an unsure case takes the higher lane. A lane only
raises the existing change gate / review / tests-first for that request, never lowers your settings. The lane, the recommended validation level
and a verification label (Verified - executed in sandbox / container, Statically checked, Unverified (user must run)) appear in the Delivery notes;
a Fast lane gets a few lines.

The **skill pack** (`skills/risk-proportional-code-dev/`) is the written playbook the lane selects from: `SKILL.md` is always added to the
change / plan prompts, the files in `references/` only when the lane needs them (validation ladder, regulated domains, delivery and review).
Ask-only questions get none of it. To change the rules, edit the markdown (picked up on the next request, no restart). To swap the pack, point
`ENGINE_SKILL_PACK_DIR` at another folder with a `SKILL.md` (frontmatter `name:` and `description:`) and the same four `references/*.md` file names.
If the folder is missing or broken the request runs as in v2 and one line is logged.

Force a lane for one request by writing `lane: fast` (or `lane: standard`, `use the strict lane`) in it, or for all requests with `ENGINE_LANE_OVERRIDE`.
An override can raise a lane but cannot take a request below Strict while a Strict trigger is present.

| Setting (env prefix `ENGINE_`) | Default | Meaning |
|---|---|---|
| `SKILL_PACK_ENABLED` | `true` | `false` = no skill text in prompts (lanes still raise the baseline policy and show in the notes) |
| `SKILL_PACK_DIR` | `skills/risk-proportional-code-dev` | pack folder; a relative path is resolved against the project root |
| `SKILL_PACK_MAX_CHARS` | `12000` | cap on the pack text added to a prompt (about 3k tokens; the full Strict set is about 10k characters) |
| `LANE_OVERRIDE` | empty | `fast` / `standard` / `strict` for every request; empty = automatic |

**v3.1 notes.** A request that mixes types (for example colours plus JavaScript audio) takes the highest lane of the types it touches. The Delivery notes end
with a validation run: `Recommended: L1, L3 · Achieved: L1 (boot smoke)` and one line per recommended level saying whether it ran or was skipped and why.
"Verified - executed in container: N sandbox tests passed" appears only when N tests really ran; a boot check alone is labelled as such. A message that starts
with an unknown `/word` (for example `/apply`) gets an "Unknown command" reply with the command list instead of being answered as a question; paths such as
`/home/x fails` are not commands. Use `/change` to propose a revision.

**v3.2 notes.** A `/change` built on unapplied revisions says so ("Builds on unapplied revision(s) 22 ...") and gives the cumulative stats and the cumulative `.patch`. Asking
again for what the latest unapplied revision already implements (request >= 90% similar) returns that revision with its apply links and makes no model call; add "redo",
"again", "regenerate" or "from scratch" to rebuild, or set `ENGINE_REPEAT_REQUEST_GUARD=false`. A short **Summary** (changed / apply / verified / next) heads each saved change.



## Runners (inference backends)
Chat and embeddings go through `services/runners/`; the backend is a config choice. The default, `ollama`, is the behaviour the app always had.

| Setting (env, `ENGINE_` prefix optional) | Default | Meaning |
|---|---|---|
| `LLM_RUNNER` | `ollama` | `ollama` or `cloudflare` (Cloudflare Workers AI, OpenAI-compatible endpoint, needs `pip install langchain-openai`) |
| `EMBEDDING_RUNNER` | `ollama` | separate from `LLM_RUNNER`: vectors of different models are not comparable, so switching it needs a re-index (`python reset_repo_index.py`) |
| `CF_LIST_ALL_MODELS` | `true` | Cloudflare runner: the model picker lists all Workers AI text-generation models on the account (`CF_MODELS` first, cached `CF_MODELS_CACHE_S`=600 s); `false` = only `CF_MODELS` |
| `CF_ACCOUNT_ID`, `CF_API_TOKEN` | *(empty)* | required for `cloudflare`; the app refuses to start without them |
| `CF_LLM_MODEL` / `CF_VISION_MODEL` | `@cf/google/gemma-4-26b-a4b-it` | Ollama-style names (contain `:`, no `@`, e.g. `gemma4:31b-cloud`) are mapped to these when `LLM_RUNNER=cloudflare` |
| `CF_EMBEDDING_MODEL` | `@cf/baai/bge-base-en-v1.5` | embedding model of the cloudflare embedding runner |
| `CF_MODELS` | gemma-4-26b-a4b-it, gemma-3-12b-it | comma list offered in the UI model picker |
| `CF_AI_GATEWAY_URL` | *(empty)* | optional base URL override (AI Gateway) |

Not supported on `cloudflare`: `num_ctx` (ignored), `ask_reasoning` (ignored). The decision model (`/v1/systemone`) stays Ollama-only.
Setup and troubleshooting: `docs/CLOUDFLARE_SETUP.md`. New backend: implement `services/runners/base.Runner` and call `register_runner(name, factory)`.
