# Changelog

## v2 (unreleased)
### Added
- Agentic search (`services/agentic_search.py`): grep / glob / read / symbol search over the raw snapshot with a confidence score.
- Background vectorize jobs (`services/vectorize_jobs.py`, `jobs` table): queued automatically when agentic search is weak
  (`ENGINE_VECTORIZE_FALLBACK=auto`) or on demand (`POST /api/v1/repos/{id}/vectorize`, `ingest --vectorize`); progress and completion
  notifications; stale jobs are closed as FAILED ("interrupted by restart") at startup.
- Reasoning supervisor (`services/reasoning/`): intake, risk-proportional baseline policy, cross-check and "Delivery notes" on every
  chat answer. Failures of the reasoning layer fall back to the plain answer.
- Action log (`services/action_log.py`): every step is a structured event with `request_id`; `GET /api/v1/actions`, SSE `kind="action"`,
  CLI `ingest chat --trace`.
- API: `request_id` in chat responses, `repo_id` from `/ingest`, `GET /api/v1/jobs[/{id}]`, `GET /api/v1/repos/{id}/search-mode`,
  `vector_status` / `ingest_mode` in the repo list. CLI: `ingest --vectorize`, `ingest jobs [repo-id]`.
- Settings: `ingest_mode`, `retrieval_mode`, `vectorize_fallback`, `vectorize_confidence_threshold`, `reasoning_supervisor`,
  `reasoning_intake_llm`, `action_log_enabled`, `watch_on_ingest`.
### Changed
- `ingest_repository` is now the fast agentic ingest (scan, snapshot, COMPLETED); chunking and embedding moved to
  `vectorize_repository` (embedding runs in a worker thread, progress per file batch). `ingest_mode=vector` embeds right after and waits.
- A re-ingested repo that already has vectors refreshes them in the background; file-watcher edits always refresh the raw snapshot and
  only re-embed when the vector index is ready.
- Offline catch-up at startup compares file hashes only for repos with a ready vector index; the raw snapshot is always synced.
- A named file (`utils.py`) in a question is read from the raw snapshot when the repo has no chunks.
- Conceptual queries that also ask for similarity ("similar to", "everywhere", ...) now score strictly below the default threshold
  (similarity penalty 0.2 -> 0.25), so they trigger the vectorize fallback.
- Ask agent: the default model's LLM event comes from the telemetry callback only (no duplicate agent-level event).
### Compatibility
- `chat_with_repo` still returns the answer string; the new optional `request_id=` argument lets the caller know the id up front.
- Pre-v2 repositories (chunks, no `vector_status`) are treated as vector-ready.
