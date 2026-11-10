# Changelog

## v3.2 (unreleased)
Fixes from re-running the same `/change` on v3.1.1 (revision 23 built on unapplied revision 22).
### Fixed
- **Chat crashed with `ollama._types.ResponseError: Add usage credits or subscribe ... (status code: 402)`.** The default model `gemma4:31b-cloud`
  runs on Ollama Cloud; when its credits / free usage run out ollama.com refuses every call with HTTP 402 (a provider limit, not a bug) and the
  request ended as an unhandled 500. New `services/llm_errors.py`: the 402 (and a 429 that talks about usage / credits) is recognised and
  becomes `ModelQuotaError`; the API answers HTTP 402 with a readable `detail` (the UI toast shows it) instead of a traceback. With
  `ENGINE_LLM_FALLBACK_MODEL=<local model>` the call is retried once on that model, later calls skip the cloud model for
  `ENGINE_LLM_QUOTA_COOLDOWN_S` seconds (default 300, 0 = always try the cloud model first) and the answer ends with a "Model note" line.
  Wired into the answer step (`rag_graph`), the ask agent (`code_agent._ollama`) and `model_calls.ask_text` / `ask_json` (patcher, locator, ...).
  A busy local Ollama answering 429 is not treated as a quota problem. Tests: `tests/test_llm_quota.py`.
- **Stacked revisions were invisible.** A revision built on unapplied ones showed only its own delta (+6 -6). The reply now carries "Builds on unapplied
  revision(s) 22. Shown below: this revision's changes only. The cumulative change against your disk: <per-file stats> - use Download .patch (cumulative) or apply
  22 and 23 together." (`services/reasoning/stacking.stack_info`, called from `RepoChangeService._run`, rendered by `ChangeResult.to_markdown`). Same data as
  the sandbox "Verified on top of N file(s)" message and the cumulative `.patch`; both full diffs are never inlined.
- **Repeated request regenerated the whole change.** A `/change` whose request is >= 0.9 similar (difflib, normalised) to the latest unapplied edit revision's request
  makes no model call: "This is the same request as revision N (not applied yet), which already implements it ..." with the apply / patch links. Override words:
  redo, again, regenerate, from scratch. A revision without a stored request is not a repeat. Setting `repeat_request_guard` (default true). Seam:
  `EngineController._repeat_hit` -> `stacking.find_repeat`. The patch prompt now says to make no edits when the code already satisfies the request and never to
  change unrelated values (timings, shadows, spacing); the review prompt flags edits unrelated to the request.
- **No plain summary.** A deterministic "**Summary**" block (no model call) heads the change reply: what changed per file (from the plan), how to apply
  (cumulative patch when stacked), what was verified, next step (`stacking.change_summary`, added by `ReasoningSupervisor`).
- **Contradictory notes.** "Sandbox: boot check passed" is no longer listed under Not verified; "No test covers this change; only the boot check ran" and
  "Runtime behaviour beyond the boot check" are one statement, not two. Requirement cross-check is fact-based: "covered by a test that fails before and
  passes after" is [not met] without tests, [met] only with tests and achieved L3-fail-to-pass, else [partial]; "existing behaviour still works" is
  [partial - boot check only] or [met] when tests passed; keyword-only matches are shown "(keyword match only, not proof)" and [unverified].
### Changed tests
- `tests/test_v31_facts_labels.py::test_ask_that_proposed_edits_gets_lane_labels_and_summary`: the boot-check statement is asserted as a limitation and no longer in
  Not verified (deliberate 4b dedupe).
- `tests/conftest.py` sets `ENGINE_REPEAT_REQUEST_GUARD=false` for legacy tests; v3.2 tests turn it on.
### Added
- `tests/stubbed_imports.py`: imports the controller and `repo_change` with inert stand-ins for missing third-party packages so call sites run against fakes.

## v3.1 (unreleased)
Fixes from a real run of v3 (a `/apply` ask and a `/change` on `templates/dashboard.html`).
### Fixed
- **Dishonest label.** A boot-only sandbox result ("no test covers this change") was reported as "Verified - executed in container: sandbox tests
  passed". `tests_ran` / `tests_passed` are now set only when pytest actually ran (mode `pytest`, pass or fail count above 0); in boot mode `passed`
  counts boot steps, not tests. New facts: `tests_count`, `boot_checked`, `import_checked`, `achieved_level`, `sandbox_not_run`, `sandbox_note`
  (`crosscheck.facts_from_sandbox`, used by `/change` and the ask agent). Labels: "Verified - executed in container: N sandbox tests passed" only
  with N > 0; "...: boot check only (no tests cover this change)"; "...: import smoke check only (...)". A pass reported without a count is Unverified.
- **Wrong lane on mixed requests.** Intake now detects every change type (`change_types`; `change_type` stays the primary one) and the lane is the highest
  across them. JS behaviour terms (audio / sound, click, fetch, event handlers, "starting and stopping", ...) make a request at least Standard, so
  "modern look + colour-blind palette + audible effects on start/stop + responsive" is Standard; a pure CSS colour tweak stays Fast.
- **Unknown slash command.** `/apply ...` no longer becomes a silent read-only ask. A leading `/word` (followed by space or end) that is not a command
  returns "Unknown command `/apply`. Did you mean `/change` (propose a revision) or `/ask`?" plus the command list, with no model call.
  Paths such as `/home/x fails` are not commands. `/apply` is not an alias.
- **Guidance leaking into output.** The guidance header is now "INTERNAL CONTEXT ... never repeat, quote or mention it", the lane statement says not to state
  the lane, and echoed lane / guidance header lines are stripped from the plan text and from the delivered answer.
### Changed
- Delivery notes: the separate "Level: L3" text is replaced by one validation run summary on the ladder scale, in every lane:
  "Recommended: L1, L3 · Achieved: L1 (boot smoke)" plus one line per recommended level ("L3 headless browser: skipped - no browser runner in this version",
  "L1 tests: skipped - no test covers this change"). Reasons come from the facts; no capability is claimed that the app does not have.
  The repo's own "Verification level" (L0-static ... L3-fail-to-pass) maps onto this scale: static = L0; boot / import smoke and repo tests = L1.
- The ask agent feeds its sandbox check into the same facts (`edits_proposed`, `static_checked`, boot / test facts), so an ask that proposes edits gets the lane,
  labels and validation summary, and "Runtime behaviour (no tests executed)" is no longer printed when a boot check ran.
- An ask that reads like an edit request (`IntakeBrief.change_like`) gets the CORE skill text only (no references) in the ask agent's system prompt
  (`CodeAgent.answer`), within `skill_pack_max_chars`. Other asks still get no pack.
- Skill pack: the FE behavioural row of `validation-ladder.md` gains the browser autoplay / user-gesture, single long-lived AudioContext, feedback-after-outcome,
  user-controllable effects and "not colour alone" practices.
- `ChatFacts` ContextVar moved to `reasoning/baseline.py` (`CHAT_FACTS`, `note_facts`); the controller keeps `_CHAT_FACTS` as an alias.
### Tests
- New: `tests/test_v31_facts_labels.py`, `tests/test_v31_lanes_router.py`. Changed v3 expectations (they encoded the bugs): tests in `test_v3_lanes.py` that
  passed `tests_passed=True` without a test count now pass `tests_count`, and one asserted the old "Validation level (recommended)" text.

## v3 (unreleased)
### Added
- Skill pack `skills/risk-proportional-code-dev/` (data, not code): a ~57-line `SKILL.md` core (flow, lanes, never-skip rules) plus
  `references/` (task-types-and-risk, validation-ladder, regulated-domains, delivery-and-review). Edit or swap it without touching Python.
- `services/reasoning/skill_pack.py`: loads the pack from `skill_pack_dir` (no yaml dependency), picks sections by lane / change type
  (core always; validation ladder for Standard/Strict and for Fast presentational front-end, where only the relevant part is loaded;
  regulated domains for Strict or High risk; delivery and review for Standard/Strict), caps the size, caches by file mtime.
  A missing or broken pack gives no guidance (one log line) and never an error in the chat path.
- Lanes (`Lane` FAST / STANDARD / STRICT) and `ChangeType` (FE presentational / behavioural / security-relevant, docs, SQL, migration,
  IaC-config, script, backend-other) on `IntakeBrief`, with `lane_reasons`, a recommended `validation_level` (ladder L0-L5, shown in the notes
  as "Validation level (recommended)") and `needs_handoff` (SSO / MFA / login / OTP / CAPTCHA). Highest trigger wins; High risk, migration, IaC,
  auth, payments and front-end security triggers are Strict; global CSS / design tokens, consent and disclosure copy, and flags / timeouts /
  env vars are at least Standard. `lane: fast` (or `use the strict lane`) in a request, or `ENGINE_LANE_OVERRIDE`, picks a lane, but never
  below Strict while a Strict trigger is present.
- Delivery notes carry `lane`, `validation_level` and verification labels ("Verified - executed in sandbox / container / headless browser",
  "Verified - live env, user-assisted", "Statically checked", "Unverified (user must run)"), derived only from facts that say the check ran.
  Fast-lane notes are a few lines.
- Settings: `skill_pack_enabled`, `skill_pack_dir`, `skill_pack_max_chars`, `lane_override`.
### Changed
- `BaselinePolicy` maps the lane onto the existing knobs (Standard = Medium depth, Strict = High depth). It can only raise a setting, as before.
  With the default settings (`change_gate=strict`, `change_review=true`) nothing changes; if you switched those off, Standard and Strict
  requests now switch review / gate on for that request.
- `classify_risk` no longer treats "design token(s)" (styling) as an auth token.
- `policy_scope(overrides, guidance="")` also carries the skill-pack text; `repo_change._ctx_block` puts it in front of the plan, contract,
  tests-first and patch prompts of a supervised change / plan turn. Ask / help turns get no pack.
- Facts for the cross-check gain `sandbox_backend` and `static_checked`.
### Compatibility
- `IntakeBrief()` built without a lane (the fallback path, old tests) behaves exactly as in v2: no lane rules, no guidance.
- `ReasoningSupervisor(...)`, `IntakeAgent(...)` and `policy_scope(...)` keep their v2 call signatures (new arguments are optional).
- "Verification level" in a change proposal (L0-static ... L3-fail-to-pass, `repo_change.py`) is what was achieved; the notes' "Validation level"
  is the ladder level recommended for the change. They are different scales.

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
