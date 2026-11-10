
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
| `LLM_RUNNER` | `ollama` | `ollama`, `cloudflare`, `aistudio` or `openai` (the last three speak an OpenAI-style API and need `pip install langchain-openai`; see the table below) |
| `EMBEDDING_RUNNER` | `ollama` | separate from `LLM_RUNNER`: vectors of different models are not comparable, so switching it needs a re-index (`python reset_repo_index.py`) |
| `CF_LIST_ALL_MODELS` | `true` | Cloudflare runner: the model picker lists all Workers AI text-generation models on the account (`CF_MODELS` first, cached `CF_MODELS_CACHE_S`=600 s); `false` = only `CF_MODELS` |
| `CF_ACCOUNT_ID`, `CF_API_TOKEN` | *(empty)* | required for `cloudflare`; the app refuses to start without them when `LLM_RUNNER` / `EMBEDDING_RUNNER` is `cloudflare` |
| `CF_LLM_MODEL` / `CF_VISION_MODEL` | `@cf/google/gemma-4-26b-a4b-it` | Ollama-style names (contain `:`, no `@`, e.g. `gemma4:31b-cloud`) are mapped to these when `LLM_RUNNER=cloudflare` |
| `CF_EMBEDDING_MODEL` | `@cf/baai/bge-base-en-v1.5` | embedding model of the cloudflare embedding runner |
| `CF_MODELS` | gemma-4-26b-a4b-it, gemma-3-12b-it | comma list offered in the UI model picker |
| `CF_AI_GATEWAY_URL` | *(empty)* | optional base URL override (AI Gateway) |

Not supported on `cloudflare`: `num_ctx` (ignored), `ask_reasoning` (ignored). The decision model (`/v1/systemone`) stays Ollama-only.
Setup and troubleshooting: `docs/CLOUDFLARE_SETUP.md`.

### Available runners
| Runner (`name`) | UI label | Remote (lean tokens in `auto`) | Needs | Notes |
|---|---|---|---|---|
| `ollama` | Ollama (local) | no | a running Ollama | default; the behaviour the app always had |
| `cloudflare` | Cloudflare Workers AI | yes | `CF_ACCOUNT_ID`, `CF_API_TOKEN` | `docs/CLOUDFLARE_SETUP.md` |
| `aistudio` | Google AI Studio | yes | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini API, OpenAI-compatible endpoint; `docs/GOOGLE_AI_STUDIO_SETUP.md` |
| `openai` | OpenAI-compatible | `OPENAI_COMPAT_REMOTE` (default yes) | `OPENAI_COMPAT_BASE_URL` | any `/chat/completions` server, configured by settings only |

Only the runners named by `LLM_RUNNER` and `EMBEDDING_RUNNER` are checked at startup. Any other runner that is not set up simply shows as "(not set up)" in the UI runner select, with the reason as tooltip, and a chat request that picks it is answered with HTTP 400.

**Google AI Studio** (`aistudio`): `GEMINI_API_KEY`, `AISTUDIO_BASE_URL` (default `https://generativelanguage.googleapis.com/v1beta/openai/`), `AISTUDIO_LLM_MODEL` / `AISTUDIO_VISION_MODEL` (default `gemma-4-31b-it`), `AISTUDIO_EMBEDDING_MODEL` (`gemini-embedding-001`), `AISTUDIO_MODELS` (pinned picker list, default `gemma-4-31b-it,gemma-4-26b-a4b-it`), `AISTUDIO_LIST_ALL_MODELS` (`true`: also list `GET {base}/models`, gemini* / gemma* chat models, cached `COMPAT_MODELS_CACHE_S`=600 s), `AISTUDIO_FOLD_SYSTEM` (`auto|on|off`: merge the system prompt into the first user message; auto = for `gemma*` ids), `AISTUDIO_JSON_MODE` (`auto|on|off`: `response_format=json_object`; auto = `gemini*` only, Gemma relies on the loose JSON parser), `AISTUDIO_MAX_INPUT_TOKENS` (60000, lean mode). Ollama-style names map through a small table in `services/runners/aistudio_runner.py` (`gemma4:31b*` to `gemma-4-31b-it`, `gemma4:26b*` to `gemma-4-26b-a4b-it`, any other `x:y` to `AISTUDIO_LLM_MODEL`); the table does not apply to `LLM_MODEL` itself, which means `AISTUDIO_LLM_MODEL` (see "Default model" below). Also `AISTUDIO_TIMEOUT_S` (180), `AISTUDIO_MAX_RETRIES` (1), `AISTUDIO_REASONING_EFFORT` (empty = not sent).

**Generic OpenAI-compatible** (`openai`): `OPENAI_COMPAT_BASE_URL` (no value = not set up), `OPENAI_COMPAT_API_KEY` (optional for local servers), `OPENAI_COMPAT_MODELS` (pinned picker list), `OPENAI_COMPAT_LLM_MODEL` (what Ollama-style names map to; defaults to the first pinned model), `OPENAI_COMPAT_EMBEDDING_MODEL` (only for `EMBEDDING_RUNNER=openai`), `OPENAI_COMPAT_REMOTE` (`true`: billed service, so `TOKEN_MODE=auto` runs lean; `false` for a local server), `OPENAI_COMPAT_LIST_ALL_MODELS` (`true`: also list `GET {base}/models`), `OPENAI_COMPAT_MAX_INPUT_TOKENS` (0 = no cap). Examples **[unverified: base URLs taken from the providers' docs as known to the author, not tested here]**:
```
# OpenRouter
OPENAI_COMPAT_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPAT_API_KEY=sk-or-...
OPENAI_COMPAT_MODELS=qwen/qwen3-coder,meta-llama/llama-3.3-70b-instruct
OPENAI_COMPAT_LLM_MODEL=qwen/qwen3-coder
# Groq
OPENAI_COMPAT_BASE_URL=https://api.groq.com/openai/v1
OPENAI_COMPAT_API_KEY=gsk_...
OPENAI_COMPAT_LLM_MODEL=llama-3.3-70b-versatile
# local vLLM (no key, not billed)
OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
OPENAI_COMPAT_LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
OPENAI_COMPAT_REMOTE=false
```
Then `LLM_RUNNER=openai`, or pick it in the UI.

**Embeddings on `aistudio` / `openai`**: `EMBEDDING_RUNNER=aistudio` (or `openai`) works, but vectors of different embedding models are not comparable: after changing `EMBEDDING_RUNNER` or the embedding model, re-index every repo (`python reset_repo_index.py`). The per-request runner picked in the UI never changes embeddings.

### Default model, timeouts and the task timer (non-Ollama runners)
**What "Default" means.** The request carries `LLM_MODEL` (`gemma4:31b-cloud`, an Ollama tag) when the model picker is on "Default". On `cloudflare`, `aistudio` and `openai` that name, or an empty one, means *the runner's own default*: `CF_LLM_MODEL` / `AISTUDIO_LLM_MODEL` / `OPENAI_COMPAT_LLM_MODEL` (image calls: the runner's vision model). So `AISTUDIO_LLM_MODEL=gemini-2.5-flash` really calls Flash, `GET /api/v1/models` reports it as `default` (the UI shows "Default (gemini-2.5-flash)") and telemetry shows it. Only an explicit Ollama-style name that is *not* `LLM_MODEL` (for example a `LLM_FALLBACK_MODEL`) goes through the `aistudio` table (`gemma4:26b*` to `gemma-4-26b-a4b-it`; anything else to the default). Role models (`PATCHER_MODEL`, `DIAGNOSER_MODEL`, `CHANGE_ESCALATION_MODEL`, `TESTER_MODEL`, ...) that are Ollama tags are ignored on these runners (the runner default is used; an escalation model is skipped); `VISION_MODEL` maps to the vision default; `ROUTER_MODEL` is a decision model and stays Ollama-only. A fallback model that resolves to the same model as the refused one is never "fallen back" to. Every request logs `model resolved: runner=aistudio picked=Default -> gemini-2.5-flash` (Activity and the telemetry list); each telemetry row has `model` (the id really called) and `requested_model` (what the code asked for).

**Timeouts.** Per attempt: `AISTUDIO_TIMEOUT_S` (180), `OPENAI_COMPAT_TIMEOUT_S` (default `COMPAT_TIMEOUT_S`=120), `CF_TIMEOUT_S` (120); retries `AISTUDIO_MAX_RETRIES` (1), `OPENAI_COMPAT_MAX_RETRIES` (default `COMPAT_MAX_RETRIES`=2), `CF_MAX_RETRIES` (2). A hang costs `timeout x (retries + 1)`. Ollama has no client-side limit (`timeout_s` is `null`). `GET /api/v1/runners` returns `timeout_s` and `max_retries` per runner. A timeout answers **HTTP 504** `{"detail": "Google AI Studio did not answer within 180s (model ..., 2 attempt(s)). Try a faster model (e.g. a Gemini Flash model), Lean mode, or raise AISTUDIO_TIMEOUT_S.", "kind": "timeout", "runner", "model", "timeout_s"}`; an unreachable endpoint answers **503** (`"kind": "unavailable"`); any other unexpected error **500** `{"detail": "Type: message", "kind": "internal"}` (traceback in the server log). The chat shows each of them as an error message plus a toast, never nothing.

**Thinking.** `ASK_REASONING` only reaches Ollama. For `aistudio`, `AISTUDIO_REASONING_EFFORT` (`""` = never sent, the default; `none|minimal|low|medium|high` = sent on every call; `auto` = follow `ASK_REASONING` on the calls that carry it: `off` becomes `none`) sets `reasoning_effort` on the Gemini OpenAI-compatible endpoint. Gemini thinking models accept it; Gemma may reject it (unverified).

**Task timer.** While a chat request runs, a toast shows `Working - <runner> - <model>`, the elapsed `m:ss`, a bar toward the runner's per-call limit (blue below 50 %, amber 50-90 %, red above) and a **Stop** button. When an LLM call of the request finishes, the per-call part restarts ("step N - this call m:ss / T - total m:ss"). It ends as "Done in m:ss" (3 s) or "Failed after m:ss" (until clicked). Switch it off with "Show task timer" in the chat settings (browser setting `crag.taskTimer`).

### Switching runner and token mode from the UI
Next to the model picker the chat box has a **runner** select (built from `GET /api/v1/runners`, so a newly registered runner shows up by itself; runners that are not configured are disabled and show the reason as a tooltip) and a **Tokens: Auto / Lean / Normal** select. Both are saved in the browser (`runner`, `tokenMode`) and sent with every chat call (`runner`, `token_mode`; a runner that is unknown or not configured is answered with HTTP 400). Picking a runner reloads the model list (`GET /api/v1/models?runner=name`); a saved model that the new runner does not offer falls back to Default. The switch only affects **chat**: embeddings always stay on `EMBEDDING_RUNNER`, otherwise the vector index would stop matching.

### Token mode (fewer billed tokens on remote runners)
Every ask-agent step resends the whole conversation, so on a paid runner the tool output the agent has already read is billed again on each step. `TOKEN_MODE` controls how much is resent. `normal` is exactly the behaviour of earlier versions (same prompts, byte for byte).

| Setting (env) | Default | Meaning |
|---|---|---|
| `TOKEN_MODE` | `auto` | `auto` = `lean` for remote runners (`Runner.remote = True`, i.e. Cloudflare) and `normal` for Ollama; `lean` / `normal` force it. The UI toggle overrides it per question |
| `LEAN_ASK_MAX_STEPS`, `LEAN_ASK_OBS_CHARS`, `LEAN_ASK_READ_CHARS`, `LEAN_ASK_SEED_CHARS` | 6, 40000, 12000, 4000 | lean value of `ask_max_steps` / `ask_obs_chars` / `ask_read_chars` / `ask_seed_chars` (normal: 10 / 90000 / 30000 / 8000). A lean budget is never above the normal one |
| `LEAN_SKILL_PACK_MAX_CHARS`, `LEAN_CHANGE_CONTEXT_CHARS` | 6000, 40000 | lean value of `skill_pack_max_chars` / `change_context_chars`. The skill text is added only to edit requests (and to edit-like asks, core only), as in normal mode |
| `LEAN_KEEP_FULL_OBS` | 2 | the newest N tool outputs are resent in full; older ones become one line, e.g. `OBSERVATION (read) [compacted: services/x.py lines 1-185, 6,420 chars already seen; read again if needed]`. The model's own replies, the system prompt and your question are never touched |
| `LEAN_HISTORY_TURNS` | 6 | earlier chat messages sent along with the question |
| `MAX_INPUT_TOKENS` / `CF_MAX_INPUT_TOKENS` | unset / 60000 | estimated-token cap (characters / 4) for one call; `0` = off. `CF_MAX_INPUT_TOKENS` is the Cloudflare default and applies in lean mode; `MAX_INPUT_TOKENS`, when set, overrides it for every runner and also applies in normal mode on remote runners (a safety net). Over the cap the agent drops the oldest earlier chat messages first, then compacts older tool output; it logs what it trimmed and never drops the system prompt or your latest message |
| `CF_SESSION_AFFINITY` | `true` | Cloudflare: send `x-session-affinity: <stable id per repo + chat thread>` so repeated prompt prefixes can hit the prompt cache |
| `SHOW_TOKEN_FOOTER` | `true` | add `_Tokens: 18.2k in (6.1k cached) · 0.9k out · 6 calls · runner cloudflare · mode lean (saved ~31k)_` under each ask answer |

Telemetry: every LLM call event carries runner, mode, input / output / cached tokens and the estimated tokens saved, and each question ends with one "Token usage" event (sum over its calls). Both appear in the Telemetry tab. "Saved" is an estimate (characters not sent / 4) from compaction, history trimming, the input cap, the lean read and seed budgets; step and observation-total budgets are not counted. Cached tokens are shown when the server reports them.

### How to add a runner
1. For any OpenAI-style API, subclass `OpenAICompatRunner` (`services/runners/openai_compat.py`) and fill in `name`, `label`, `remote`, `base_url()`, `api_key()`, `check_config()`, `llm_default()` / `embedding_default()` (plus optional hooks: `map_model`, `json_enabled`, `extra_headers`, `chat_class`, `pinned_models`, `_fetch_catalog`); `aistudio_runner.py` is a ~100 line example. For any other API write a class in `services/runners/` with `label` (shown in the UI), `name`, `remote` (True = paid/remote, makes `TOKEN_MODE=auto` lean), `max_input_tokens` (0 = no cap), `supports`, and the methods `check_config()` (raise `RunnerConfigError`, no network), `map_model(model, vision=False)`, `chat(model, **kw)` (a LangChain chat model), `embeddings(model=None, ...)` and `async list_models()`. See `services/runners/base.py` for the exact contract and `ollama_runner.py` / `cloudflare_runner.py` for examples.
2. Register it: `register_runner("myrunner", MyRunner)` (the registry in `services/runners/__init__.py`).
3. That is all: `LLM_RUNNER=myrunner` works, and it shows up in `GET /api/v1/runners` and the UI runner select. Call sites always use `make_chat()` / `resolve_model()`, which honour the per-request runner.
