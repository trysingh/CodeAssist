# CodeAssist v2 — Design & Contracts

Owner of this spec: supervisor (Opus). Implementers MUST follow these contracts verbatim so parallel work integrates.

## 0. Goals
1. Align every request with the Reasoning & Action loop (Intake → Decompose → Baseline ∝ risk → Contracts → Core → Hardening → Tests → Sandbox → Fix → Cross-check → Deliver-with-notes).
2. Log **every action** (agent step, tool call, LLM call, sandbox run, ingest/vectorize step) as a structured event and show it live in the UI ("Activity" timeline), grouped per request.
3. Default ingest = **agentic** (raw snapshot + lightweight file/symbol map, NO embeddings). Retrieval = grep/glob/read/symbol search. When agentic search is weak (edge cases), enqueue a **background vectorize job**, tell the user why, notify on progress/completion.

Constraint: new modules are dependency-light (stdlib + pydantic only) and take config by injection so they are unit-testable without langchain/fastapi/pydantic_settings.

## 1. Sub-agent model (what is new vs existing)
| Loop step | Agent | Status |
|---|---|---|
| 1 Intake (intent/risk/ambiguity, goal, success criteria, constraints, NFRs, assumptions) | `IntakeAgent` (services/reasoning/intake.py) | NEW |
| 2 Decompose | existing `_plan` + `tasks.plan_tasks`; IntakeBrief.outcome_tasks for ask/plan | existing + NEW brief field |
| 3 Baseline ∝ risk | `BaselinePolicy` (services/reasoning/baseline.py) → per-request policy overlay | NEW |
| 4–9 Contract/Core/Hardening/Tests/Sandbox/Fix | existing change graph (designer/implementer/tester/diagnoser/reviewer) | existing, now logged per node |
| 10 Cross-check direct + professional reqs | `VerifierAgent` (services/reasoning/crosscheck.py) | NEW |
| 11 Deliver with assumptions/limitations/verified | `DeliveryNotes.to_markdown()` appended to every answer | NEW |
| Orchestration | `ReasoningSupervisor` (services/reasoning/supervisor.py) | NEW |
| Retrieval | `AgenticSearch` (services/agentic_search.py) + `VectorizeJobManager` | NEW |

## 2. Action log — services/action_log.py (Agent A)
```python
@dataclass
class ActionEvent:
    uid: str; ts: str                     # ISO UTC ms Z
    request_id: str | None; thread_id: str | None; repo_id: str | None
    phase: str        # INTAKE|DECOMPOSE|BASELINE|CONTRACT|IMPLEMENT|HARDEN|TEST|SANDBOX|FIX|CROSSCHECK|DELIVER|RETRIEVE|INGEST|VECTORIZE|LLM
    agent: str        # e.g. "IntakeAgent", "AskAgent", "ChangeGraph.patch", "AgenticSearch", "Vectorizer"
    action: str       # short verb phrase: "grep", "read src/a.py:10-80", "llm_call", "run tests"
    status: str       # start|ok|fail|skip|info
    detail: str = ""  # one human line
    data: dict = {}   # small JSON-safe payload (≤ 4 KB after truncation)
    duration_ms: int | None = None
    model: str | None = None
    parent_uid: str | None = None   # start-event uid for ok/fail closing events
    def to_dict(self) -> dict

request_scope(request_id, thread_id=None)   # contextmanager; ContextVars
current_request_id() / current_thread_id()
set_sink(fn: Callable[[ActionEvent], None]) / add_sink / clear_sinks   # sinks never raise into caller
log_action(phase, agent, action, status="info", detail="", data=None, model=None, duration_ms=None, parent_uid=None) -> ActionEvent
@contextmanager action(phase, agent, action, detail="", data=None, model=None)  # emits start, then ok (with duration) or fail (exception repr) and re-raises; yields a handle with .note(detail=..., **data) to enrich the closing event
def traced(phase, agent, action=None)   # decorator for sync AND async functions
RingBuffer: recent(request_id=None, repo_id=None, limit=500) -> list[dict]   # in-memory, cap 5000
```
Wiring sink (done by Agent C in engine_controller/api startup): sink → `notifier.notify(title="Action", message=detail, level=<INFO|ERROR on fail>, repo_id=ev.repo_id, kind="action", request_id=..., thread_id=..., data=ev.to_dict())`.

### Notifier / TelemetryEvent extension (Agent A edits notifier.py + schemas.py; Agent B edits database.py)
`TelemetryEvent` gains optional: `kind: str = "log"` (log|action|telemetry|job), `request_id: Optional[str]`, `thread_id: Optional[str]`, `data: Optional[dict]`.
`notify(title, message, level="INFO", repo_id=None, *, kind="log", request_id=None, thread_id=None, data=None)` — backward compatible; request/thread default from action_log ContextVars.
`telemetry_events` table: additive columns `kind TEXT DEFAULT 'log'`, `request_id TEXT`, `thread_id TEXT`, `data TEXT` (JSON); insert/select updated; index on request_id.
SSE: unchanged event names (`history`, `notification`); payload now carries the new fields. New REST: `GET /api/v1/actions?request_id=&repo_id=&limit=` (ring buffer, then DB).

### Where actions are emitted (Agent A)
- `CodeAgent`: each tool call (find/grep/ls/read) phase RETRIEVE; each LLM turn phase LLM; validate/sandbox phase SANDBOX.
- `RepoChangeService._build`: wrap every node with `action(phase=<map>, agent=f"ChangeGraph.{node}", action=node)`. Map: locate→RETRIEVE, plan→DECOMPOSE, contract→CONTRACT, tests_first/author_tests→TEST, tasks/patch→IMPLEMENT, apply_check/fix_tasks→HARDEN, sandbox→SANDBOX, gate/review→CROSSCHECK, persist→DELIVER.
- `model_calls` / telemetry callback: one `LLM` event per call with model + tokens + duration.

## 3. Reasoning package — services/reasoning/ (Agent A)
```python
class Risk(str, Enum): LOW="low"; MEDIUM="medium"; HIGH="high"
class IntakeBrief(BaseModel):
    intent: str                      # ask|plan|change|help (from router)
    explicit_ask: str; implicit_needs: list[str]
    goal: str; success_criteria: list[str]; constraints: list[str]; nfrs: list[str]
    risk: Risk; risk_reasons: list[str]
    ambiguity: float (0..1); ambiguity_reasons: list[str]; clarifying_questions: list[str]
    assumptions: list[str]; outcome_tasks: list[str]
    enterprise: bool                 # "smells enterprise" → raise bar (§7 of approach note)
    source: str                      # "heuristic" | "llm"
class IntakeAgent:
    def __init__(self, llm_json: Callable[[str], dict] | None = None)
    def analyze(self, query: str, intent: str, repo_files: list[str] | None = None) -> IntakeBrief
```
Heuristics (always run; LLM optional and merged, never required): risk keywords (auth, password, token, payment, money, PII, delete, drop table, migration, prod, security, encryption, compliance, GDPR, KYC, AML, transaction) → HIGH; schema/API/concurrency/perf → MEDIUM. Ambiguity: very short request, pronouns without referent ("fix it"), no file/symbol/identifier, "something/somehow/etc", missing success criteria. Baseline NFRs always: correctness, clear error handling, readability, no obvious security holes, reasonable performance.

```python
class BaselinePolicy(BaseModel):
    depth: str                       # baseline|elevated|strict
    checklist: list[str]             # approach note §2 L3 items applied
    overrides: dict[str, Any]        # settings overlay, keys ⊆ {change_gate, change_review, change_tests_first, change_author_tests, change_max_attempts}
    clarify_first: bool              # ambiguity ≥ 0.7 AND risk HIGH AND intent==change
def policy_for(brief: IntakeBrief, defaults: Mapping[str, Any]) -> BaselinePolicy
# overlay (ContextVar):
policy_scope(overrides: dict) contextmanager;  flag(name: str, default) -> value   # overlay value else default
```
Risk mapping: LOW → baseline, no overrides (user's settings stand). MEDIUM → elevated: change_review=True, change_gate=max(current,"smoke"). HIGH → strict: change_gate="strict", change_review=True, change_tests_first=True. Overrides only ever RAISE strictness, never lower a user's setting.
Agent A replaces the reads of exactly those settings in repo_change.py with `flag("<name>", settings.<name>)`.

```python
class DeliveryNotes(BaseModel):
    assumptions: list[str]; limitations: list[str]
    verified: list[str]; unverified: list[str]
    requirements: list[dict]         # {"req": str, "kind": "direct"|"professional", "status": "met"|"partial"|"unmet"|"unknown", "evidence": str}
    def to_markdown(self) -> str     # "### Delivery notes" section; never claims a test passed unless verified lists it
class VerifierAgent:
    def __init__(self, llm_json=None)
    def check(self, brief: IntakeBrief, answer_md: str, facts: dict) -> DeliveryNotes
    # facts: {"tests_ran": bool, "tests_passed": bool|None, "sandbox": str|None, "files_changed": [...], "retrieval_mode": "agentic"|"vector"|"hybrid", "retrieval_confidence": float, "vectorize_job": str|None}
```

```python
class ReasoningSupervisor:
    def __init__(self, intake: IntakeAgent, verifier: VerifierAgent, defaults_provider: Callable[[], Mapping], enabled: bool = True)
    async def run(self, *, query, intent, repo_files, execute: Callable[[IntakeBrief, BaselinePolicy], Awaitable[tuple[str, dict]]]) -> str
    # 1 log INTAKE (brief) 2 BASELINE (policy) 3 if clarify_first → return clarifying questions md (DELIVER) without executing
    # 4 with policy_scope(policy.overrides): answer, facts = await execute(brief, policy)
    # 5 CROSSCHECK → notes 6 DELIVER: return answer + "\n\n" + notes.to_markdown()
    # every step wrapped in action(...). If enabled=False → just execute and return answer.
```
Agent C wires `_chat_impl` → supervisor.run(execute=<existing _route_chat path>). Facts collected from the existing paths best-effort.

## 4. Agentic search — services/agentic_search.py (Agent B)
```python
@dataclass
class Hit: path: str; line: int; text: str; score: float
@dataclass
class Block: path: str; start: int; end: int; text: str; score: float; why: str
@dataclass
class SearchResult:
    blocks: list[Block]; confidence: float (0..1); reasons: list[str]   # why low/high
    terms: list[str]; files_considered: int; elapsed_ms: int
    def to_context(self, max_chars: int) -> str        # same "--- path:start-end ---\n<text>" style rag uses
class RepoCorpus:            # in-memory view, built from {path: text}
    @classmethod from_mapping(cls, files: Mapping[str, str | bytes]) -> RepoCorpus
    @classmethod from_dir(cls, root, include_globs=None, exclude_dirs=DEFAULT_EXCLUDES, max_file_bytes=1_000_000) -> RepoCorpus
    paths() -> list[str]
class AgenticSearch:
    def __init__(self, corpus: RepoCorpus, on_action: Callable[[str, str, dict], None] | None = None)  # (action, detail, data) → used to log RETRIEVE events
    def glob(self, pattern: str, limit=200) -> list[str]          # supports **, *, ?, {a,b}
    def grep(self, pattern: str, path_glob: str | None = None, regex=True, ignore_case=True, limit=200) -> list[Hit]
    def read(self, path: str, start: int = 1, end: int | None = None) -> str
    def ls(self, dir: str = "") -> list[str]
    def symbols(self, name: str | None = None) -> list[Hit]       # regex defs: def/class/function/const/let/var/interface/type/struct/fn/func/public|private ... (py, js/ts, java, go, rs, cs, kt, rb, php)
    def locate_files(self, query: str, k: int = 8) -> list[str]   # for change-graph _locate
    def retrieve(self, query: str, k_blocks: int = 6, window: int = 30) -> SearchResult
```
`retrieve` pipeline: extract terms (quoted strings, paths, CamelCase/snake_case identifiers split + whole, file extensions, remaining keywords minus stopwords) → path/glob match → symbol-definition match (weight 3) → grep term hits (weight 1, IDF-weighted, cap per file) → filename match bonus → merge nearby hits into windows → top k blocks.
Confidence: start 1.0; penalties — no identifier/path terms (conceptual NL query) −0.35; zero hits → 0.0; top block score < threshold −0.3; hits spread across > 25 files with flat scores −0.2; corpus > 5000 files −0.1; query mentions "similar to / like / concept / semantic / meaning / everywhere / all places that" −0.2. Each penalty appends a human reason. Clamp 0..1.

```python
def needs_vectors(result: SearchResult, threshold: float = 0.45) -> tuple[bool, str]   # (True, reason) when below threshold
```

## 5. Vectorize jobs — services/vectorize_jobs.py (Agent B)
```python
class JobState(str, Enum): QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
@dataclass
class Job: job_id; repo_id; kind="vectorize"; state; reason; progress:int; total:int; error:str|None; created_at; updated_at; def to_dict()
class JobStore(Protocol): async upsert(job: Job); async get(job_id); async list(repo_id=None, limit=50); async latest(repo_id, kind)
class InMemoryJobStore(JobStore)
class VectorizeJobManager:
    def __init__(self, store: JobStore, work: Callable[[str, Callable[[int,int],None]], Awaitable[None]], notify: Callable[..., None], set_vector_status: Callable[[str,str], Awaitable[None]] | None = None)
    async def enqueue(self, repo_id: str, reason: str) -> Job      # dedup: return existing QUEUED/RUNNING job for repo
    async def cancel(self, job_id) -> bool
    def status(self, repo_id) -> str   # none|queued|running|ready|failed (cached)
    async def wait(self, job_id, timeout=None)   # tests
```
Runs `work(repo_id, progress_cb)` in an asyncio task; progress_cb throttles notifications (every ≥10% or 15 s). Notifications: start (INFO, "Vectorizing <repo> in background: <reason>. This may take a while; you'll be notified."), progress (INFO, kind="job"), done (SUCCESS → UI toast "Vector index ready — semantic search enabled"), fail (ERROR with error). Also emits VECTORIZE action events via action_log if importable (soft import).
`set_vector_status(repo_id, "queued|running|ready|failed")` updates DB.

### DB (Agent B, models/database.py)
- `repositories`: additive `vector_status TEXT DEFAULT 'none'`, `ingest_mode TEXT DEFAULT 'agentic'`.
- new `jobs` table matching Job; `DbJobStore` implementing JobStore over aiosqlite (in database.py or vectorize_jobs.py with soft import).
- telemetry_events additive columns (see §2) + update insert/select helpers.
- helpers: `set_vector_status(repo_id, status)`, `get_vector_status(repo_id)`.

### Retrieval switch (Agent B, rag_graph.py)
`_retrieve_and_expand_node`: named-file path unchanged. Else: if `vector_status == "ready"` and `settings.retrieval_mode in ("vector","hybrid")` → existing vector path (hybrid = agentic blocks + vector hits). Otherwise agentic: build corpus from raw store snapshot (cache per repo_id + store revision), `retrieve(query)`; if `needs_vectors` and `settings.vectorize_fallback == "auto"` and status in (none, failed) → `await jobs.enqueue(repo_id, reason)`; state gets `retrieval_meta = {"mode","confidence","reasons","job_id"}` for the supervisor facts. Also fix `ChatOllama` base_url to `chat_base_url()`.

## 6. Integration (Agent C)
- settings: `ingest_mode: str = "agentic"` (agentic|vector), `retrieval_mode: str = "agentic"` (agentic|vector|hybrid; auto-upgrades to hybrid when vectors ready), `vectorize_fallback: str = "auto"` (auto|off), `vectorize_confidence_threshold: float = 0.45`, `reasoning_supervisor: bool = True`, `reasoning_intake_llm: bool = False`, `action_log_enabled: bool = True`.
- engine_controller: split ingest → `ingest_repository` (snapshot + repo map + status COMPLETED, vector_status none; if ingest_mode=="vector" enqueue vectorize immediately) and `vectorize_repository(repo_id, progress_cb)` (old chunk/embed loop, embedding via `asyncio.to_thread`); start watcher for newly ingested repos; `_chat_impl` sets `request_scope` and runs `ReasoningSupervisor`.
- api_view: `/ingest` returns repo_id; `POST /api/v1/repos/{id}/vectorize`; `GET /api/v1/jobs`; `GET /api/v1/jobs/{job_id}`; `GET /api/v1/actions`; `/chat` response adds `request_id`.
- CLI: `ingest --repo X [--vectorize]`, `ingest jobs`.
- frontend: Activity tab (live timeline grouped by request: phase chips, agent, action, status icon, duration, model badge, expandable data), per-chat "Show steps" link, job progress bar + toasts, repo badge showing search mode (Agentic / Vectorizing n% / Hybrid).
- tests/conftest.py: `ENGINE_REASONING_SUPERVISOR=false` for legacy tests unless they opt in.
