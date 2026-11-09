# CodeAssist v3 - Skill pack & lanes

Extends the v2 reasoning loop (`docs/DESIGN_v2.md` section 3); nothing new runs beside it. Source: the "risk-proportional code development" skill.

## 1. Where it plugs in
| v2 step | v3 addition |
|---|---|
| Intake (`IntakeAgent`) | `IntakeBrief.lane`, `lane_reasons`, `change_type`, `validation_level`, `needs_handoff` (heuristics in `intake.py`) |
| Baseline (`policy_for`) | lane -> tier: Fast = what the risk says, Standard = Medium depth, Strict = High depth. Only raises (`max(risk, lane)`) |
| Execute | `ReasoningSupervisor` asks `guidance_provider(brief, policy)` for the skill text and runs `execute` inside `policy_scope(overrides, guidance=...)` |
| Prompts | `repo_change._ctx_block` prepends `guidance_block()` to the plan, contract, tests-first and patch prompts. Outside a supervised turn it returns "" |
| Cross-check / deliver | `DeliveryNotes.lane`, `validation_level`, `verification` (labels); Fast-lane rendering is a few lines |

The seam is the existing per-request ContextVar overlay in `baseline.py` (the same one `flag()` uses): the guidance rides with the overrides, so it
is isolated across concurrent chats and reaches worker threads. No prompt template was restructured.

## 2. Lane rules (`intake.py: detect_change_type`, `classify_lane`)
1. Change type: Strict types are tried first (migration, IaC / CI-CD, front-end security), then front-end behaviour, SQL, script, presentational
   front-end, docs, else backend / other. Terms come from the request and from files it names (an extension in the text, or a repo file whose base name
   the text mentions); the repo file list alone is never a signal.
2. Start from the type's lane (presentational FE and docs Fast; behavioural FE, SQL, scripts, backend Standard; the rest Strict). A script marked
   disposable and a rename inside one named file or a test-only tidy start at Fast.
3. Raise: High risk, auth / payments / prod-data terms -> Strict. Medium risk, global CSS / design tokens, consent / disclosure copy, flags /
   timeouts / env vars -> at least Standard. Enterprise context already makes risk Medium.
4. Override (`lane: x` in the request, else `settings.lane_override`): any raise is honoured; a lower lane is honoured unless step 3 found a Strict trigger.
5. `validation_level` is the ladder level(s) worth aiming for (FE presentational L3, FE behaviour L1+L3, docs L1, SQL L1+L2, migration / IaC L2,
   Strict backend L2, +L4 when `needs_handoff` and Strict). L5 is never recommended. It is not the "Verification level" label of a proposal.

## 3. Skill pack (`services/reasoning/skill_pack.py`)
`SKILL.md` always; `validation-ladder` for Standard / Strict, or Fast + presentational FE (only the levels table and the FE row);
`regulated-domains` for Strict or High risk; `delivery-and-review` for Standard / Strict; `task-types-and-risk` is reference only (intake already
classifies). Intents other than change / plan get "". Text is capped at `skill_pack_max_chars` (later sections are dropped first, the core is cut only
if it alone exceeds the cap). Cache key: (name, mtime_ns, size) of the core and references. Missing, unreadable or broken core -> "" and one log line.
A missing reference is skipped.

## 4. Labels
Derived in `crosscheck.verification_labels` from facts only: tests ran and passed -> "Verified - executed in container" (`sandbox_backend=docker`) or
"... in sandbox"; `static_checked` (a saved proposal passed apply_check) -> "Statically checked"; otherwise "Unverified (user must run)". The headless-browser and
live-env labels need `browser_ran/browser_ok` or `live_env/live_ok/user_assisted` facts, which no current code path sets.

## 5. Failure behaviour
Pack problems and a failing guidance provider only remove the guidance (the latter logs a failed BASELINE/SkillPack action). Intake / baseline
failures still fall back to plain execution as in v2.

## 6. v3.1 changes
- **Facts from the sandbox** (`crosscheck.facts_from_sandbox`): tests_* only for mode `pytest` with a count; boot and import checks are separate facts. In boot mode
  `passed` counts boot steps. The labels, the Verified list and the validation summary all read these facts.
- **One scale.** Notes show "Recommended: L1, L3 · Achieved: L1 (boot smoke)" and one ran / skipped line per recommended level. The repo's "Verification level" maps to
  the ladder as static = L0; boot / import smoke and repo tests = L1 (the sandbox is a container running tests or a smoke check; ephemeral services and rollback are not covered, so L2
  is reported "partial" at best). L3 / L4 need `browser_*` / `live_*` facts that no code path sets, so they read "skipped - no browser runner in this version".
- **All change types.** `detect_change_types` returns every type present; the lane is the highest across them (`classify_lane` accepts a list) and the validation levels are merged.
- **Ask that proposes edits.** `CodeAgent._check` writes `edits_proposed`, `static_checked` and the sandbox facts into the turn's fact box (`baseline.CHAT_FACTS`), so the
  verifier treats it like a change (lane, labels, summary). A change-like ask also gets the core skill text via `guidance_block()` in `CodeAgent.answer`; this only applies
  when `ask_agent` is on (the single-shot ask path is unchanged).
- **Unknown `/command`** is answered in `intent_router` before any model call (`RouteDecision.reply`).
- **No echo.** The guidance header is marked internal; `strip_guidance_echo` removes echoed header / lane lines from plans and answers.

## 7. v3.2 changes
- **Stacking** (`services/reasoning/stacking.py`): "unapplied" follows the app's own model (revision 0 = disk; `applied_at` marks applied revisions; disk-sync and
  revert revisions are skipped). The cumulative stats are index(0) vs index(rev), as in the cumulative `.patch`.
- **Repeat guard**: `EngineController._route_chat` asks `find_repeat` before `propose_change`; a hit returns `repeat_reply` and sets `skip_notes` so the supervisor adds
  no notes. The guard is repo-level (latest unapplied edit revision), not per thread; any lookup error falls through to a normal build.
- **Summary** is built from facts (`revision`, `plan`, `change_files`, `stacked_on`, test facts) only.
- **Requirement cross-check** judges two criteria from facts; the label is carried separately from the 4-value status so v2 consumers keep working.

