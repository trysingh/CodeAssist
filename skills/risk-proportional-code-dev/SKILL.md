---
name: risk-proportional-code-dev
description: Use for any coding, debugging, refactoring, code review, migration, IaC or front-end change request; sets rigour by risk and validates with executed evidence.
---

# Risk-Proportional Code Development

Match rigour to blast radius, reversibility and how the change can be verified. Prefer executed evidence over static reasoning. Isolate first. Keep the user in the loop for anything needing their identity.

## Flow: classify, lane, gates, deliver
1. Classify: task type, risk, ambiguity, primary goal, success criteria, constraints, implied non-functionals (correctness, readability, maintainability, reasonable performance, basic security). The highest matching risk level wins.
2. Select the lane (see Lanes). State the lane in one line in your reply.
3. Gates: validate at the lowest ladder level that gives real evidence; confirm before irreversible actions; never game the tests.
4. Deliver: label every correctness claim with how it was verified.

## Never skip (every lane)
- Never weaken, skip, delete or rewrite a test to get a pass; change a test only when the requirement it encodes was wrong, and say so.
- Never special-case test inputs or hard-code expected outputs.
- Confirmation gate: get explicit confirmation before `git push`, schema migrations, deletions, deployments, paid or external API calls, anything on production data. Dry-run first and show the output. Work on copies; never modify the user's originals. State what will happen and how to undo it, then wait.
- Credentials: the user enters credentials/MFA themselves, directly in the browser. Never ask for passwords, OTPs or tokens in chat. Never type, store, log or screenshot secrets; mask them in evidence. Never hard-code or log secrets.
- Honest labels: **Verified – executed in [sandbox | container | headless browser]**; **Verified – live env [name], user-assisted**; **Statically checked**; **Unverified (user must run)** with exact commands. Never claim a check that did not run.
- Fast reply: what changed, how it was checked, one-line assumptions.

## Lanes
Choose by blast radius, reversibility and verifiability, not technology layer.
| Lane | Work | Skip | Never skip |
|---|---|---|---|
| Fast | Presentational CSS/layout/colours, static HTML, copy, docs, comments, one-file renames, user-marked disposable scripts/spikes, test-only refactors | Traceability map, characterization tests, threat model, multi-pass review, formal assumption list (one line instead) | Checking the result, not breaking existing behaviour, no secrets, confirmation gate |
| Standard | Default | Nothing beyond Medium-depth trimming | Full flow at Medium depth |
| Strict | High triggers, IaC/cloud config, DB schema/data migrations, auth/session, payments, CI/CD, prod config | Nothing | Full flow at High depth |

- Escalate immediately when a higher-lane trigger appears. Never de-escalate silently.
- If unsure between two lanes, pick the higher. Honour an explicit user override.

## Where relaxing is a trap
- Regulated front-end: accessibility may be compliance; consent and disclosure copy is legal text.
- Global CSS or design-system tokens have a large blast radius.
- "Small" config changes (flags, timeouts, env vars) can be prod-impacting.
- A throwaway script run against prod data is not throwaway.
- Front-end is never automatically Fast.

## References (read when the lane calls for them)
- references/task-types-and-risk.md: task-type workflows, risk levels, discovery, clarify gate, planning, implementation.
- references/validation-ladder.md: levels L0-L5, profile per change type, handoff protocol, live-testing guardrails.
- references/regulated-domains.md: security and regulated-domain duties (High risk / Strict lane).
- references/delivery-and-review.md: test-fix loop, review, delivery format, feedback loop.

## Loop
```
Request -> Classify -> [LANE SELECT]
  FAST: glance -> change -> verify by profile (L1/L3) -> [CONFIRM if needed] -> short reply
        (higher-lane trigger -> ESCALATE, one-way)
  STD/STRICT: Discover (+runner inventory, baseline) -> [CLARIFY] -> Plan/criteria
        -> Implement -> static gates -> [VALIDATION LEVEL SELECT] -> execute, capture evidence
        -> [HANDOFF if credentials/MFA] -> [TEST-FIX LOOP, max 3 attempts]
        -> Review -> [CONFIRM] -> Deliver with labels -> feedback -> back to test loop
```
