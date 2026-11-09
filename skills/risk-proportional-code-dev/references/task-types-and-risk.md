# Task types, risk and the work before the change

## Classify
Before writing code, identify: task type, risk, ambiguity, primary goal, success criteria, constraints, implied non-functionals (correctness, readability, maintainability, reasonable performance, basic security).

| Task type | Workflow |
|---|---|
| Bug fix | Reproduce with a failing test, find root cause, minimal diff, keep regression test |
| New feature | Acceptance criteria, contracts, implementation, tests from criteria |
| Refactor | Characterization tests first, behaviour unchanged |
| Code review | Read-only; findings by severity; do not edit unless asked |
| Migration | Rollback plan, dry-run on a copy, staged execution |
| Spike | Time-boxed; label not production-ready |

Risk: the highest matching level wins.
| Level | Triggers | Verification depth |
|---|---|---|
| Low | Local scripts, internal tooling, docs, prototypes, no sensitive data | Unit tests on core logic; lint/format |
| Medium | Shared libs, business logic, external API calls, non-prod schema changes, user-facing behaviour | Low + integration, edge/failure tests, type check, self-review |
| High | Money movement, PII/PHI, auth, prod data, irreversible ops, regulatory scope | Medium + threat model, SAST, dependency audit, abuse tests, audit-trail checks, independent + human review |

## Discover context (Standard/Strict; Fast: a glance)
- Read the relevant code, conventions (naming, errors, logging, config, layout), build/test setup, CI config and existing tests for the area.
- Check the real environment: runtimes and versions, package manager, network access. Assume the network may be restricted.
- Inventory runners: Docker, browsers (headless or the user's), dev/staging URLs, test accounts, and whether the user is present for handoffs.
- Run the existing test suite for a baseline before changing anything. Record pre-existing failures.

## Clarify gate
Decide by cost of a wrong guess x ambiguity.
- Ask up front, in one batched round with a proposed default per question, when the work is expensive or irreversible and ambiguity is high, or when the request contradicts itself.
- Otherwise proceed and state assumptions explicitly.
- Surface conflicting requirements; never silently pick one.

## Plan (Standard/Strict)
- Break the request into a few outcome-oriented tasks. Decide contracts and data shapes first, then approach, then edge cases and failure modes. Keep it light unless the problem is large or high-risk.
- Turn each requirement into testable acceptance criteria. Keep a lightweight map: requirement, criterion, implementation, test. Use it for the final cross-check.
- Baseline always: correct on expected inputs, clear error handling, readable structure, no injection/auth bypass/secret leakage, reasonable performance.
- Escalate with risk: input validation, structured logs/metrics, idempotency, back-pressure, consistency model, failure isolation, cost awareness, auditability.
- Enterprise: explicit error types, structured logging, externalized config, observability hooks, backward compatibility. Do not add ceremony the problem does not justify.

## Implement
- Order: contracts, core logic, edge cases and errors, hardening, tests. Mandatory failing-test-first for bug fixes.
- Derive test cases from requirements, acceptance criteria and up-front edge cases, never from your own code.
- Make the minimal diff. No unrequested refactors or style churn. Match existing conventions. Mention adjacent issues you notice; do not fix them silently.
- Prefer stdlib, then existing dependencies. Verify that every API and package exists and matches the installed version. Pin versions. Check licences in enterprise contexts.
- Run the project's own linter, formatter, type checker and SAST, plus a dependency-vulnerability check when dependencies change or risk is Medium+.
