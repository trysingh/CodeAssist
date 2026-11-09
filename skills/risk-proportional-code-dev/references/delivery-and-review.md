# Test-fix loop, review and delivery

The never-weaken / never-special-case test rules and the verification labels are in SKILL.md.

## Test-fix loop
- Maximum 3 attempts on the same failure; then step back, re-diagnose from first principles and report.
- Separate environment failures (missing package, no network, wrong runtime, flaky timing, baseline failure) from code failures; do not change code to work around the environment.

## Review
- Self-review to find defects: try to break it (empty, null, huge, concurrent, malicious input), re-check each acceptance criterion, look for unhandled errors and leftover debug code.
- High risk: recommend independent review (not the author) plus human review. Do the final cross-check with the requirement map.

## Deliver
- Standard/Strict reply: what changed (files/diff) | how to run | verification by label | assumptions | limitations/risks | next steps incl. adjacent issues. Do not recap the process.

## Feedback loop
Fold feedback in with a minimal diff, re-run affected tests (full suite if shared code changed), update the requirement map if requirements changed, and give a short change note.
