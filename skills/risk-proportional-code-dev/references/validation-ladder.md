# Validation ladder

Use the lowest level that gives real evidence for the acceptance criteria. Go up only when a lower level cannot prove the behaviour. "Available" means already present or one command away; do not spend longer building an environment than the change warrants, and say so if that is the trade-off.

## Levels
| Level | Environment | Use when | Evidence |
|---|---|---|---|
| L0 | Static: lint, types, review | Nothing can run | Tool output |
| L1 | Local sandbox | Unit/integration tests | Commands, reports |
| L2 | Isolated container, pinned images, ephemeral DB/queue/mocks with synthetic data, egress off unless needed | Integration, migrations (forward + rollback), services | Logs, reports |
| L3 | Isolated headless browser (Playwright/Puppeteer) | FE rendering at breakpoints, flows, console errors, a11y (e.g. axe), snapshots | Screenshots, console, a11y report |
| L4 | Live browser profile with user in the loop, dev/staging/UAT, read-only by default | Real SSO/MFA, third-party integrations, env-specific behaviour | Screenshots, HAR/console (secrets masked) |
| L5 | Production: not a test environment | Read-only smoke check with explicit approval; no writes, no real transactions | Output + approval |

Fast usually L1/L3 (quick render); Standard L1-L3; Strict L2-L4 where available. If no runner exists at the needed level, use the highest available and list the unverified items with exact run steps.

## Profile by change type
| Change type | Lane | Check | Level |
|---|---|---|---|
| FE presentational (CSS, HTML, styling, show/hide) | Fast | Render and look; no unit tests. Breakpoints, light/dark, neighbouring components (specificity, global selectors), cross-browser note for non-standard features, existing visual snapshots, a11y basics (contrast, focus, not colour-only) | L3 |
| FE behavioural JS | Standard | Pure functions unit-tested; component/interaction tests if a framework exists; console clean; keyboard and ARIA. Respect browser autoplay / user-gesture rules (create or resume audio on the gesture); reuse long-lived resources (one AudioContext, not one per action); give feedback after the outcome is known (success or failure), not before; make non-essential effects (sound, motion) user-controllable and honour a stored preference; never convey status by colour alone (add text or an icon) | L1, L3 |
| FE security-relevant (innerHTML/eval/document.write, URL handling, postMessage, token/session storage, cookies, CSP, third-party scripts/SRI, payment/KYC forms, user-data analytics) | Strict | XSS review; prefer textContent and safe templating; no secrets/PII in client code or logs; server re-validates | L3, + L4 if SSO |
| Docs / README | Fast | Check links; run samples | L1 |
| SQL | Standard | Run on sample/fixture; query plan for large tables | L1/L2 |
| Migrations | Strict | Forward + rollback + dry-run on a copy | L2 |
| IaC / config | Strict | plan/diff, policy checks; never apply without confirmation | L2 |
| Scripts / notebooks | Fast if disposable, else Standard | Run on sample data | L1 |

## Handoff protocol
For L4, or any credentials, MFA/OTP, CAPTCHA, consent prompt, manual step:
- Pause and state in one or two lines exactly what is needed and why.
- The user enters credentials/MFA themselves, directly in the browser. Never ask for passwords, OTPs or tokens in chat. Never type, store, log or screenshot secrets; mask them in evidence.
- Prefer non-prod, least-privilege test accounts; avoid personal or privileged ones.
- Wait for the user's confirmation; resume from a stated checkpoint.
- If the user is unavailable, stop, report what was verified, leave exact steps.
- Ask before using the user's real browser with existing sign-ins; prefer a separate profile.

## Live-testing guardrails
- Isolation by default; synthetic or masked data only, never real customer PII.
- No real payments, emails, SMS or notifications; use sandbox/test modes of payment and messaging providers.
- No destructive actions without the confirmation gate.
- Respect client security policy: in banks/insurers, pointing tools at client systems or using corporate sessions may need approval; ask if unsure.
- Watch cost and rate limits; time-box environment setup; clean up containers, data and sessions; record what ran against which environment.
