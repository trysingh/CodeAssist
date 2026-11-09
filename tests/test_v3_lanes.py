"""v3: lanes, change types, validation levels, baseline lane mapping, delivery-note labels.
Pure modules only (services.reasoning: stdlib + pydantic): no pydantic_settings / langchain needed."""
import asyncio

import pytest

from services import action_log as al
from services.reasoning import (BaselinePolicy, ChangeType, DeliveryNotes, IntakeAgent, IntakeBrief, Lane,
                                ReasoningSupervisor, Risk, VerifierAgent, flag, guidance_block, policy_for, policy_scope)
from services.reasoning.baseline import GATE_ORDER
from services.reasoning.crosscheck import (LABEL_BROWSER, LABEL_CONTAINER, LABEL_LIVE, LABEL_SANDBOX, LABEL_STATIC,
                                           LABEL_UNVERIFIED, verification_labels)
from services.reasoning.intake import classify_risk, detect_change_type, recommend_validation_level

ORDER = {Lane.FAST: 0, Lane.STANDARD: 1, Lane.STRICT: 2}


def brief(q, intent="change", override="", **kw):
    return IntakeAgent(lane_override=override).analyze(q, intent, **kw)


# ---------------------------------------------------------------- lane classification
@pytest.mark.parametrize("q,lane", [
    ("change the button colour to blue in styles.css", Lane.FAST),
    ("make the dashboard layout responsive in layout.scss", Lane.FAST),
    ("fix typo in README.md", Lane.FAST),
    ("add a docstring to the helper in utils.py", Lane.FAST),
    ("rename variable foo_bar to count in utils.py", Lane.FAST),
    ("write a disposable script to dump the sample fixtures", Lane.FAST),
    ("use innerHTML to render the user comment in app.js", Lane.STRICT),
    ("store the token in localStorage", Lane.STRICT),
    ("load the widget from a third-party script on a CDN", Lane.STRICT),
    ("call eval on the config string in app.js", Lane.STRICT),
    ("add a migration that adds a column to orders", Lane.STRICT),
    ("update the terraform for the s3 bucket", Lane.STRICT),
    ("change the GitHub Actions workflow to deploy on tag", Lane.STRICT),
    ("change the password hashing in auth.py", Lane.STRICT),
    ("add a payment retry in checkout.py", Lane.STRICT),
    ("add retries to client.py", Lane.STANDARD),
    ("add a dropdown to the page", Lane.STANDARD),
    ("optimise this SQL query in report.sql", Lane.STANDARD),
])
def test_lane_cases(q, lane):
    assert brief(q).lane == lane, brief(q).lane_reasons


@pytest.mark.parametrize("q", [
    "update the global design tokens in tokens.css",
    "change the colour variables in :root of base.css",
    "tweak the global css reset",
    "change the consent disclosure text on the signup page",
    "update the privacy notice copy in footer.html",
    "bump the feature flag default in the settings page text",
    "change the request timeout in client.js",
    "set the env var default in the README",
    "write a throwaway script that sets the timeout",
])
def test_relaxing_is_a_trap_never_fast(q):
    assert ORDER[brief(q).lane] >= ORDER[Lane.STANDARD], brief(q).lane_reasons


def test_design_token_is_styling_not_a_secret():
    q = "update the global design tokens in tokens.css"
    assert classify_risk(q)[0] == Risk.LOW
    assert classify_risk("rotate the auth token in client.py")[0] == Risk.HIGH


def test_script_against_prod_data_is_strict_even_when_disposable():
    for q in ("run a one-off script against prod data", "throwaway script on the live data",
              "disposable script that reads customer data"):
        assert brief(q).lane == Lane.STRICT


def test_frontend_is_never_automatically_fast():
    for q in ("update the frontend", "tweak the UI behaviour", "fix the click handler in menu.js"):
        assert brief(q).lane != Lane.FAST
    b = brief("make the header sticky in site.css")
    assert b.change_type == ChangeType.FE_PRESENTATIONAL and b.lane == Lane.FAST


def test_high_risk_means_strict_and_enterprise_means_at_least_standard():
    b = brief("fix the claims payout calculation in payout.py")
    assert b.risk == Risk.HIGH and b.lane == Lane.STRICT
    assert brief("fix the colour in the core banking portal styles.css").lane != Lane.FAST    # enterprise context raises the bar


@pytest.mark.parametrize("q,ctype", [
    ("change the colour in styles.css", ChangeType.FE_PRESENTATIONAL),
    ("add a click handler in menu.js", ChangeType.FE_BEHAVIOURAL),
    ("use innerHTML in render.js", ChangeType.FE_SECURITY),
    ("update the README", ChangeType.DOCS),
    ("rewrite the SQL query in report.sql", ChangeType.SQL),
    ("write an alembic migration", ChangeType.MIGRATION),
    ("edit main.tf", ChangeType.IAC_CONFIG),
    ("write a script cleanup.sh", ChangeType.SCRIPT),
    ("add retries to client.py", ChangeType.BACKEND),
])
def test_change_types(q, ctype):
    assert detect_change_type(q)[0] == ctype


def test_change_type_uses_repo_files_only_when_the_request_names_them():
    files = ["web/styles.css", "src/app.py", "docs/README.md"]
    assert detect_change_type("tweak styles.css spacing", files)[0] == ChangeType.FE_PRESENTATIONAL
    assert detect_change_type("improve app.py logging", files)[0] == ChangeType.BACKEND      # a .css file in the repo is not a signal


def test_ask_intent_still_gets_a_lane_but_no_validation_level():
    b = brief("explain how the login works", "ask")
    assert b.lane == Lane.STRICT and b.validation_level == ""


# ---------------------------------------------------------------- overrides
def test_override_phrase_and_setting():
    assert brief("lane: strict change the colour in styles.css").lane == Lane.STRICT          # raising is always honoured
    assert brief("use the standard lane: change the colour in styles.css").lane == Lane.STANDARD
    assert brief("change the colour in styles.css", override="standard").lane == Lane.STANDARD
    assert brief("change the colour in styles.css", override="bogus").lane == Lane.FAST       # unknown value ignored
    b = brief("lane: standard change the colour in styles.css", override="strict")
    assert b.lane == Lane.STANDARD                                                            # the request's own words win


def test_override_can_lower_below_a_floor_but_not_below_strict_triggers():
    b = brief("lane: fast update the global design tokens in tokens.css")
    assert b.lane == Lane.FAST and any("override" in r for r in b.lane_reasons)
    for q in ("lane: fast change the password hashing in auth.py", "lane: fast add a migration for orders",
              "lane: fast use innerHTML in app.js", "lane: fast run a script against prod data",
              "lane: fast fix the claims payout in payout.py"):
        b = brief(q)
        assert b.lane == Lane.STRICT, q
        assert any("ignored" in r for r in b.lane_reasons)
    assert brief("change the password hashing in auth.py", override="fast").lane == Lane.STRICT


# ---------------------------------------------------------------- validation level + handoff
def test_validation_levels_follow_the_ladder():
    assert brief("change the colour in styles.css").validation_level == "L3"
    assert brief("add a click handler in menu.js").validation_level == "L1, L3"
    assert brief("fix typo in README.md").validation_level == "L1"
    assert brief("add a migration for orders").validation_level == "L2"
    assert brief("edit main.tf").validation_level == "L2"
    assert brief("add retries to client.py").validation_level == "L1"
    assert recommend_validation_level(Lane.STRICT, ChangeType.BACKEND) == "L2"
    assert "L5" not in recommend_validation_level(Lane.STRICT, ChangeType.BACKEND, True)


def test_needs_handoff():
    for q in ("fix the SSO redirect in auth.py", "add MFA prompt handling in login.js", "OTP entry page in otp.js",
              "the CAPTCHA widget in signup.js", "fix login in auth.py"):
        assert brief(q).needs_handoff, q
    assert not brief("add retries to client.py").needs_handoff
    b = brief("fix the SSO redirect in auth.py")
    assert b.lane == Lane.STRICT and b.validation_level.endswith("L4")
    assert not brief("fix the SSO redirect in auth.py").validation_level.startswith("L4")


# ---------------------------------------------------------------- baseline: lanes only raise
GATES = ["off", "smoke", "strict"]


def _b(lane, risk=Risk.LOW):
    return IntakeBrief(intent="change", lane=lane, risk=risk)


def _defaults(gate, review, tf, at):
    return {"change_gate": gate, "change_review": review, "change_tests_first": tf, "change_author_tests": at}


@pytest.mark.parametrize("lane", list(Lane) + [None])
@pytest.mark.parametrize("risk", list(Risk))
@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("review,tf,at", [(False, False, False), (True, False, True), (True, True, True)])
def test_policy_only_raises(lane, risk, gate, review, tf, at):
    d = _defaults(gate, review, tf, at)
    ov = policy_for(_b(lane, risk), d).overrides
    if "change_gate" in ov:
        assert GATE_ORDER[ov["change_gate"]] > GATE_ORDER[gate]
    for k in ("change_review", "change_tests_first", "change_author_tests"):
        if k in ov:
            assert ov[k] is True and d[k] is False            # an override only ever switches a stage ON
    assert set(ov) <= {"change_gate", "change_review", "change_tests_first", "change_author_tests", "change_max_attempts"}
    with policy_scope(ov):                                      # ... and the overlay itself cannot lower a setting
        assert GATE_ORDER[flag("change_gate", gate)] >= GATE_ORDER[gate]
        assert all(flag(k, d[k]) or not d[k] for k in ("change_review", "change_tests_first", "change_author_tests"))


def test_lane_never_raises_when_settings_are_already_strict():
    d = _defaults("strict", True, True, True)
    for lane in Lane:
        assert policy_for(_b(lane), d).overrides == {}


def test_lane_mapping_onto_knobs():
    off = _defaults("off", False, False, False)
    assert policy_for(_b(Lane.FAST), off).overrides == {}
    assert policy_for(_b(Lane.STANDARD), off).overrides == {"change_review": True, "change_gate": "smoke"}
    p = policy_for(_b(Lane.STRICT), off)
    assert p.depth == "strict" and p.overrides == {"change_gate": "strict", "change_review": True,
                                                   "change_tests_first": True, "change_author_tests": True}
    assert policy_for(_b(Lane.FAST, Risk.HIGH), off).depth == "strict"          # a Fast lane does not lower a High risk
    assert policy_for(_b(Lane.STRICT, Risk.LOW), off).depth == "strict"         # ... and a Strict lane raises a Low one
    assert policy_for(_b(None, Risk.LOW), off).overrides == {}                   # unclassified (fallback) brief: v2 behaviour


# ---------------------------------------------------------------- guidance seam
def test_guidance_block_is_scoped_to_the_request():
    assert guidance_block() == ""
    with policy_scope({}, guidance="SKILL TEXT"):
        g = guidance_block()
        assert "SKILL TEXT" in g and "OUTPUT FORMAT" in g
    assert guidance_block() == ""
    with policy_scope({"change_gate": "smoke"}):
        assert guidance_block() == ""


def test_overlay_and_guidance_isolated_across_concurrent_tasks():
    async def one(text):
        with policy_scope({}, guidance=text):
            await asyncio.sleep(0.01)
            return guidance_block()

    async def main():
        return await asyncio.gather(one("A-TEXT"), one(""), one("C-TEXT"))
    a, b, c = asyncio.run(main())
    assert "A-TEXT" in a and b == "" and "C-TEXT" in c and "A-TEXT" not in c


def _sup(provider):
    return ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {"change_gate": "off"}, True,
                               guidance_provider=provider)


def test_supervisor_passes_guidance_to_execute_and_provider_gets_the_brief():
    seen = {}

    def provider(b, p):
        seen["lane"] = b.lane
        return "GUIDE"

    async def execute(b, p):
        seen["block"] = guidance_block()
        return "ANSWER", {}
    asyncio.run(_sup(provider).run(query="change the colour in styles.css", intent="change", repo_files=[], execute=execute))
    assert seen["lane"] == Lane.FAST and "GUIDE" in seen["block"]
    assert guidance_block() == ""


def test_supervisor_survives_a_failing_guidance_provider():
    def boom(b, p):
        raise RuntimeError("pack exploded")

    async def execute(b, p):
        return "ANSWER", {"block": guidance_block()}
    al.ring.clear()
    out = asyncio.run(_sup(boom).run(query="add retries to client.py", intent="change", repo_files=[], execute=execute))
    assert out.startswith("ANSWER") and "### Delivery notes" in out
    assert any(e["status"] == "fail" and "pack exploded" in e["detail"] for e in al.recent())


def test_supervisor_without_provider_matches_v2():
    seen = {}

    async def execute(b, p):
        seen["block"] = guidance_block()
        return "ANSWER", {}
    asyncio.run(ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {}).run(
        query="add retries to client.py", intent="change", repo_files=[], execute=execute))
    assert seen["block"] == ""


# ---------------------------------------------------------------- delivery notes: labels
def _facts(**kw):
    base = {"tests_ran": False, "tests_passed": None, "sandbox": None, "files_changed": [], "retrieval_mode": "agentic",
            "retrieval_confidence": 1.0, "vectorize_job": None}
    base.update(kw)
    return base


def test_labels_from_sandbox_facts():
    ok = _facts(tests_ran=True, tests_passed=True, tests_count=3, sandbox="passed", files_changed=["a.py"])   # v3.1: a pass needs a test count
    assert verification_labels({**ok, "sandbox_backend": "docker"}, True)[0].startswith(LABEL_CONTAINER)
    assert verification_labels({**ok, "sandbox_backend": "local"}, True)[0].startswith(LABEL_SANDBOX)
    assert verification_labels(ok, True)[0].startswith(LABEL_SANDBOX)


def test_labels_never_claim_what_did_not_run():
    for facts in (_facts(), _facts(tests_ran=False, tests_passed=True), _facts(tests_ran=True, tests_passed=None),
                  _facts(tests_ran=True, tests_passed=False), {}):
        labels = verification_labels(facts, True)
        assert labels and all(l.startswith((LABEL_UNVERIFIED, LABEL_STATIC)) for l in labels), labels
        assert not any(l.startswith("Verified") for l in labels)
    assert any("failed" in l for l in verification_labels(_facts(tests_ran=True, tests_passed=False), True))
    assert verification_labels(_facts(), False) == []                               # ask turns carry no labels


def test_static_label_only_when_the_checks_ran():
    labels = verification_labels(_facts(static_checked=True), True)
    assert labels[0].startswith(LABEL_STATIC) and labels[1].startswith(LABEL_UNVERIFIED)
    assert not any(l.startswith(LABEL_STATIC) for l in verification_labels(_facts(), True))
    ran = verification_labels(_facts(static_checked=True, tests_ran=True, tests_passed=True, tests_count=2), True)   # v3.1: needs a count
    assert len(ran) == 1 and ran[0].startswith("Verified")


def test_browser_and_live_labels_need_every_fact():
    assert verification_labels(_facts(browser_ran=True, browser_ok=True), True)[0].startswith(LABEL_BROWSER)
    assert not any(l.startswith(LABEL_BROWSER) for l in verification_labels(_facts(browser_ran=True), True))
    live = verification_labels(_facts(live_env="staging", live_ok=True, user_assisted=True), True)[0]
    assert live.startswith(LABEL_LIVE) and "staging" in live and "user-assisted" in live
    assert not any(l.startswith(LABEL_LIVE) for l in verification_labels(_facts(live_env="staging", live_ok=True), True))


def test_labels_use_the_documented_wording():
    assert LABEL_SANDBOX == "Verified – executed in sandbox"
    assert LABEL_CONTAINER == "Verified – executed in container"
    assert LABEL_BROWSER == "Verified – executed in headless browser"
    assert LABEL_LIVE == "Verified – live env"
    assert (LABEL_STATIC, LABEL_UNVERIFIED) == ("Statically checked", "Unverified (user must run)")


def test_verifier_fills_lane_level_and_labels_for_changes():
    b = brief("add retries to client.py")
    n = VerifierAgent().check(b, "Changed client.py", _facts(tests_ran=True, tests_passed=True, tests_count=4,
                                                             files_changed=["client.py"], sandbox="passed", sandbox_backend="docker"))   # v3.1: count
    assert n.lane == "standard" and n.validation_level == "L1" and n.verification[0].startswith(LABEL_CONTAINER)
    md = n.to_markdown()
    assert "**Lane:** standard" in md and "Recommended: L1" in md      # v3.1: one validation-run summary and "**Verification**" in md
    assert "Requirements cross-check" in md


def test_verifier_ask_turn_has_no_lane_or_labels():
    n = VerifierAgent().check(brief("explain how login works in auth.py", "ask"), "answer", _facts())
    assert n.lane == "" and n.validation_level == "" and n.verification == []
    md = n.to_markdown()
    assert "Lane" not in md and "Verification**" not in md


def test_handoff_adds_an_unverified_item_for_changes_only():
    n = VerifierAgent().check(brief("fix the SSO redirect in auth.py"), "x", _facts())
    assert any("hands-on" in u for u in n.unverified)
    n = VerifierAgent().check(brief("explain the SSO redirect in auth.py", "ask"), "x", _facts())
    assert not any("hands-on" in u for u in n.unverified)


def test_fast_lane_notes_are_very_short():
    b = brief("change the button colour to blue in styles.css")
    assert b.lane == Lane.FAST
    n = VerifierAgent().check(b, "Changed styles.css", _facts(files_changed=["styles.css"], static_checked=True))
    md = n.to_markdown()
    assert md.startswith("### Delivery notes") and "Lane: fast" in md
    assert LABEL_STATIC in md
    assert len([l for l in md.splitlines() if l.strip()]) <= 6 and len(md) < 600
    assert "Requirements cross-check" not in md and "Professional" not in md
    full = VerifierAgent().check(brief("add retries to client.py"), "Changed client.py", _facts(files_changed=["client.py"]))
    assert len(full.to_markdown()) > 2 * len(md)


def test_fast_lane_notes_still_report_failures():
    b = brief("change the button colour to blue in styles.css")
    n = VerifierAgent().check(b, "x", _facts(tests_ran=True, tests_passed=False, files_changed=["styles.css"]))
    md = n.to_markdown()
    assert "FAILED" in md and LABEL_UNVERIFIED in md and "Verified –" not in md


def test_notes_without_a_lane_render_as_in_v2():
    n = DeliveryNotes(assumptions=["A1"], limitations=["L1"], verified=["V1"], unverified=["U1"])
    md = n.to_markdown()
    assert "Lane" not in md and "Verification**" not in md and "**Assumptions**" in md
