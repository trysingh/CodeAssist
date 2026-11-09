"""v3.1: honest verification labels, one validation scale, ask-path facts. Pure modules only (no pydantic_settings / langchain)."""
import pytest

from services.reasoning import IntakeAgent, Lane, VerifierAgent
from services.reasoning.baseline import CHAT_FACTS, note_facts
from services.reasoning.crosscheck import (DASH, LABEL_BROWSER, LABEL_CONTAINER, LABEL_SANDBOX, LABEL_STATIC,
                                           LABEL_UNVERIFIED, DeliveryNotes, achieved_level, facts_from_sandbox,
                                           ladder_achieved, validation_summary, verification_labels)

# The /change request of transcript_change.txt, and the sandbox result it produced: only the boot smoke check ran.
REQUEST = ("give a more modern look to the landing page with color schemes for red-green color blindness friendly users. "
           "Add audible effects when starting and stopping services and ensure it is responsive")
BOOT_ONLY = {"mode": "boot", "ran": True, "ok": True, "backend": "docker", "passed": 9, "failed": 0, "preexisting": 0,
             "selected": 0, "note": "", "proof": {}, "not_run": None, "boot": {"ran": True, "ok": True, "steps": 9}}


def pytest_run(passed=5, failed=0, ok=True, **kw):
    return {"mode": "pytest", "ran": True, "ok": ok, "backend": "docker", "passed": passed, "failed": failed,
            "preexisting": 0, "selected": 2, "proof": {}, "boot": {}, **kw}


def change_facts(t, **extra):
    return {**facts_from_sandbox(t), "achieved_level": achieved_level(t), "static_checked": True,
            "files_changed": ["templates/dashboard.html"], "retrieval_mode": "agentic", **extra}


def notes_for(facts, request=REQUEST, intent="change"):
    return VerifierAgent().check(IntakeAgent().analyze(request, intent), "Change proposal saved as revision 20", facts)


# ---------------------------------------------------------------- bug 1: the boot-only transcript
def test_boot_only_result_does_not_set_tests_facts():
    f = facts_from_sandbox(BOOT_ONLY)
    assert f["tests_ran"] is False and f["tests_passed"] is None and f["tests_count"] == 0
    assert f["boot_checked"] is True and f["import_checked"] is False
    assert f["sandbox"] == "boot check passed" and f["sandbox_backend"] == "docker"
    assert f["achieved_level"] == "L1-boot-smoke"


def test_boot_only_label_is_exact_and_never_says_tests_passed():
    labels = verification_labels(change_facts(BOOT_ONLY), True)
    assert labels[0] == f"{LABEL_CONTAINER}: boot check only (no tests cover this change)"
    assert labels[1].startswith(LABEL_STATIC)
    md = notes_for(change_facts(BOOT_ONLY)).to_markdown()
    for banned in ("sandbox tests passed", "tests passed", "Tests ran in the sandbox and passed"):
        assert banned not in md, banned
    assert "boot check only (no tests cover this change)" in md


def test_real_tests_give_a_count_label():
    f = change_facts(pytest_run(passed=5))
    assert f["tests_ran"] and f["tests_passed"] is True and f["tests_count"] == 5
    assert verification_labels(f, True)[0] == f"{LABEL_CONTAINER}: 5 sandbox tests passed"
    assert verification_labels(change_facts(pytest_run(passed=1)), True)[0].endswith("1 sandbox test passed")
    local = change_facts({**pytest_run(passed=2), "backend": "local"})
    assert verification_labels(local, True)[0] == f"{LABEL_SANDBOX}: 2 sandbox tests passed"


def test_real_tests_with_a_boot_check_still_report_tests():
    t = pytest_run(passed=3, boot={"ran": True, "ok": True, "steps": 9})
    f = facts_from_sandbox(t)
    assert f["tests_ran"] and f["boot_checked"]
    assert verification_labels(change_facts(t), True) == [f"{LABEL_CONTAINER}: 3 sandbox tests passed"]


@pytest.mark.parametrize("t", [
    pytest_run(passed=0, failed=0),                                  # nothing collected / nothing passed
    {**BOOT_ONLY, "ok": False},                                     # boot failed
    {**BOOT_ONLY, "inconclusive": True},
    {"mode": "none", "ran": False, "ok": False, "not_run": "docker_unavailable"},
    {},
])
def test_no_pass_claim_without_a_test_count(t):
    f = change_facts(t)
    assert f["tests_ran"] is False and f["tests_passed"] is None
    md = notes_for(f).to_markdown()
    assert "tests passed" not in md.lower().replace("no tests", "")
    assert not any(l.startswith("Verified") and "test" in l and "only" not in l for l in verification_labels(f, True))


def test_failed_tests_are_reported_as_failed():
    f = change_facts(pytest_run(passed=2, failed=1, ok=False))
    assert f["tests_ran"] is True and f["tests_passed"] is False
    labels = verification_labels(f, True)
    assert labels[-1].startswith(LABEL_UNVERIFIED) and "failed" in labels[-1] and not labels[0].startswith("Verified")
    assert ladder_achieved(f).startswith("none")


def test_import_smoke_only_label():
    t = {"mode": "smoke", "ran": True, "ok": True, "backend": "docker", "passed": 4, "failed": 0}
    f = facts_from_sandbox(t)
    assert f["tests_ran"] is False and f["import_checked"] is True and f["achieved_level"] == "L1-import-smoke"
    assert verification_labels(f, True)[0] == f"{LABEL_CONTAINER}: import smoke check only (no tests cover this change)"


def test_legacy_pass_without_count_is_not_a_verified_label():
    labels = verification_labels({"tests_ran": True, "tests_passed": True, "static_checked": True}, True)
    assert not any(l.startswith("Verified") for l in labels) and "without a test count" in labels[-1]


def test_sandbox_that_did_not_run_gives_its_reason():
    f = change_facts({"mode": "none", "ran": False, "not_run": "docker_unavailable", "note": "Docker is not running"})
    assert "docker_unavailable" in verification_labels(f, True)[-1]
    steps = notes_for(f).validation_steps
    assert any(s.startswith("L1 tests: skipped") and "docker_unavailable" in s for s in steps)


# ---------------------------------------------------------------- bug 3: one scale, one summary, every lane
def test_transcript_summary_line_and_steps():
    n = notes_for(change_facts(BOOT_ONLY))
    assert n.lane == "standard"
    assert n.validation_summary == "Recommended: L1, L3 · Achieved: L1 (boot smoke)"
    assert n.validation_steps == [f"L1 tests: skipped {DASH} no test covers this change",
                                  f"L3 headless browser: skipped {DASH} no browser runner in this version"]
    md = n.to_markdown()
    assert "**Validation run**" in md and "Recommended: L1, L3 · Achieved: L1 (boot smoke)" in md
    assert "Level: L3" not in md and "Validation level (recommended)" not in md


def test_summary_when_tests_ran():
    n = notes_for(change_facts(pytest_run(passed=7, proof={"proves": 1})))
    assert "Achieved: L1 (tests, fail-to-pass)" in n.validation_summary
    assert n.validation_steps[0] == f"L1 tests: ran {DASH} 7 sandbox tests passed"


def test_each_recommended_level_has_exactly_one_line():
    s, lines = validation_summary("L0, L1, L2, L3, L4", change_facts(BOOT_ONLY))
    assert [l.split(" ")[0] for l in lines] == ["L0", "L1", "L2", "L3", "L4"]
    assert s.startswith("Recommended: L0, L1, L2, L3, L4")
    assert "partial" in lines[2] and "skipped" in lines[3] and "skipped" in lines[4]
    assert validation_summary("L3", {"browser_ran": True, "browser_ok": True})[1] == [f"L3 headless browser: ran {DASH} rendered and checked"]
    assert "Achieved: L3 (headless browser)" in validation_summary("L3", {"browser_ran": True, "browser_ok": True})[0]
    assert "Achieved: none" in validation_summary("L1", {})[0]


def test_fast_lane_notes_carry_the_summary_in_a_few_lines():
    n = notes_for(change_facts({"mode": "none", "ran": False}, files_changed=["styles.css"]),
                  request="change the button colour to blue in styles.css")
    assert n.lane == "fast"
    md = n.to_markdown()
    assert "Lane: fast. Recommended: L3 · Achieved: L0 (static)" in md
    assert f"- L3 headless browser: skipped {DASH} no browser runner in this version" in md
    assert len([l for l in md.splitlines() if l.strip()]) <= 6
    assert "Requirements cross-check" not in md


def test_strict_lane_lists_the_handoff_level():
    n = notes_for(change_facts(pytest_run(passed=3)), request="fix the SSO redirect in auth.py")
    assert n.lane == "strict" and "L4" in n.validation_level
    assert any(s.startswith("L4") and "skipped" in s for s in n.validation_steps)
    assert any("hands-on" in u for u in n.unverified)


def test_summary_headline_falls_back_for_hand_built_notes():
    md = DeliveryNotes(lane="standard", validation_level="L1").to_markdown()
    assert "Recommended: L1" in md
    assert "Lane" not in DeliveryNotes(assumptions=["a"]).to_markdown()


# ---------------------------------------------------------------- bug 5: ask turns that propose edits
def ask_facts(t):
    return {**facts_from_sandbox(t), "achieved_level": achieved_level(t), "edits_proposed": True, "static_checked": True}


def test_ask_that_proposed_edits_gets_lane_labels_and_summary():
    n = notes_for(ask_facts(BOOT_ONLY), request="do you think we can give a more modern look to the landing page with "
                  "color schemes for red-green color blindness friendly users. Add audible effects when starting and "
                  "stopping services and ensure it is responsive", intent="ask")
    assert n.lane == "standard"
    assert n.verification[0] == f"{LABEL_CONTAINER}: boot check only (no tests cover this change)"
    assert n.validation_summary.endswith("Achieved: L1 (boot smoke)")
    md = n.to_markdown()
    assert "Runtime behaviour (no tests executed)" not in md
    # v3.2 (4b): one statement of this, not two - kept as the limitation, no longer duplicated under Not verified
    assert "Runtime behaviour beyond the boot check (no test covers this change)" not in n.unverified
    assert "No test covers this change; only the boot check ran" in n.limitations


def test_ask_without_proposed_edits_is_unchanged():
    n = notes_for({"tests_ran": False}, request="how does the dashboard start a service?", intent="ask")
    assert n.lane == "" and n.verification == [] and n.validation_summary == ""
    assert "Lane" not in n.to_markdown() and "Validation run" not in n.to_markdown()


def test_change_with_nothing_executed_keeps_the_v2_wording():
    n = notes_for({"tests_ran": False, "files_changed": ["styles.css"]}, request="add retries to client.py")
    assert "Runtime behaviour (no tests executed)" in n.unverified and "Tests were not run for this change" in n.limitations


def test_note_facts_reaches_the_running_turn_and_is_a_noop_outside():
    note_facts(edits_proposed=True)                                  # no box: nothing happens, nothing raises
    box = {}
    tok = CHAT_FACTS.set(box)
    try:
        note_facts(**facts_from_sandbox(BOOT_ONLY), edits_proposed=True, static_checked=False)
        note_facts(static_checked=True, tests_passed=None)           # None is kept (it resets a fact)
    finally:
        CHAT_FACTS.reset(tok)
    assert box["boot_checked"] is True and box["edits_proposed"] and box["static_checked"] is True and box["tests_passed"] is None


# ---------------------------------------------------------------- wiring that cannot be imported here (static)
def _src(path):
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / path).read_bytes().decode("utf-8")


def test_controller_and_ask_agent_feed_facts_from_the_sandbox_result():
    ctl = _src("controllers/engine_controller.py")
    assert "facts_from_sandbox(res.tests or {})" in ctl and "achieved_level=res.verification" in ctl
    assert "tests_ran=ran" not in ctl                                 # the v3 line that read "ran" as "tests ran"
    assert "_CHAT_FACTS = CHAT_FACTS" in ctl
    agent = _src("services/code_agent.py")
    assert agent.count("note_facts(") >= 4 and "facts_from_sandbox(t)" in agent and "guidance_block()" in agent
    assert '"edits_proposed": True' in agent
