import asyncio

import pytest

from services import action_log as al
from services.reasoning import (BaselinePolicy, DeliveryNotes, IntakeAgent, IntakeBrief, ReasoningSupervisor, Risk,
                                VerifierAgent, flag, policy_for, policy_scope)
from services.reasoning.intake import split_clauses

AG = IntakeAgent()


def brief(q, intent="change", **kw):
    return AG.analyze(q, intent, **kw)


# ---------------------------------------------------------------- risk
@pytest.mark.parametrize("q", [
    "Add KYC check to onboarding in onboarding/kyc.py",
    "change the password hashing in auth.py",
    "update AML screening for each transaction",
    "store PII in the customer table",
    "fix claims payout calculation for policyholder refunds",
    "write a migration that drops a column in prod",
    "ledger posting must be idempotent",
    "PCI scope: stop logging card numbers",
])
def test_high_risk(q):
    assert brief(q).risk == Risk.HIGH


@pytest.mark.parametrize("q", [
    "add an endpoint to the API in routes.py",
    "refactor the schema loader in db_utils.py",
    "improve performance of the cache in cache.py",
])
def test_medium_risk(q):
    assert brief(q).risk == Risk.MEDIUM


def test_low_risk_and_baseline_nfrs():
    b = brief("rename variable foo_bar to count in utils.py")
    assert b.risk == Risk.LOW and not b.enterprise
    assert any("Correctness" in n for n in b.nfrs) and len(b.nfrs) >= 5


def test_no_substring_false_positive():
    assert brief("explain the product catalog in catalog.py", "ask").risk == Risk.LOW   # 'prod' inside 'product'
    assert brief("tokenize the string in lexer.py", "ask").risk == Risk.LOW


def test_enterprise_detection_and_nfr():
    b = brief("Add audit logging for the core banking ledger posting in ledger.py")
    assert b.enterprise and b.risk == Risk.HIGH
    assert any("Regulatory" in n for n in b.nfrs)
    assert not brief("rename x to y in a.py").enterprise


def test_enterprise_from_repo_files():
    b = brief("tidy up helpers in helpers.py", repo_files=["src/claims/helpers.py", "src/aml/x.py"])
    assert b.enterprise and b.risk == Risk.MEDIUM


# ---------------------------------------------------------------- ambiguity
def test_ambiguity_fix_it():
    b = brief("fix it")
    assert b.ambiguity >= 0.7
    assert any("pronoun" in r for r in b.ambiguity_reasons)
    assert 1 <= len(b.clarifying_questions) <= 3


def test_ambiguity_vague_words():
    b = brief("make the app do something better somehow etc")
    assert b.ambiguity > 0.3 and any("vague" in r for r in b.ambiguity_reasons)
    assert b.assumptions


def test_clear_request_low_ambiguity():
    b = brief("In services/tasks.py, make plan_tasks return an empty list when the input is empty, and add a unit test")
    assert b.ambiguity <= 0.3 and b.clarifying_questions == [] and b.assumptions == []


def test_ambiguity_bounds_and_questions_cap():
    b = brief("it")
    assert 0 <= b.ambiguity <= 1 and len(b.clarifying_questions) <= 3


def test_outcome_tasks_split():
    assert split_clauses("add logging, rename foo and delete bar.py; update docs") == [
        "add logging", "rename foo", "delete bar.py", "update docs"]
    assert split_clauses("1. add a field\n2. validate it\n3) write tests") == ["add a field", "validate it", "write tests"]
    assert split_clauses("research and development notes") == ["research and development notes"]
    assert split_clauses("") == []


def test_success_criteria_from_ask():
    b = brief("add retries to client.py and log each failure")
    assert any("retries" in c for c in b.success_criteria)
    assert any("log each failure" in c for c in b.success_criteria)
    assert brief("how does login work in auth.py", "ask").success_criteria


def test_constraints_extracted():
    b = brief("change parse() in a.py without breaking the public API, keep python 3.9 support")
    assert any(c.lower().startswith("without") for c in b.constraints)


def test_llm_merge_valid_and_invalid():
    ok = IntakeAgent(llm_json=lambda p: {"goal": "G", "success_criteria": ["extra crit"], "risk": "high",
                                         "ambiguity": 0.9, "clarifying_questions": ["which one?"]})
    b = ok.analyze("rename x to y in a.py", "change")
    assert b.source == "llm" and b.goal == "G" and "extra crit" in b.success_criteria
    assert b.risk == Risk.HIGH and b.ambiguity == 0.9 and "which one?" in b.clarifying_questions
    for bad in (lambda p: "garbage", lambda p: None, lambda p: {"risk": "banana", "ambiguity": "x", "goal": 5}):
        b2 = IntakeAgent(llm_json=bad).analyze("rename x to y in a.py", "change")
        assert b2.source == "heuristic" and b2.risk == Risk.LOW
    def boom(p): raise RuntimeError
    assert IntakeAgent(llm_json=boom).analyze("rename x to y in a.py", "change").source == "heuristic"


def test_llm_cannot_lower_risk():
    b = IntakeAgent(llm_json=lambda p: {"risk": "low", "ambiguity": 0.0}).analyze("change password flow in auth.py", "change")
    assert b.risk == Risk.HIGH


# ---------------------------------------------------------------- policy
def _b(risk, amb=0.0, intent="change", ent=False):
    return IntakeBrief(intent=intent, risk=risk, ambiguity=amb, enterprise=ent)


def test_policy_low_no_overrides():
    p = policy_for(_b(Risk.LOW), {"change_gate": "off", "change_review": False})
    assert p.depth == "baseline" and p.overrides == {} and not p.clarify_first and p.checklist


def test_policy_medium_raises():
    p = policy_for(_b(Risk.MEDIUM), {"change_gate": "off", "change_review": False})
    assert p.depth == "elevated" and p.overrides == {"change_review": True, "change_gate": "smoke"}


def test_policy_medium_never_lowers_strict():
    p = policy_for(_b(Risk.MEDIUM), {"change_gate": "strict", "change_review": True})
    assert p.overrides == {}


def test_policy_high_strict():
    p = policy_for(_b(Risk.HIGH), {"change_gate": "off", "change_review": False, "change_tests_first": False,
                                   "change_author_tests": True})
    assert p.depth == "strict"
    assert p.overrides["change_gate"] == "strict" and p.overrides["change_review"] and p.overrides["change_tests_first"]
    assert set(p.overrides) <= {"change_gate", "change_review", "change_tests_first", "change_author_tests",
                                "change_max_attempts"}


def test_policy_high_with_settings_already_strict():
    p = policy_for(_b(Risk.HIGH), {"change_gate": "strict", "change_review": True, "change_tests_first": True,
                                   "change_author_tests": True})
    assert p.overrides == {}


def test_clarify_first_rule():
    assert policy_for(_b(Risk.HIGH, 0.7), {}).clarify_first
    assert not policy_for(_b(Risk.HIGH, 0.69), {}).clarify_first
    assert not policy_for(_b(Risk.MEDIUM, 0.9), {}).clarify_first
    assert not policy_for(_b(Risk.HIGH, 0.9, intent="ask"), {}).clarify_first


def test_flag_default_without_scope():
    assert flag("change_gate", "off") == "off" and flag("change_review", False) is False


def test_flag_overlay_only_raises():
    with policy_scope({"change_gate": "smoke", "change_review": True, "change_max_attempts": 2, "bogus": 1}):
        assert flag("change_gate", "off") == "smoke"
        assert flag("change_gate", "strict") == "strict"       # never lowers
        assert flag("change_review", False) is True
        assert flag("change_max_attempts", 3) == 3
        assert flag("bogus", "d") == "d"                        # unknown keys dropped
    assert flag("change_gate", "off") == "off"


def test_overlay_isolated_across_concurrent_tasks():
    async def one(name, ov):
        with policy_scope(ov):
            await asyncio.sleep(0.01)
            a = flag("change_gate", "off")
            await asyncio.sleep(0.01)
            return name, a, flag("change_review", False)

    async def main():
        return await asyncio.gather(one("a", {"change_gate": "strict", "change_review": True}),
                                    one("b", {}), one("c", {"change_gate": "smoke"}))
    res = asyncio.run(main())
    assert res == [("a", "strict", True), ("b", "off", False), ("c", "smoke", False)]
    assert flag("change_gate", "off") == "off"


# ---------------------------------------------------------------- verifier
def _facts(**kw):
    base = {"tests_ran": False, "tests_passed": None, "sandbox": None, "files_changed": [],
            "retrieval_mode": "agentic", "retrieval_confidence": 1.0, "vectorize_job": None}
    base.update(kw)
    return base


def test_verifier_never_claims_tests_passed_without_facts():
    b = brief("add retries to client.py")
    for facts in (_facts(), _facts(tests_ran=True, tests_passed=None), _facts(tests_ran=True, tests_passed=False),
                  _facts(tests_ran=False, tests_passed=True), {}):
        n = VerifierAgent().check(b, "I added tests and they passed. All good, tests passed!", facts)
        assert not any("passed" in v.lower() for v in n.verified)
        assert "tests passed" not in n.to_markdown().lower().replace("tests ran in the sandbox and passed", "")
    n = VerifierAgent().check(b, "done", _facts())
    assert any("not run" in l.lower() for l in n.limitations)


def test_verifier_tests_passed_listed_when_true():
    b = brief("add retries to client.py")
    n = VerifierAgent().check(b, "Changed client.py to add retries", _facts(tests_ran=True, tests_passed=True,
                                                                           files_changed=["client.py"]))
    assert any("passed" in v for v in n.verified)
    assert any(r["kind"] == "direct" and r["status"] == "met" for r in n.requirements)


def test_verifier_tests_failed_is_limitation():
    n = VerifierAgent().check(brief("add retries to client.py"), "x", _facts(tests_ran=True, tests_passed=False))
    assert any("FAILED" in l for l in n.limitations) and not n.verified


def test_verifier_limitations_retrieval_and_job():
    n = VerifierAgent().check(brief("how does auth work", "ask"), "answer",
                              _facts(retrieval_confidence=0.2, vectorize_job="job123"))
    txt = " ".join(n.limitations)
    assert "0.20" in txt and "job123" in txt


def test_verifier_requirement_kinds_and_status_values():
    b = brief("add retries to client.py and log each failure")
    n = VerifierAgent().check(b, "I added retries to client.py", _facts())
    kinds = {r["kind"] for r in n.requirements}
    assert kinds == {"direct", "professional"}
    assert all(r["status"] in ("met", "partial", "unmet", "unknown") for r in n.requirements)
    assert all(r["status"] != "met" for r in n.requirements)      # nothing proven without tests
    # change intent with no files changed -> direct reqs unmet
    assert any(r["status"] == "unmet" for r in n.requirements if r["kind"] == "direct")


def test_verifier_llm_cannot_add_verified_or_fake_met():
    b = brief("add retries to client.py")
    def llm(p):
        return {"limitations": ["extra"], "verified": ["tests passed"],
                "requirements": [{"req": b.success_criteria[0], "status": "met", "evidence": "trust me"}]}
    n = VerifierAgent(llm_json=llm).check(b, "x", _facts())
    assert "extra" in n.limitations and n.verified == []
    assert n.requirements[0]["status"] == "partial"


def test_notes_markdown():
    n = DeliveryNotes(assumptions=["A1"], limitations=["L1"], verified=["V1"], unverified=["U1"],
                      requirements=[{"req": "R", "kind": "direct", "status": "met", "evidence": "e"}])
    md = n.to_markdown()
    assert md.startswith("### Delivery notes")
    for t in ("A1", "L1", "V1", "U1", "[met] (direct) R - e", "Assumptions", "Not verified"):
        assert t in md
    assert DeliveryNotes().to_markdown().startswith("### Delivery notes")


# ---------------------------------------------------------------- supervisor
def _sup(enabled=True, defaults=None):
    return ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: defaults or {"change_gate": "off"}, enabled)


def test_supervisor_runs_execute_under_policy_and_appends_notes():
    seen = {}

    async def execute(brief_, policy):
        seen["gate"] = flag("change_gate", "off")
        seen["risk"] = brief_.risk
        return "ANSWER", {"tests_ran": False}
    al.ring.clear()
    with al.request_scope("rq9"):
        out = asyncio.run(_sup().run(query="change password hashing in auth.py", intent="change",
                                     repo_files=["auth.py"], execute=execute))
    assert out.startswith("ANSWER") and "### Delivery notes" in out
    assert seen == {"gate": "strict", "risk": Risk.HIGH}
    assert flag("change_gate", "off") == "off"
    phases = [e["phase"] for e in al.recent(request_id="rq9") if e["status"] == "ok"]
    assert phases == ["INTAKE", "BASELINE", "CROSSCHECK", "DELIVER"]


def test_supervisor_clarify_first_short_circuits():
    called = []

    async def execute(b, p):
        called.append(1)
        return "x", {}
    out = asyncio.run(_sup().run(query="fix auth", intent="change", repo_files=[], execute=execute))
    assert called == []
    assert "?" in out and "1." in out and "Delivery notes" not in out


def test_supervisor_disabled_passthrough():
    al.ring.clear()

    async def execute(b, p):
        return "RAW", {}
    out = asyncio.run(_sup(enabled=False).run(query="fix auth", intent="change", repo_files=[], execute=execute))
    assert out == "RAW" and al.recent() == []


def test_supervisor_exception_in_execute_propagates_and_logs():
    async def execute(b, p):
        raise RuntimeError("bad")
    al.ring.clear()
    with pytest.raises(RuntimeError):
        asyncio.run(_sup().run(query="rename x to y in a.py", intent="change", repo_files=[], execute=execute))
    assert flag("change_gate", "off") == "off"


def test_red_green_criterion_not_overclaimed():
    from services.reasoning import IntakeAgent, VerifierAgent
    brief = IntakeAgent().analyze("Add input validation to create_user in users.py", "change")
    facts = {"tests_ran": True, "tests_passed": True, "files_changed": ["users.py"]}
    notes = VerifierAgent().check(brief, "done", facts)
    rg = [r for r in notes.requirements if "fails before" in r["req"].lower()]
    assert rg and rg[0]["status"] == "partial"
    notes2 = VerifierAgent().check(brief, "done", {**facts, "tests_red_first": True})
    rg2 = [r for r in notes2.requirements if "fails before" in r["req"].lower()]
    assert rg2[0]["status"] == "met"
