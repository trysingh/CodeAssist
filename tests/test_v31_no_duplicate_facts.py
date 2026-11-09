"""Regression (v3.1 crash): `_note_facts(**facts_from_sandbox(...), achieved_level=...)` raised
TypeError "got multiple values for keyword argument" and turned a saved revision into an HTTP 500.
No call may pass a keyword that the unpacked facts_from_sandbox(...) dict already contains."""
import ast
import pathlib

from services.reasoning.crosscheck import facts_from_sandbox

ROOT = pathlib.Path(__file__).resolve().parents[1]
KEYS = set(facts_from_sandbox({"ran": True, "ok": True, "mode": "pytest", "passed": 1}))


def _calls():
    for path in list(ROOT.joinpath("services").rglob("*.py")) + list(ROOT.joinpath("controllers").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path, node


def _unpacks_sandbox_facts(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg is None:                       # **expr
            for sub in ast.walk(kw.value):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", getattr(sub.func, "attr", "")) == "facts_from_sandbox":
                    return True
    return False


def test_no_call_repeats_a_sandbox_fact_keyword():
    bad = []
    for path, call in _calls():
        if _unpacks_sandbox_facts(call):
            dup = {kw.arg for kw in call.keywords if kw.arg} & KEYS
            if dup:
                bad.append(f"{path.relative_to(ROOT)}:{call.lineno} repeats {sorted(dup)}")
    assert not bad, bad


def test_facts_from_sandbox_tolerates_odd_counts():
    f = facts_from_sandbox({"ran": True, "ok": True, "mode": "pytest", "passed": "n/a", "failed": None})
    assert f["tests_ran"] is False and f["tests_count"] == 0


def test_achieved_level_override_merges_without_error():
    facts = facts_from_sandbox({"ran": True, "ok": True, "mode": "boot", "passed": 9, "backend": "docker"})
    facts.update(achieved_level="L1-boot-smoke", static_checked=True)
    box = {}
    (lambda **kw: box.update(kw))(**facts)      # the controller's call shape
    assert box["achieved_level"] == "L1-boot-smoke" and box["boot_checked"] is True and box["tests_ran"] is False
