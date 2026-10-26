"""How the sandbox result is judged: baseline on disk, reason codes, unapplied-stack note, JS / template checks.
Run from the repo root:  pytest tests/test_verdicts_and_web_checks.py -q      (no Docker, Ollama or network needed)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services import static_check
from services.repo_change import ChangeResult
from services.repo_store import RepoStore

run = asyncio.run

CALC = b"def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n"
CALC_BROKEN = b"def add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n"           # an earlier, unapplied proposal
CALC_NEXT = CALC_BROKEN + b"\ndef sub(a, b):\n    return a - b\n"                                     # the new proposal on top of it
TESTS = (b"from calc import add, mul\n\ndef test_add():\n    assert add(2, 3) == 5\n\n"
         b"def test_mul():\n    assert mul(2, 3) == 6\n")


def rid():
    return f"r_{uuid.uuid4().hex[:8]}"


async def stack():
    """disk = good calc.py; revision 1 (NOT applied) breaks add(); returns the service, store, repo id and that revision."""
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, [("calc.py", CALC), ("tests/test_calc.py", TESTS)])
    r1 = await s.create_revision(r, 0, "break add", {"calc.py": CALC_BROKEN}, {})
    svc = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
    return svc, s, r, r1


# ============================================================================= baseline
def test_failures_that_only_exist_on_unapplied_revisions_are_not_excused_by_default(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_baseline", "disk")

    async def go():
        svc, s, r, r1 = await stack()
        v = await svc.verify(r, r1, {"calc.py": CALC_NEXT})
        assert v.ran and not v.ok and [c.id for c in v.new_failures] == ["tests.test_calc::test_add"]
        assert v.preexisting == 0 and v.baseline == "disk" and v.on_unapplied == ["calc.py"]
        d = v.to_dict()
        assert d["on_unapplied_n"] == 1 and d["baseline"] == "disk" and d["not_run"] == ""
    run(go())


def test_the_old_rule_is_one_setting_away(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_baseline", "base")

    async def go():
        svc, s, r, r1 = await stack()
        v = await svc.verify(r, r1, {"calc.py": CALC_NEXT})
        assert v.ok and v.preexisting == 1 and v.report.passed == 1 and v.baseline == "base"
    run(go())


def test_a_failure_that_also_exists_on_disk_is_still_excused(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_baseline", "disk")

    async def go():
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, [("calc.py", CALC_BROKEN), ("tests/test_calc.py", TESTS)])    # broken ON DISK
        svc = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        v = await svc.verify(r, 0, {"calc.py": CALC_NEXT})
        assert v.ok and v.preexisting == 1 and v.on_unapplied == [] and v.baseline == "disk"
    run(go())


def test_the_baseline_is_not_reused_between_a_disk_and_a_base_comparison(monkeypatch):
    async def go():
        svc, s, r, r1 = await stack()
        monkeypatch.setattr(settings, "sandbox_baseline", "base")
        assert (await svc.verify(r, r1, {"calc.py": CALC_NEXT})).ok
        monkeypatch.setattr(settings, "sandbox_baseline", "disk")
        assert not (await svc.verify(r, r1, {"calc.py": CALC_NEXT})).ok
    run(go())


def test_the_unapplied_stack_is_announced():
    async def go():
        svc, s, r, r1 = await stack()
        said = []
        svc._emit_fn = lambda *a: said.append(a[1])
        await svc.verify(r, r1, {"calc.py": CALC_NEXT})
        assert any("not applied to your disk yet" in m and "calc.py" in m for m in said)
    run(go())


def test_markdown_tells_the_user_the_result_holds_for_the_stack():
    cr = ChangeResult.__new__(ChangeResult)
    cr.tests = {"ran": True, "mode": "pytest", "passed": 3, "failed": 0, "preexisting": 0, "selected": 1, "strategy": "affected",
                "on_unapplied_n": 2, "on_unapplied": ["a.py", "b.py"], "baseline": "disk"}
    line = cr._tests_line()
    assert "2 file(s) from earlier proposals" in line and "a.py, b.py" in line and "apply them together" in line
    cr.tests = {"ran": False, "note": "no tests", "not_run": "no_target"}
    assert "(reason: `no_target`)" in cr._tests_line()


# ============================================================================= reason codes
def test_every_not_run_path_carries_a_known_code():
    async def go():
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, [("page.html", b"<p>x</p>")])
        off = sbx.SandboxService(s, sbx.LocalRunner(), enabled=False, emit=lambda *a: None)
        assert (await off.verify(r, 0, {"page.html": b"<p>y</p>"})).not_run == "disabled"

        on = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        v = await on.verify(r, 0, {"page.html": b"<p>y</p>"})
        assert not v.ran and v.not_run == "no_target" and v.to_dict()["not_run"] == "no_target"

        class Down(sbx.LocalRunner):
            async def available(self):
                return False, "daemon not reachable"
        assert (await sbx.SandboxService(s, Down(), enabled=True, emit=lambda *a: None).verify(r, 0, {"a.py": b"x=1\n"})).not_run == "docker_down"
    run(go())
    assert {"disabled", "docker_down", "no_target", "timeout", "all_skipped", "collected_nothing", "failing_on_baseline",
            "no_result", "image_build", "infra", "workspace_too_large"} <= set(sbx.NOT_RUN)


def test_unverified_picks_the_code_from_what_happened():
    mk = lambda **k: sbx.RunReport(mode="pytest", ran=True, **k)
    assert sbx._unverified(mk(timed_out=True)).not_run == "timeout"
    assert sbx._unverified(mk(skipped=3)).not_run == "all_skipped"
    assert sbx._unverified(mk(note="pytest collected no tests")).not_run == "collected_nothing"
    assert sbx._unverified(mk(), preexisting=2).not_run == "failing_on_baseline"
    assert sbx._unverified(mk()).not_run == "no_result"
    assert not sbx._unverified(mk()).ran


# ============================================================================= JS + template checks
def test_js_syntax_errors_are_reported_and_modern_syntax_is_not():
    bad = static_check.check_files({"static/app.js": b"if (x) {\n  foo(\n}\n"})
    assert [r.check for r in bad] == ["js-syntax"] and not bad[0].ok and bad[0].line and "foo(" in bad[0].message
    good = (b"class K { static #p = 1; #m() {} get v() { return 1 } }\nconst x = a?.b ?? 2; x ||= 3;\n"
            b"export default async function* g() { for await (const y of z) yield y }\nconst t = `a${1}b`;\n"
            b"const el = <div className='x'>{y}</div>;\n")
    assert all(r.ok and not r.skipped for r in static_check.check_files({"a.js": good}))


def test_js_check_is_skipped_not_passed_when_the_parser_is_missing(monkeypatch):
    monkeypatch.setattr(static_check, "_ts_parser", None)
    r = static_check.check_files({"a.js": b"if ("})[0]
    assert r.ok and r.skipped


def test_a_js_error_that_was_already_there_is_not_blamed_on_the_change():
    base = static_check.check_files({"a.js": b"function f( {\n"})
    now = static_check.check_files({"a.js": b"// touched\nfunction f( {\n"})
    assert not static_check.new_failures(base, now)


def test_template_references_must_exist():
    page = b'{% extends "base.html" %}{% include "nav.html" %}{% include "opt.html" ignore missing %}{% block c %}{% endblock %}'
    res = static_check.check_files({"templates/page.html": page}, known_paths=["templates/nav.html"])
    bad = [r for r in res if not r.ok]
    assert len(bad) == 1 and bad[0].check == "jinja-ref" and "base.html" in bad[0].message
    ok = static_check.check_files({"templates/page.html": page}, known_paths=["app/templates/base.html", "templates/nav.html"])
    assert all(r.ok for r in ok)                                   # found by file name below any folder


def test_dynamic_template_names_and_missing_context_are_not_judged():
    dyn = b"{% extends layout %}{% include some_name %}"
    assert all(r.ok for r in static_check.check_files({"t.html": dyn}, known_paths=[]))
    assert [r.check for r in static_check.check_files({"t.html": b'{% extends "base.html" %}'})] == ["jinja"]     # no known_paths: parse only


def test_a_missing_template_that_the_base_file_already_referenced_is_not_new():
    known = ["templates/other.html"]
    base = static_check.check_files({"templates/p.html": b'{% extends "gone.html" %}'}, known_paths=known)
    now = static_check.check_files({"templates/p.html": b'{% extends "gone.html" %}<p>edit</p>'}, known_paths=known)
    assert not static_check.new_failures(base, now)
