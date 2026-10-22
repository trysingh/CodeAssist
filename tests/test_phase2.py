"""Phase 2 tests: decision client, sandbox, test-failure fix loop, triage, apply-to-disk.
Run from the repo root:  pytest tests -q
No Docker, Ollama or network needed. The sandbox tests run real pytest in a subprocess
through LocalRunner (same entry logic as the Docker runner, minus the container).
"""
import asyncio
import uuid

import httpx
import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services.apply_service import apply_revision
from services.decision import (Choice, DecisionClient, decide_choice, decide_noul, systemone_url)
from services.patch_service import make_diff
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def rid():
    return f"r_{uuid.uuid4().hex[:8]}"


# ----------------------------------------------------------------------------- decision client
def _client(handler):
    return DecisionClient(transport=httpx.MockTransport(handler))


def test_systemone_url_normalisation(monkeypatch):
    for raw in ("http://h:11434", "http://h:11434/", "http://h:11434/v1", "http://h:11434/v1/systemone"):
        assert systemone_url(raw) == "http://h:11434/v1/systemone"
    monkeypatch.setattr(settings, "ollama_base_url_for_sys1_model", "")
    monkeypatch.setattr(settings, "ollama_base_url", "http://fallback:1")
    assert systemone_url() == "http://fallback:1/v1/systemone"


def test_noul_and_choice_parse_and_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        import json
        seen["body"] = json.loads(req.content)
        q = seen["body"]["questions"]["q"]
        if q["type"] == "noul":
            return httpx.Response(200, json={"answers": {"q": {"noul": 0.93}}})
        return httpx.Response(200, json={"answers": {"q": {
            "choice": "code_bug", "probabilities": {"code_bug": 0.8, "test_outdated": 0.2}, "confidence": 0.7}}})

    async def go():
        c = _client(handler)
        p = await decide_noul("tev1:0.8b", "x" * 5000, "Is it?", true="yes", false="no", client=c)
        assert p == 0.93
        assert seen["body"]["model"] == "tev1:0.8b"
        assert len(seen["body"]["state"]) == settings.decision_max_state_chars      # state is capped
        assert seen["body"]["questions"]["q"]["criteria"] == {"true": "yes", "false": "no"}
        ch = await decide_choice("tev1", "s", "Why?", {"code_bug": None, "test_outdated": None}, client=c)
        assert isinstance(ch, Choice) and ch.choice == "code_bug" and ch.confidence == 0.7
        # an option the caller did not offer is rejected
        c2 = _client(lambda r: httpx.Response(200, json={"answers": {"q": {"choice": "bogus", "confidence": 1}}}))
        assert await decide_choice("tev1", "s", "Why?", {"a": None, "b": None}, client=c2) is None
    run(go())


def test_circuit_breaker_skips_dead_endpoint(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    async def go():
        c = _client(handler)
        assert await c.decide("tev1", "s", {"q": {"type": "noul", "instructions": "x"}}) is None
        assert await c.decide("tev1", "s", {"q": {"type": "noul", "instructions": "x"}}) is None
        assert calls["n"] == 1                                       # second call never left the process
        monkeypatch.setattr(settings, "decision_cooldown_s", 0)
        c._down_until = 0
        await c.decide("tev1", "s", {"q": {"type": "noul", "instructions": "x"}})
        assert calls["n"] == 2

        def boom(req):
            raise httpx.ConnectError("refused")
        c = _client(boom)
        assert await c.decide("tev1", "s", {"q": {"type": "noul", "instructions": "x"}}) is None
    run(go())


def test_bad_request_does_not_trip_breaker():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, text="bad question")

    async def go():
        c = _client(handler)
        q = {"q": {"type": "noul", "instructions": "x"}}
        await c.decide("tev1", "s", q)
        await c.decide("tev1", "s", q)
        assert calls["n"] == 2
    run(go())


# ----------------------------------------------------------------------------- selection + parsing
TREE = {
    "services/calc.py": b"def add(a, b):\n    return a + b\n",
    "services/other.py": b"X = 1\n",
    "tests/test_calc.py": b"from services.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    "tests/test_other.py": b"from services import other\n\ndef test_x():\n    assert other.X == 1\n",
    "tests/test_page.py": b"def test_p():\n    assert 'page.html'\n",
    "templates/page.html": b"<p>hi</p>",
}


def test_select_tests_by_import_and_fallbacks():
    assert sbx.select_tests(TREE, ["services/calc.py"]) == (["tests/test_calc.py"], "affected")
    assert sbx.select_tests(TREE, ["services/other.py"]) == (["tests/test_other.py"], "affected")   # from pkg import mod
    assert sbx.select_tests(TREE, ["templates/page.html"]) == (["tests/test_page.py"], "affected")   # referenced by name
    assert sbx.select_tests(TREE, ["tests/test_calc.py"]) == (["tests/test_calc.py"], "affected")
    sel, strat = sbx.select_tests(TREE, ["README.md"])
    assert strat == "all" and len(sel) == 3
    assert sbx.select_tests({"a.py": b"x=1"}, ["a.py"]) == ([], "none")
    assert sbx.smoke_modules(TREE, ["services/calc.py", "tests/test_calc.py", "README.md", "bad-name.py"]) == ["services.calc"]


JUNIT = """<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="3">
<testcase classname="tests.test_a" name="test_ok" time="0.01"/>
<testcase classname="tests.test_a" name="test_bad"><failure message="assert 1 == 2">E   assert 1 == 2</failure></testcase>
<testcase classname="tests.test_a" name="test_skip"><skipped message="s"/></testcase>
</testsuite></testsuites>"""


def test_parse_junit_and_run_variants():
    passed, failed, skipped, cases = sbx.parse_junit(JUNIT)
    assert (passed, failed, skipped) == (1, 1, 1) and cases[0].id == "tests.test_a::test_bad"
    out = f"console\n\n::SBX-EXIT:: 1\n::SBX-JUNIT-BEGIN::\n{JUNIT}\n::SBX-JUNIT-END::\n"
    rep = sbx.parse_run(sbx.RawRun(stdout=out), "pytest", "docker")
    assert rep.ran and rep.failed == 1 and rep.failures[0].message == "assert 1 == 2"
    assert sbx.parse_run(sbx.RawRun(error="Docker CLI not found on PATH"), "pytest", "docker").ran is False
    assert sbx.parse_run(sbx.RawRun(stdout="x\n::SBX-EXIT:: 124\n"), "pytest", "docker").timed_out
    killed = sbx.parse_run(sbx.RawRun(stdout="x\n::SBX-EXIT:: 137\n"), "pytest", "docker")
    assert killed.failures[0].kind == "killed"
    smoke = sbx.parse_run(sbx.RawRun(stdout="SMOKE|a.b|OK\nSMOKE|c|ERR|NameError: nope\n::SBX-EXIT:: 0\n"), "smoke", "docker")
    assert smoke.passed == 1 and smoke.failures[0].id == "import::c"


def test_docker_command_is_locked_down():
    r = sbx.DockerRunner()
    args = r.docker_args("img:1", "nm", 99, ["python", "-m", "pytest"])
    joined = " ".join(args)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                 "--user 1000:1000", "--pids-limit", "--memory", "-e SBX_TIMEOUT=99"):
        assert flag in joined
    assert "-v " not in joined and "--mount" not in joined and "--privileged" not in joined   # no host mounts
    assert args[-4:] == ["img:1", "python", "-m", "pytest"] and "-i" in args and "--rm" in args
    assert r.image_tag("a==1") != r.image_tag("a==2") and r.image_tag("a==1") == r.image_tag("a==1")


def test_docker_unavailable_is_reported_not_raised(monkeypatch):
    def nope(*a, **k):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(sbx, "_exec", nope)
    ok, why = run(sbx.DockerRunner().available())
    assert not ok and "not found" in why


# ----------------------------------------------------------------------------- sandbox (real pytest, local runner)
GOOD = b"def add(a, b):\n    return a + b\n"
T_CALC = b"from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


async def _svc(files):
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, files)
    return sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None), s, r


def test_sandbox_green_and_new_failure_and_preexisting():
    async def go():
        svc, s, r = await _svc([("calc.py", GOOD), ("tests/test_calc.py", T_CALC),
                                ("tests/test_old.py", b"def test_old():\n    assert False\n"), ("big.py", b"x = 1\n")])
        v = await svc.verify(r, 0, {"calc.py": GOOD + b"\n# touched\n"})
        assert v.ran and v.report.mode == "pytest" and v.report.strategy == "affected" and v.ok
        assert v.report.passed == 1

        v = await svc.verify(r, 0, {"calc.py": b"def add(a, b):\n    return a - b\n"})
        assert not v.ok and [c.id for c in v.new_failures] == ["tests.test_calc::test_add"]
        assert "assert" in v.new_failures[0].detail

        # no test imports big.py -> every test runs; the already-broken test_old is ignored, not blamed
        v = await svc.verify(r, 0, {"big.py": b"x = 2\n"})
        assert v.report.strategy == "all" and v.ok and v.preexisting == 1
    run(go())


def test_sandbox_smoke_mode_and_timeout(monkeypatch):
    async def go():
        svc, s, r = await _svc([("pkg/__init__.py", b""), ("pkg/mod.py", b"X = 1\n")])
        v = await svc.verify(r, 0, {"pkg/mod.py": b"X = 2\n"})
        assert v.report.mode == "smoke" and v.ok and v.report.passed == 1
        v = await svc.verify(r, 0, {"pkg/mod.py": b"X = undefined_name\n"})
        assert not v.ok and v.new_failures[0].id == "import::pkg.mod" and "NameError" in v.new_failures[0].message
        v = await svc.verify(r, 0, {"README.md": b"hi"})
        assert not v.ran and "no tests" in v.report.note

        svc, s, r = await _svc([("loop.py", b"X = 1\n"), ("tests/test_loop.py", b"from loop import X\n\ndef test_x():\n    assert X == 1\n")])
        # A cold pytest start takes several seconds on Windows. The limit must sit above a healthy run
        # (the base run is subject to it too, otherwise "timed out on base as well" cancels the timeout).
        warm = await svc.verify(r, 0, {"loop.py": b"X = 1\n# touched\n"})
        assert warm.ok and warm.report.passed == 1
        monkeypatch.setattr(settings, "sandbox_test_timeout_s", max(2, int(warm.report.duration * 3) + 1))
        v = await svc.verify(r, 0, {"loop.py": b"import time\ntime.sleep(600)\nX = 1\n"})
        assert v.report.timed_out and not v.ok and v.new_failures[0].kind == "timeout"
    run(go())


def test_local_env_defines_home_the_way_windows_looks_it_up():
    host = {"PATH": "p", "SYSTEMROOT": "sr", "TEMP": "t", "TMP": "t", "COMSPEC": "c", "MY_SECRET_TOKEN": "x"}
    env = sbx.local_env("/tmp/ws", host)
    # Windows reads USERPROFILE (HOME is ignored): with only HOME, Path.home() raises
    assert env["USERPROFILE"] == env["HOME"] == "/tmp/ws" and env["HOMEPATH"] and env["HOMEDRIVE"]
    assert env["SYSTEMROOT"] == "sr" and env["COMSPEC"] == "c"
    assert "MY_SECRET_TOKEN" not in env                      # host secrets are not forwarded


def test_local_runner_subprocess_can_resolve_a_home_directory():
    code = "import pathlib; print('HOME=' + str(pathlib.Path.home()))"
    raw = run(sbx.LocalRunner().run("", sbx.build_tar({"a.py": b""}), ["python", "-c", code], 60))
    assert "Could not determine home directory" not in raw.stdout
    assert "HOME=" in raw.stdout and "coderag_local_" in raw.stdout      # resolved to the throw-away workspace


BROKEN_CONFTEST = b"raise RuntimeError('pytest cannot start')\n"


def test_pytest_that_cannot_start_is_inconclusive_not_green():
    async def go():
        # pytest dies in the base run AND in the changed run: the baseline must not cancel it into a pass
        files = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC), ("conftest.py", BROKEN_CONFTEST)]
        svc, s, r = await _svc(files)
        v = await svc.verify(r, 0, {"calc.py": GOOD + b"\n# touched\n"})
        assert not v.ran and not v.ok and v.report.passed == 0
        assert "no test produced a result" in v.report.note

        # the fix loop then falls back to static checks only and says so
        llm = Script(block("calc.py", "    return a + b", "    return a + b + 0"))
        svc, s, r = await _change_svc(llm, files)
        res = await svc.run(r, "refactor calc.py add")
        assert res.ok and res.attempts == 1 and res.verification == "L0-static"
        assert "tests not run" in " ".join(res.notes)

        # whereas a change that breaks pytest start-up in a healthy repo IS blamed on the change
        svc, s, r = await _svc([("calc.py", GOOD), ("tests/test_calc.py", T_CALC)])
        v = await svc.verify(r, 0, {"conftest.py": BROKEN_CONFTEST})
        assert v.ran and not v.ok and v.new_failures[0].id == "pytest::run"
    run(go())


def test_sandbox_disabled_or_unavailable_degrades_gracefully():
    class Down:
        name = "docker"

        async def available(self):
            return False, "Docker is installed but the daemon is not reachable"

    async def go():
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, [("a.py", b"x = 1\n")])
        v = await sbx.SandboxService(s, sbx.LocalRunner(), enabled=False).verify(r, 0, {"a.py": b"x = 2\n"})
        assert not v.ran and v.report.note == "sandbox disabled"
        v = await sbx.SandboxService(s, Down(), enabled=True).verify(r, 0, {"a.py": b"x = 2\n"})
        assert not v.ran and "daemon" in v.report.note
    run(go())


# ----------------------------------------------------------------------------- fix loop with real test runs
class Doc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class Vec:
    def __init__(self, paths):
        self.paths = paths

    def similarity_search(self, repo_id, q, k=4):
        return [Doc(p) for p in self.paths]


class Script:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


async def _no_json(*a, **k):
    return None


async def _no_choice(*a, **k):
    return None


async def _change_svc(llm, files, choice_fn=_no_choice, runner=None):
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, files)
    emit = lambda *a: None
    sb = sbx.SandboxService(s, runner or sbx.LocalRunner(), enabled=True, emit=emit)
    svc = RepoChangeService(None, Vec([]), s, llm_text=llm, llm_json=_no_json, emit=emit,
                            sandbox=sb, choice_fn=choice_fn)
    return svc, s, r


REPO = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC)]


def test_fix_loop_recovers_from_a_test_failure():
    async def go():
        llm = Script(block("calc.py", "    return a + b", "    return a - b"),     # breaks the test
                     block("calc.py", "    return a - b", "    return a + b + 0"))  # fixes it
        svc, s, r = await _change_svc(llm, REPO)
        res = await svc.run(r, "refactor calc.py add")
        assert res.ok and res.attempts == 2 and res.verification == "L2-tests"
        assert "tests.test_calc::test_add" in llm.prompts[1]          # the model saw the failing test
        assert "tests/test_calc.py" in llm.prompts[1]                 # and the test file itself
        md = res.to_markdown()
        assert "L2-tests" in md and "1 passed" in md and "[Apply to repo]" in md
        assert (await s.get_file(r, "calc.py")).data.endswith(b"a + b + 0\n")
    run(go())


def test_no_progress_stops_early_and_saves_nothing():
    async def go():
        bad = block("calc.py", "    return a + b", "    return a - b")
        same_again = block("calc.py", "    return a - b", "    return a - b  # still wrong")
        llm = Script(bad, same_again, same_again)
        svc, s, r = await _change_svc(llm, REPO)
        res = await svc.run(r, "refactor calc.py add")
        assert not res.ok and any("no progress" in e for e in res.errors)
        assert res.attempts == 2 and await s.latest_rev(r) == 0
    run(go())


def test_triage_lets_the_patcher_update_an_outdated_test():
    async def go():
        async def outdated(model, state, instructions, options, **kw):
            assert "test_add" in state and set(options) == {"code_bug", "test_outdated"}
            return Choice("test_outdated", 0.9, {})
        settings.router_model = "tev1:0.8b"
        try:
            llm = Script(block("calc.py", "    return a + b", "    return a * b"),
                         block("tests/test_calc.py", "assert add(2, 3) == 5", "assert add(2, 3) == 6"))
            svc, s, r = await _change_svc(llm, REPO, choice_fn=outdated)
            res = await svc.run(r, "make add in calc.py multiply instead")
        finally:
            settings.router_model = ""
        assert res.ok and res.attempts == 2
        assert "expects OLD behaviour" in llm.prompts[1]
        assert (await s.get_file(r, "tests/test_calc.py")).data.count(b"== 6") == 1
    run(go())


def test_low_confidence_triage_defaults_to_code_bug():
    async def go():
        async def unsure(*a, **k):
            return Choice("test_outdated", 0.3, {})
        settings.router_model = "tev1:0.8b"
        try:
            llm = Script(block("calc.py", "    return a + b", "    return a - b"),
                         block("calc.py", "    return a - b", "    return a + b  # fixed"))
            svc, s, r = await _change_svc(llm, REPO, choice_fn=unsure)
            res = await svc.run(r, "refactor add in calc.py")
        finally:
            settings.router_model = ""
        assert res.ok and "OLD behaviour" not in llm.prompts[1]
    run(go())


def test_missing_third_party_package_is_inconclusive_not_a_failure():
    async def go():
        files = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC)]
        llm = Script(block("calc.py", "    return a + b", "    import not_installed_pkg_xyz\n    return a + b"))
        svc, s, r = await _change_svc(llm, files)
        res = await svc.run(r, "use the xyz library in calc.py add")
        assert res.ok and res.verification == "L0-static" and res.tests["inconclusive"]
        assert "Tests inconclusive" in res.to_markdown() and res.attempts == 1
    run(go())


def test_docker_down_still_saves_with_clear_note():
    class Down:
        name = "docker"

        async def available(self):
            return False, "Docker CLI not found on PATH"

    async def go():
        llm = Script(block("calc.py", "    return a + b", "    return a + b + 1"))
        svc, s, r = await _change_svc(llm, REPO, runner=Down())
        res = await svc.run(r, "tweak add in calc.py")
        assert res.ok and res.verification == "L0-static"
        assert "Tests not run:** Docker CLI not found" in res.to_markdown()
    run(go())


# ----------------------------------------------------------------------------- apply to disk
def test_apply_writes_exact_bytes_and_resyncs_base(tmp_path):
    async def go():
        s, r = RepoStore(), rid()
        (tmp_path / "a.py").write_bytes(b"x = 1\r\n")
        await s.upsert_base_files(r, [("a.py", b"x = 1\r\n")])
        rev = await s.create_revision(r, 0, "req", {"a.py": b"x = 2\r\n", "sub/new.py": b"n = 1\n"}, {})
        res = await apply_revision(s, r, rev, tmp_path)
        assert res["ok"] and sorted(res["written"]) == ["a.py", "sub/new.py"]
        assert (tmp_path / "a.py").read_bytes() == b"x = 2\r\n"              # CRLF kept
        assert (tmp_path / "sub" / "new.py").read_bytes() == b"n = 1\n"
        assert not list(tmp_path.rglob("*.coderag-tmp"))
        # base now equals disk: the watcher's re-index sees nothing to warn about
        out = await s.upsert_base_files(r, [("a.py", b"x = 2\r\n")])
        assert out["unchanged"] == ["a.py"] and out["shadowed"] == []
        assert (await s.get_revision(r, rev))["report"]["applied_at"]
        again = await apply_revision(s, r, rev, tmp_path)
        assert again["ok"] and again["written"] == [] and "already matches" in again["note"]
    run(go())


def test_apply_detects_conflicts_force_and_refuses_unsafe(tmp_path):
    async def go():
        s, r = RepoStore(), rid()
        (tmp_path / "a.py").write_bytes(b"x = 1\n")
        await s.upsert_base_files(r, [("a.py", b"x = 1\n")])
        rev = await s.create_revision(r, 0, "req", {"a.py": b"x = 2\n"}, {})
        (tmp_path / "a.py").write_bytes(b"x = 999  # edited by hand\n")
        res = await apply_revision(s, r, rev, tmp_path)
        assert not res["ok"] and res["conflicts"][0]["path"] == "a.py"
        assert (tmp_path / "a.py").read_bytes().startswith(b"x = 999")            # untouched
        forced = await apply_revision(s, r, rev, tmp_path, force=True)
        assert forced["ok"] and (tmp_path / "a.py").read_bytes() == b"x = 2\n"

        rev2 = await s.create_revision(r, rev, "bad", {".github/x.yml": b"a: 1\n", "prod.env": b"K=1\n"}, {})
        res = await apply_revision(s, r, rev2, tmp_path)
        assert not res["ok"] and set(res["refused"]) == {".github/x.yml", "prod.env"}
        assert not (tmp_path / ".github").exists()
        with pytest.raises(ValueError):
            await apply_revision(s, r, 99, tmp_path)
    run(go())


def test_apply_deletion_and_rollback_on_failure(tmp_path, monkeypatch):
    async def go():
        s, r = RepoStore(), rid()
        (tmp_path / "a.py").write_bytes(b"a = 1\n")
        (tmp_path / "b.py").write_bytes(b"b = 1\n")
        await s.upsert_base_files(r, [("a.py", b"a = 1\n"), ("b.py", b"b = 1\n")])
        rev = await s.create_revision(r, 0, "del", {"a.py": None, "b.py": b"b = 2\n"}, {})
        res = await apply_revision(s, r, rev, tmp_path)
        assert res["ok"] and res["deleted"] == ["a.py"] and not (tmp_path / "a.py").exists()
        assert await s.list_paths(r, 0) == ["b.py"]

        # second revision: make the 2nd write fail -> the 1st is restored
        (tmp_path / "c.py").write_bytes(b"c = 1\n")
        (tmp_path / "d.py").write_bytes(b"d = 1\n")
        await s.upsert_base_files(r, [("c.py", b"c = 1\n"), ("d.py", b"d = 1\n")])
        rev2 = await s.create_revision(r, rev, "two", {"c.py": b"c = 2\n", "d.py": b"d = 2\n"}, {})
        import os as _os
        real = _os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:
                raise PermissionError("file is locked")
            return real(src, dst)
        monkeypatch.setattr("services.apply_service.os.replace", flaky)
        with pytest.raises(RuntimeError, match="rolled back"):
            await apply_revision(s, r, rev2, tmp_path)
        assert (tmp_path / "c.py").read_bytes() == b"c = 1\n" and (tmp_path / "d.py").read_bytes() == b"d = 1\n"
    run(go())


def test_disk_edit_under_a_revision_is_recorded_and_older_revision_cannot_overwrite_it(tmp_path):
    async def go():
        s, r = RepoStore(), rid()
        (tmp_path / "a.py").write_bytes(b"x = 1\n")
        await s.upsert_base_files(r, [("a.py", b"x = 1\n")])
        rev1 = await s.create_revision(r, 0, "p1", {"a.py": b"x = 2\n"}, {})
        assert (await apply_revision(s, r, rev1, tmp_path))["ok"]

        # user hand-edits a.py; ingest sees the change and the file is shadowed by revision 1
        (tmp_path / "a.py").write_bytes(b"x = 3  # by hand\n")
        out = await s.upsert_base_files(r, [("a.py", b"x = 3  # by hand\n")])
        assert out["shadowed"] == ["a.py"]
        assert (await s.get_file(r, "a.py")).data == b"x = 2\n"                 # stale view before the sync
        sync = await s.record_disk_sync(r, {"a.py": b"x = 3  # by hand\n"})
        assert sync == 2 and (await s.get_file(r, "a.py")).data == b"x = 3  # by hand\n"
        assert (await s.get_revision(r, sync))["report"]["kind"] == "disk-sync"
        assert await s.record_disk_sync(r, {}) is None

        # applying revision 1 again is a no-op: disk already matches what it wrote
        again = await apply_revision(s, r, rev1, tmp_path)
        assert again["ok"] is False or again["written"] == []
        assert (tmp_path / "a.py").read_bytes() == b"x = 3  # by hand\n"

        # an OLDER revision may not overwrite a file that a NEWER revision changes
        (tmp_path / "b.py").write_bytes(b"b = 1\n")
        await s.upsert_base_files(r, [("b.py", b"b = 1\n")])
        latest = await s.latest_rev(r)
        rev_a = await s.create_revision(r, latest, "A", {"b.py": b"b = 2\n"}, {})
        rev_b = await s.create_revision(r, rev_a, "B", {"b.py": b"b = 3\n"}, {})
        res = await apply_revision(s, r, rev_a, tmp_path)
        assert not res["ok"] and f"newer revision ({rev_b})" in res["conflicts"][0]["reason"]
        assert (tmp_path / "b.py").read_bytes() == b"b = 1\n"
        assert (await apply_revision(s, r, rev_b, tmp_path))["ok"]
        assert (tmp_path / "b.py").read_bytes() == b"b = 3\n"
    run(go())
