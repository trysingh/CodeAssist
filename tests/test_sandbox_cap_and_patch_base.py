"""Sandbox time-cap handling, the run-everything limit, the per-file watcher debounce and the disk-conflict check behind
the .patch download. Run from the repo root:  pytest tests/test_sandbox_cap_and_patch_base.py -q
No Docker, Ollama or network needed (the sandbox runner is a fake)."""
import asyncio
import sys
import time
import types
import uuid

from config.settings import settings
from services import sandbox_service as sbx
from services.apply_service import disk_conflicts
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run


class FakeRunner:
    name = "local"

    def __init__(self, *runs):
        self.runs, self.calls = list(runs), []

    async def available(self):
        return True, "fake"

    async def prepare(self, requirements):
        return None

    async def run(self, requirements, tar, argv, timeout):
        self.calls.append(argv)
        return self.runs.pop(0)


async def _svc(files, *runs):
    s, rid = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(rid, files)
    events = []
    runner = FakeRunner(*runs)
    return sbx.SandboxService(s, runner, enabled=True, emit=lambda title, msg, level="INFO": events.append((msg, level))), runner, events, rid


CALC = [("calc.py", b"def add(a, b):\n    return a + b\n"),
        ("tests/test_calc.py", b"from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")]


def test_pytest_progress_counts_the_dots_of_a_killed_run():
    assert sbx.pytest_progress("....F.  [ 40%]\nE..s  [ 70%]\nnot progress ...\n") == (7, 2)   # 7 dots, F + E; the skipped "s" is neither
    assert sbx.pytest_progress("") == (0, 0)
    assert sbx.pytest_progress("collecting ...\n") == (0, 0)


def test_a_run_cut_off_while_passing_is_unverified_and_skips_the_second_cap(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_test_timeout_s", 180)

    async def go():
        cut = sbx.RawRun(stdout="..........  [ 40%]\n....  [ 70%]\n", timed_out=True)
        svc, runner, events, rid = await _svc(CALC, cut)
        v = await svc.verify(rid, 0, {"calc.py": CALC[0][1] + b"# touched\n"})
        assert not v.ran and len(runner.calls) == 1                      # no untouched-base run waiting for another cap
        assert "after 14 test(s) (14 passed, 0 failed so far)" in v.report.note
        assert "ENGINE_SANDBOX_TEST_TIMEOUT_S" in v.report.note
    run(go())


def test_a_run_that_hangs_without_progress_still_gets_the_base_comparison(monkeypatch):
    async def go():
        hang = sbx.RawRun(stdout="", timed_out=True)
        svc, runner, events, rid = await _svc(CALC, hang, hang)
        v = await svc.verify(rid, 0, {"calc.py": CALC[0][1] + b"# touched\n"})
        assert len(runner.calls) == 2 and not v.ran                       # change run + base run, as before
    run(go())


def _many_tests(n):
    return [(f"tests/test_k{i}.py", b"def test_x():\n    assert True\n") for i in range(n)] + [("README.md", b"hi\n")]


def test_no_related_test_and_a_big_suite_does_not_run_everything(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_all_max_files", 12)

    async def go():
        svc, runner, events, rid = await _svc(_many_tests(14))
        v = await svc.verify(rid, 0, {"README.md": b"changed\n"})
        assert not v.ran and runner.calls == []
        assert any("14 test files (limit 12)" in m and lvl == "WARNING" for m, lvl in events)
    run(go())


def test_the_limit_can_be_lifted_and_small_suites_are_unaffected(monkeypatch):
    ok = sbx.RawRun(stdout="::SBX-EXIT:: 0\n", exit_code=0)

    async def go():
        monkeypatch.setattr(settings, "sandbox_all_max_files", 0)               # 0 = always run everything
        svc, runner, events, rid = await _svc(_many_tests(14), ok)
        await svc.verify(rid, 0, {"README.md": b"changed\n"})
        assert len(runner.calls) == 1
        monkeypatch.setattr(settings, "sandbox_all_max_files", 12)
        svc, runner, events, rid = await _svc(_many_tests(3), ok)               # 3 files: under the limit
        await svc.verify(rid, 0, {"README.md": b"changed\n"})
        assert len(runner.calls) == 1
    run(go())


# --------------------------------------------------------------------------- watcher: debounce per file
def test_files_written_together_are_all_reindexed_and_a_burst_on_one_file_runs_once(monkeypatch, tmp_path):
    done = []

    class FakeController:
        def is_excluded(self, p, root):
            return False

        async def incremental_reingest_file(self, repo_id, root, rel):
            done.append(rel)

    fake = types.ModuleType("controllers.engine_controller")
    fake.EngineController = FakeController
    monkeypatch.setitem(sys.modules, "controllers.engine_controller", fake)
    from services.watcher_service import RepoChangeHandler
    monkeypatch.setattr(RepoChangeHandler, "DEBOUNCE_S", 0.05)

    h = RepoChangeHandler("r", tmp_path)
    for _ in range(5):                                       # an editor saving: several events for one file
        h._handle_file_change(str(tmp_path / "a.py"))
    for name in ("b.py", "c.py", "d.html"):                  # `git apply` / a folder copy: other files in the same instant
        h._handle_file_change(str(tmp_path / name))
    time.sleep(0.6)
    assert sorted(done) == ["a.py", "b.py", "c.py", "d.html"] and done.count("a.py") == 1


# --------------------------------------------------------------------------- patch base: what changed on disk since
def test_disk_conflicts_name_files_that_no_longer_match_the_snapshot(tmp_path):
    async def go():
        s, rid = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        base = [("a.py", b"X = 1\n"), ("b.js", b"const y = 1;\n")]
        await s.upsert_base_files(rid, base)

        class Doc:
            metadata = {"file_path": "a.py"}

        class Vec:
            def similarity_search(self, *a, **k):
                return [Doc()]

        async def llm(model, prompt, **kw):
            return ("FILE: a.py\n<<<<<<< SEARCH\nX = 1\n=======\nX = 2\n>>>>>>> REPLACE\n"
                    "FILE: b.js\n<<<<<<< SEARCH\nconst y = 1;\n=======\nconst y = 2;\n>>>>>>> REPLACE\n")

        async def nj(*a, **k):
            return None

        svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=nj, emit=lambda *a: None)
        r = await svc.run(rid, "change X in a.py and y in b.js")
        assert r.ok
        for p, data in base:                                  # the disk still matches the snapshot
            (tmp_path / p).write_bytes(data)
        assert await disk_conflicts(s, rid, r.revision, tmp_path) == []
        (tmp_path / "b.js").write_bytes(b"const y = 1; // edited by hand\n")      # then someone edits it
        found = await disk_conflicts(s, rid, r.revision, tmp_path)
        assert [(c["path"], c["reason"]) for c in found] == [("b.js", "file changed on disk since ingest")]
        (tmp_path / "a.py").unlink()
        assert {c["path"] for c in await disk_conflicts(s, rid, r.revision, tmp_path)} == {"a.py", "b.js"}
    run(go())

def test_closest_tests_keeps_name_matches_only():
    tests = ["tests/test_contract.py", "tests/test_phase1.py", "tests/test_plan.py"]
    assert sbx.closest_tests(tests, ["services/contract.py"]) == ["tests/test_contract.py"]
    assert sbx.closest_tests(tests, ["config/settings.py"]) == []