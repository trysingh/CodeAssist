"""Sandbox pre-warm: dependency images are built in the background instead of on the first change request.
Run from the repo root:  pytest tests -q
No Docker, Ollama or network needed. DockerRunner is exercised with its subprocess helpers replaced;
the warmer is exercised against a fake sandbox; the only real process is a short-lived Python sleeper
that proves a running build can be killed.
"""
import asyncio
import hashlib
import sqlite3
import sys
import threading
import time
import uuid

import pytest

from config.settings import settings
from models.database import DatabaseManager
from models.schemas import GENERAL_REPO_ID, JobStatus
from services import sandbox_service as sbx
from services import sandbox_warmer as warmer_mod
from services.repo_store import RepoStore
from services.sandbox_warmer import SandboxWarmer, is_requirements_path, request_prewarm

run = asyncio.run

GOOD = b"def add(a, b):\n    return a + b\n"
T_CALC = b"from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def rid():
    return f"r_{uuid.uuid4().hex[:8]}"


async def until(pred, timeout=5.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.005)


# ============================================================================= DockerRunner
class _Calls:
    """Replaces sbx._exec: records argv, answers from `answers(argv)`."""
    def __init__(self, answers):
        self.answers, self.argvs = answers, []

    def __call__(self, args, stdin=None, timeout=0, cwd=None, env=None):
        self.argvs.append(list(args))
        return self.answers(list(args))


def test_available_is_cached_and_a_down_answer_only_briefly(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_available_ttl_s", 60)
    up = _Calls(lambda a: (0, b"27.1\n", False))
    monkeypatch.setattr(sbx, "_exec", up)
    r = sbx.DockerRunner()

    async def go():
        assert await r.available() == (True, "27.1")
        assert await r.available() == (True, "27.1")
        assert len(up.argvs) == 1                                  # `docker version` ran once, not twice
        r.invalidate_availability()
        await r.available()
        assert len(up.argvs) == 2

        down = _Calls(lambda a: (1, b"", False))
        monkeypatch.setattr(sbx, "_exec", down)
        r.invalidate_availability()
        ok, why = await r.available()
        assert not ok and "daemon" in why
        await r.available()
        assert len(down.argvs) == 1                                # a fresh "down" is reused ...
        ts, ok, info = r._avail
        r._avail = (ts - 11, ok, info)                             # ... but only for ~10 s: Docker Desktop may have started
        await r.available()
        assert len(down.argvs) == 2
    run(go())


def test_available_cache_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_available_ttl_s", 0)
    c = _Calls(lambda a: (0, b"27.1", False))
    monkeypatch.setattr(sbx, "_exec", c)
    r = sbx.DockerRunner()

    async def go():
        await r.available()
        await r.available()
    run(go())
    assert len(c.argvs) == 2


def test_docker_itself_failing_drops_the_cached_up_answer(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_available_ttl_s", 60)
    r = sbx.DockerRunner()
    r._avail = (time.monotonic(), True, "27.1")
    monkeypatch.setattr(sbx, "_exec", _Calls(lambda a: (125, b"daemon gone", False)))
    out = run(r.run("", b"tar", ["python", "-V"], 5))
    assert out.exit_code == 125 and r._avail is None


def test_failed_build_is_retried_after_the_cooldown_not_never(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_build_retry_s", 600)
    r = sbx.DockerRunner()
    builds = []
    results = [(1, b"pip: network unreachable", False), (0, b"", False)]

    def build(args, timeout):
        builds.append(args)
        return results.pop(0)
    monkeypatch.setattr(r, "_exec_tracked", build)
    monkeypatch.setattr(sbx, "_exec", _Calls(lambda a: (1, b"", False)))      # `docker image inspect`: not there
    tag = r.image_tag("requests==2.0")

    async def go():
        err = await r.prepare("requests==2.0")
        assert err and "image build failed" in err and "network unreachable" in err
        assert await r.prepare("requests==2.0") == err and len(builds) == 1   # inside the cooldown: no second build
        ts, msg = r._failed[tag]
        r._failed[tag] = (ts - 601, msg)                                       # cooldown over
        assert await r.prepare("requests==2.0") is None and len(builds) == 2
        assert tag not in r._failed
    run(go())


def test_a_build_in_flight_is_awaited_by_the_next_request_not_started_twice(monkeypatch):
    r = sbx.DockerRunner()
    built, release, starting = [], threading.Event(), threading.Event()

    def inspect(args):
        return (0 if built else 1), b"", False
    monkeypatch.setattr(sbx, "_exec", _Calls(inspect))

    def build(args, timeout):
        built.append(1)
        starting.set()
        release.wait(5)
        return 0, b"", False
    monkeypatch.setattr(r, "_exec_tracked", build)

    async def go():
        warm = asyncio.create_task(r.prepare("flask"))          # the background pre-warm
        await asyncio.to_thread(starting.wait, 5)
        change_request = asyncio.create_task(r.prepare("flask"))  # a change request arrives mid-build
        await asyncio.sleep(0.05)
        assert not change_request.done()                         # it waits ...
        release.set()
        assert await warm is None and await change_request is None
        assert len(built) == 1                                   # ... and the image was built exactly once
    run(go())


def test_stop_kills_a_running_build_and_refuses_new_ones():
    r = sbx.DockerRunner()
    out = {}

    def long_build():
        out["res"] = r._exec_tracked([sys.executable, "-c", "import time; time.sleep(60)"], 90)
    t = threading.Thread(target=long_build)
    t0 = time.monotonic()
    t.start()
    deadline = time.monotonic() + 10
    while not r._procs and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r._procs                                              # the build process is tracked
    r.stop()
    t.join(15)
    assert not t.is_alive() and time.monotonic() - t0 < 14       # not held for the 60 s the child wanted
    assert out["res"][0] != 0 and out["res"][2] is False         # terminated, not reported as a timeout
    rc, msg, timed = r._exec_tracked([sys.executable, "-c", "print(1)"], 5)
    assert rc == 1 and b"shutting down" in msg and not r._procs  # nothing new is started after stop()


def test_a_build_killed_by_shutdown_is_not_remembered_as_a_failure(monkeypatch):
    r = sbx.DockerRunner()
    monkeypatch.setattr(sbx, "_exec", _Calls(lambda a: (1, b"", False)))

    def killed(args, timeout):
        r.stop()                                                 # shutdown arrives while the build runs
        return -15, b"terminated", False
    monkeypatch.setattr(r, "_exec_tracked", killed)
    err = run(r.prepare("x==1"))
    assert err and r._failed == {}


def test_list_and_remove_only_touch_this_apps_images(monkeypatch):
    r = sbx.DockerRunner()
    pre = settings.sandbox_image_prefix
    listing = f"{pre}:aaa\n{pre}:bbb\n{pre}:<none>\nsomeone/else:latest\n\n"
    c = _Calls(lambda a: (0, listing.encode(), False))
    monkeypatch.setattr(sbx, "_exec", c)
    assert run(r.list_images()) == [f"{pre}:aaa", f"{pre}:bbb"]
    assert run(r.remove_image("someone/else:latest")) is False
    assert all(a[:2] != ["docker", "rmi"] for a in c.argvs)       # a foreign tag never reaches docker
    assert run(r.remove_image(f"{pre}:aaa")) is True
    assert c.argvs[-1] == ["docker", "rmi", f"{pre}:aaa"] and "--force" not in c.argvs[-1] and "-f" not in c.argvs[-1]


# ============================================================================= SandboxService.prewarm
class FakeDocker(sbx.LocalRunner):
    """Runs tests locally (like the Phase 2 tests) but presents itself as the docker backend."""
    name = "docker"

    def __init__(self):
        self.up, self.prepares, self.errors, self.images, self.removed = True, [], [], [], []

    def image_tag(self, requirements):
        return f"{settings.sandbox_image_prefix}:" + hashlib.sha256(requirements.encode()).hexdigest()[:12]

    async def available(self):
        return (self.up, "fake 1.0") if self.up else (False, "Docker is installed but the daemon is not reachable")

    async def prepare(self, requirements):
        self.prepares.append(requirements)
        return self.errors.pop(0) if self.errors else None

    async def list_images(self):
        return list(self.images)

    async def remove_image(self, tag):
        self.removed.append(tag)
        return True


async def _svc(files, runner=None, enabled=True):
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, files)
    emitted = []
    svc = sbx.SandboxService(s, runner or FakeDocker(), enabled=enabled,
                             emit=lambda *a: emitted.append(a[1:]))
    svc.emitted = emitted
    return svc, s, r


def test_prewarm_builds_once_and_the_tag_is_the_one_verify_uses():
    async def go():
        svc, s, r = await _svc([("calc.py", GOOD), ("requirements.txt", b"requests==2.0\n")])
        res = await svc.prewarm(r)
        assert res["status"] == "ready" and svc.runner.prepares == ["requests==2.0\n"]
        tree = await s.snapshot_bytes(r, await s.latest_rev(r))
        assert res["tag"] == svc.runner.image_tag(svc._requirements(tree))    # same tag as verify() computes

        again = await svc.prewarm(r)
        assert again["status"] == "ready" and again["note"] == "image already prepared"
        assert len(svc.runner.prepares) == 1                                   # no second build / inspect
    run(go())


def test_requirements_for_matches_verify_for_every_layout_and_after_a_revision():
    layouts = [
        [("requirements.txt", b"a==1\n")],
        [("requirements.txt", b"a==1\n"), ("requirements-test.txt", b"pytest==8\n")],           # both files are merged
        [("requirements.txt", b"a==1\n"), ("requirements-dev.txt", b"b==2\n")],                 # both files are merged
        [("sub/requirements.txt", b"nested==1\n")],                                             # only the repo root counts
        [],
    ]

    async def go():
        for extra in layouts:
            svc, s, r = await _svc([("calc.py", GOOD)] + extra)
            for _ in range(2):
                rev = await s.latest_rev(r)
                tree = await s.snapshot_bytes(r, rev)
                assert await svc.requirements_for(r) == svc._requirements(tree)
                # a proposal revision that edits the requirements: warm-up must follow the latest revision
                await s.create_revision(r, rev, "bump", {"requirements.txt": b"a==2\n"}, {})
    run(go())


def test_verify_after_prewarm_does_not_build_again_and_without_it_builds_once():
    async def go():
        files = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC), ("requirements.txt", b"x==1\n")]
        svc, s, r = await _svc(files)
        await svc.prewarm(r)
        v = await svc.verify(r, 0, {"calc.py": GOOD + b"\n# touched\n"})
        assert v.ran and v.ok and v.report.passed == 1
        assert len(svc.runner.prepares) == 1                                   # verify reused the warmed image

        cold, s2, r2 = await _svc(files)                                       # unchanged behaviour without a warm-up
        v = await cold.verify(r2, 0, {"calc.py": GOOD + b"\n# touched\n"})
        assert v.ran and v.ok and len(cold.runner.prepares) == 1
    run(go())


def test_prewarm_skips_instead_of_failing_when_there_is_nothing_to_do():
    async def go():
        svc, s, r = await _svc([("calc.py", GOOD)], enabled=False)
        assert (await svc.prewarm(r))["status"] == "skipped" and svc.runner.prepares == []

        svc, s, r = await _svc([("calc.py", GOOD)], runner=sbx.LocalRunner())
        res = await svc.prewarm(r)
        assert res["status"] == "skipped" and "needs no image" in res["note"]

        svc, s, r = await _svc([("README.md", b"# docs\n"), ("requirements.txt", b"x==1\n")])
        res = await svc.prewarm(r)
        assert res["status"] == "skipped" and "no Python" in res["note"] and svc.runner.prepares == []

        svc, s, r = await _svc([("calc.py", GOOD)])
        svc.runner.up = False
        res = await svc.prewarm(r)
        assert res["status"] == "skipped" and res["retry"] is True and "daemon" in res["note"]   # ask again later
        assert svc.runner.prepares == []
    run(go())


def test_failed_prewarm_warns_does_not_mark_ready_and_can_succeed_later():
    async def go():
        svc, s, r = await _svc([("calc.py", GOOD), ("requirements.txt", b"x==1\n")])
        svc.runner.errors = ["image build failed: no network"]
        res = await svc.prewarm(r)
        assert res["status"] == "failed" and "no network" in res["note"]
        assert any(lvl == "WARNING" and "could not be built" in msg for msg, lvl in
                   [(e[0], e[1]) for e in svc.emitted])
        assert svc.runner.image_tag("x==1\n") not in svc._ready
        res = await svc.prewarm(r)                                              # network is back
        assert res["status"] == "ready" and len(svc.runner.prepares) == 2
    run(go())


def test_prune_removes_only_images_no_repo_needs():
    async def go():
        svc, s, keep_repo = await _svc([("calc.py", GOOD), ("requirements.txt", b"keep==1\n")])
        pre = settings.sandbox_image_prefix
        keep_tag = svc.runner.image_tag("keep==1\n")
        svc.runner.images = [keep_tag, f"{pre}:old111", f"{pre}:old222"]
        svc._ready.update(svc.runner.images)
        removed = await svc.prune_images([keep_repo])
        assert sorted(removed) == [f"{pre}:old111", f"{pre}:old222"] and svc.runner.removed == removed
        assert keep_tag in svc._ready and f"{pre}:old111" not in svc._ready     # verify would rebuild a removed one

        svc.runner.up = False
        assert await svc.prune_images([keep_repo]) == []                         # Docker down: touch nothing
    run(go())


# ============================================================================= SandboxWarmer
class FakeSandbox:
    def __init__(self, results=None, gate=None):
        self.calls, self.inflight, self.max_inflight = [], 0, 0
        self.pruned, self.closed, self.emitted = None, False, []
        self.results, self.gate = results or {}, gate

    async def prewarm(self, repo_id, rev=None):
        self.calls.append(repo_id)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            r = self.results.get(repo_id)
            if isinstance(r, list):
                r = r.pop(0) if r else None
            if isinstance(r, Exception):
                raise r
            return r or {"repo_id": repo_id, "status": "ready", "note": ""}
        finally:
            self.inflight -= 1

    async def prune_images(self, repo_ids):
        self.pruned = list(repo_ids)
        return []

    def close(self):
        self.closed = True

    def _emit(self, msg, level="INFO"):
        self.emitted.append((level, msg))


def _ids(*ids):
    async def f():
        return list(ids)
    return f


@pytest.fixture
def warm_on(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_prewarm", True)
    monkeypatch.setattr(settings, "sandbox_prewarm_max_repos", 5)
    monkeypatch.setattr(settings, "sandbox_prune_images", False)


def test_startup_warms_recent_repos_first_one_at_a_time_and_never_blocks_start(warm_on, monkeypatch):
    monkeypatch.setattr(settings, "sandbox_prewarm_max_repos", 2)

    async def go():
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids("a", "b", "c"))
        w.start()
        assert sb.calls == []                       # start() returned before any work happened
        await w.join()
        assert sb.calls == ["a", "b"]               # most recent first, capped at sandbox_prewarm_max_repos
        assert sb.max_inflight == 1                 # never two builds at once
        assert w.results["a"]["status"] == "ready"
        await w.stop()
    run(go())


def test_disabled_by_setting_nothing_starts_and_requests_are_ignored(warm_on, monkeypatch):
    monkeypatch.setattr(settings, "sandbox_prewarm", False)

    async def go():
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids("a"))
        w.start()
        w.request("b")
        await asyncio.sleep(0.02)
        assert sb.calls == [] and w._loop is None
    run(go())


def test_requests_from_the_watcher_thread_are_safe_and_merged(warm_on):
    async def go():
        gate = asyncio.Event()
        sb = FakeSandbox(gate=gate)
        w = SandboxWarmer(sb, _ids())
        w.start()
        await w.join()
        w.request("z")
        await until(lambda: sb.calls == ["z"])      # z is building (blocked on the gate)

        def watcher_thread():                        # like RepoChangeHandler: another thread, another loop
            for rid_ in ("x", "x", "y", "x"):
                w.request(rid_)
        t = threading.Thread(target=watcher_thread)
        t.start()
        await asyncio.to_thread(t.join)
        await asyncio.sleep(0.05)
        assert list(w._queue) == ["x", "y"]         # repeated requests for a waiting repo were merged
        gate.set()
        await w.join()
        assert sb.calls == ["z", "x", "y"] and sb.max_inflight == 1
        await w.stop()
    run(go())


def test_docker_not_up_yet_is_retried_then_gives_up(warm_on, monkeypatch):
    monkeypatch.setattr(warmer_mod, "_RETRY_S", 0.01)

    async def go():
        down = {"repo_id": "a", "status": "skipped", "retry": True, "note": "daemon not reachable"}
        sb = FakeSandbox(results={"a": [dict(down), dict(down)]})    # down twice, then the default: ready
        w = SandboxWarmer(sb, _ids("a"))
        w.start()
        await until(lambda: w.results.get("a", {}).get("status") == "ready")
        assert sb.calls == ["a", "a", "a"] and "a" not in w._attempts
        await w.stop()

        monkeypatch.setattr(warmer_mod, "_MAX_RETRIES", 2)
        sb2 = FakeSandbox(results={"a": [dict(down)] * 10})
        w2 = SandboxWarmer(sb2, _ids("a"))
        w2.start()
        await until(lambda: len(sb2.calls) == 3)                      # first try + 2 retries
        await asyncio.sleep(0.15)
        assert len(sb2.calls) == 3                                    # then it leaves it to the first change request
        await w2.stop()
    run(go())


def test_one_crashing_warm_up_does_not_stop_the_others(warm_on):
    async def go():
        sb = FakeSandbox(results={"a": RuntimeError("boom")})
        w = SandboxWarmer(sb, _ids("a", "b"))
        w.start()
        await w.join()
        assert sb.calls == ["a", "b"]
        assert w.results["a"]["status"] == "failed" and "boom" in w.results["a"]["note"]
        assert w.results["b"]["status"] == "ready"
        assert any(lvl == "WARNING" and "crashed" in msg for lvl, msg in sb.emitted)
        await w.stop()
    run(go())


def test_listing_repos_can_fail_without_taking_the_app_down(warm_on):
    async def boom():
        raise OSError("db locked")

    async def go():
        sb = FakeSandbox()
        w = SandboxWarmer(sb, boom)
        w.start()
        await w.join()
        assert sb.calls == [] and any(lvl == "WARNING" for lvl, _ in sb.emitted)
        await w.stop()
    run(go())


def test_stop_cancels_running_work_pending_retries_and_later_requests(warm_on, monkeypatch):
    monkeypatch.setattr(warmer_mod, "_RETRY_S", 0.05)

    async def go():
        gate = asyncio.Event()                       # never opened: a build that would run for minutes
        sb = FakeSandbox(gate=gate)
        w = SandboxWarmer(sb, _ids("a"))
        w.start()
        await until(lambda: sb.calls == ["a"])
        await asyncio.wait_for(w.stop(), 2)          # returns promptly instead of waiting for the build
        assert sb.closed                             # the sandbox was told to kill its docker build
        w.request("late")
        await asyncio.sleep(0.05)
        assert sb.calls == ["a"]

        down = {"repo_id": "r", "status": "skipped", "retry": True, "note": "down"}
        sb2 = FakeSandbox(results={"r": [dict(down)] * 5})
        w2 = SandboxWarmer(sb2, _ids("r"))
        w2.start()
        await until(lambda: sb2.calls == ["r"])
        await w2.join()
        await w2.stop()                              # a retry timer is pending right now
        await asyncio.sleep(0.15)
        assert sb2.calls == ["r"]                    # it never fired
    run(go())


def test_prune_runs_after_warming_with_every_repo_and_only_when_enabled(warm_on, monkeypatch):
    monkeypatch.setattr(settings, "sandbox_prewarm_max_repos", 1)

    async def go():
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids("a", "b", "c"))
        w.start()
        await w.join()
        assert sb.calls == ["a"] and sb.pruned is None               # off by default

        monkeypatch.setattr(settings, "sandbox_prune_images", True)
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids("a", "b", "c"))
        w.start()
        await w.join()
        assert sb.calls == ["a"] and sb.pruned == ["a", "b", "c"]    # b and c are warmed later, but their images are kept

        monkeypatch.setattr(settings, "sandbox_prewarm_max_repos", 0)   # prune must not depend on a warm-up happening
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids("a", "b"))
        w.start()
        await w.join()
        assert sb.calls == [] and sb.pruned == ["a", "b"]

        sb = FakeSandbox()                                              # no repos at all: never "keep nothing"
        w = SandboxWarmer(sb, _ids())
        w.start()
        await w.join()
        assert sb.pruned is None
    run(go())


def test_module_level_request_prewarm_reaches_the_running_warmer_only(warm_on):
    async def go():
        request_prewarm("nobody-listening")          # no warmer running (CLI, tests): silently nothing
        sb = FakeSandbox()
        w = SandboxWarmer(sb, _ids())
        w.start()
        request_prewarm("q")
        await w.join()
        assert sb.calls == ["q"]
        await w.stop()
        request_prewarm("after-stop")
        await asyncio.sleep(0.02)
        assert sb.calls == ["q"]
    run(go())


def test_is_requirements_path_is_the_repo_root_files_only():
    assert is_requirements_path("requirements.txt") and is_requirements_path("requirements-test.txt")
    assert is_requirements_path("requirements-dev.txt")
    assert not is_requirements_path("sub/requirements.txt") and not is_requirements_path("app.py")
    assert not is_requirements_path("sub\\requirements.txt")


# ============================================================================= startup ordering: most recently used repos first
def test_repos_are_listed_most_recently_chatted_first_without_the_general_placeholder():
    async def go():
        db = DatabaseManager()
        await db.init_db()
        ids = {k: rid() for k in "abcdn"}
        for k, status in (("a", JobStatus.COMPLETED), ("b", JobStatus.COMPLETED), ("c", JobStatus.COMPLETED),
                          ("d", JobStatus.PROCESSING), ("n", JobStatus.COMPLETED)):
            await db.upsert_repo(ids[k], k, f"/tmp/{k}", status, 1)
        for k in "ab":
            await db.create_thread(f"t_{ids[k]}", ids[k], "t")
            await db.save_chat_message(f"t_{ids[k]}", "user", "hi")
        con = sqlite3.connect(db.db_path)             # fixed timestamps: CURRENT_TIMESTAMP has 1 s resolution
        con.execute("UPDATE chat_messages SET timestamp='2026-02-01 10:00:00' WHERE thread_id=?", (f"t_{ids['a']}",))
        con.execute("UPDATE chat_messages SET timestamp='2026-03-01 10:00:00' WHERE thread_id=?", (f"t_{ids['b']}",))
        con.execute("UPDATE repositories SET created_at='2026-01-01T00:00:00' WHERE repo_id=?", (ids["c"],))
        con.execute("UPDATE repositories SET created_at='2026-04-01T00:00:00' WHERE repo_id=?", (ids["n"],))   # never chatted
        con.commit()
        con.close()
        order = [i for i in await db.list_repo_ids_by_activity() if i in ids.values()]
        assert order == [ids["n"], ids["b"], ids["a"], ids["c"]]     # d is still PROCESSING: not listed
        assert GENERAL_REPO_ID not in await db.list_repo_ids_by_activity()
    run(go())


# ============================================================================= controller wiring (needs the full app stack)
class _FakeDB:
    def __init__(self):
        self.repos = []

    async def init_db(self):
        pass

    async def upsert_repo(self, *a):
        self.repos.append(a[0])

    async def get_completed_file_hashes(self, repo_id):
        return {}

    async def record_file_job(self, *a, **k):
        pass

    async def update_repo_status(self, *a):
        pass

    async def get_repo(self, repo_id):
        return repo_id


class _NoVectors:
    def delete_chunks_by_file(self, *a):
        pass

    def embed_and_persist_chunks(self, *a):
        pass


class _NoChunks:
    def chunk_file(self, *a, **k):
        return []


def _bare_controller(monkeypatch):
    ec = pytest.importorskip("controllers.engine_controller")
    calls = []
    monkeypatch.setattr(ec, "request_prewarm", calls.append)
    monkeypatch.setattr(ec.ChunkingRegistry, "get_strategy", staticmethod(lambda ext: _NoChunks()))
    ctrl = ec.EngineController.__new__(ec.EngineController)
    ctrl.db, ctrl.vector_service, ctrl.repo_store = _FakeDB(), _NoVectors(), RepoStore()
    return ec, ctrl, calls


def test_changing_a_requirements_file_on_disk_requests_a_warm_up_other_files_do_not(tmp_path, monkeypatch):
    ec, ctrl, calls = _bare_controller(monkeypatch)
    (tmp_path / "sub").mkdir()
    for name, text in (("requirements.txt", "flask==3\n"), ("app.py", "x = 1\n"), ("sub/requirements.txt", "y\n")):
        (tmp_path / name).write_text(text)

    async def go():
        await ctrl.incremental_reingest_file("repo1", tmp_path, "app.py")
        await ctrl.incremental_reingest_file("repo1", tmp_path, "sub/requirements.txt")
        assert calls == []
        await ctrl.incremental_reingest_file("repo1", tmp_path, "requirements.txt")
        assert calls == ["repo1"]
    run(go())


def test_ingest_requests_a_warm_up_right_after_the_raw_snapshot_and_before_chunking(tmp_path, monkeypatch):
    ec, ctrl, calls = _bare_controller(monkeypatch)
    order = []

    async def snapshot(repo_id, root):
        order.append("snapshot")
    monkeypatch.setattr(ctrl, "snapshot_raw_files", snapshot)
    monkeypatch.setattr(ec, "request_prewarm", lambda rid_: order.append(("warm", rid_)))
    monkeypatch.setattr(ctrl, "scan_supported_files", lambda root: [])

    out = run(ctrl.ingest_repository(str(tmp_path)))
    assert order == ["snapshot", ("warm", ctrl.compute_repo_id(tmp_path))] and out


def test_controller_owns_a_warmer_that_lists_repos_by_activity(monkeypatch):
    ec = pytest.importorskip("controllers.engine_controller")
    ctrl = ec.EngineController()
    assert isinstance(ctrl.warmer, SandboxWarmer) and ctrl.warmer.sandbox is ctrl.change_service.sandbox
    assert ctrl.warmer._loop is None                  # constructing a controller (the file watcher does it per event) starts nothing

    async def go():
        a, b = rid(), rid()
        await ctrl.db.init_db()
        await ctrl.db.upsert_repo(a, "a", "/tmp/a", JobStatus.COMPLETED, 1)
        await ctrl.db.upsert_repo(b, "b", "/tmp/b", JobStatus.COMPLETED, 1)
        ids = await ctrl._warm_repo_ids()
        assert a in ids and b in ids and GENERAL_REPO_ID not in ids
    run(go())
