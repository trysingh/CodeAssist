import asyncio

from services.vectorize_jobs import (DbJobStore, InMemoryJobStore, Job, JobState, VectorizeJobManager)


class Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, title, message, level="INFO", repo_id=None, **kw):
        self.calls.append({"title": title, "message": message, "level": level, "repo_id": repo_id, **kw})


def make(work, **kw):
    notes, statuses = Rec(), []

    async def setter(repo, status):
        statuses.append((repo, status))
    store = InMemoryJobStore()
    mgr = VectorizeJobManager(store, work, notes, setter, **kw)
    return mgr, store, notes, statuses


def test_success_flow_and_notifications():
    async def main():
        async def work(repo, cb):
            cb(0, 4)
            await asyncio.sleep(0)
            cb(4, 4)
        mgr, store, notes, statuses = make(work)
        job = await mgr.enqueue("r1", "weak search")
        assert job.state == JobState.QUEUED and mgr.status("r1") == "queued"
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.COMPLETED and done.progress == 4 and done.total == 4
        assert mgr.status("r1") == "ready"
        assert [s for _, s in statuses] == ["queued", "running", "ready"]
        first, last = notes.calls[0], notes.calls[-1]
        assert first["level"] == "INFO" and "Vectorizing r1 in background: weak search" in first["message"]
        assert "you'll be notified" in first["message"]
        assert last["level"] == "SUCCESS" and "Vector index ready" in last["message"] and "semantic search enabled" in last["message"]
        assert all(c["kind"] == "job" and c["repo_id"] == "r1" and c["data"]["job_id"] == job.job_id for c in notes.calls)
        stored = await store.get(job.job_id)
        assert stored.state == JobState.COMPLETED
        assert (await store.latest("r1", "vectorize")).job_id == job.job_id
        assert [j.job_id for j in await store.list("r1")] == [job.job_id]
        assert await store.list("other") == []
    asyncio.run(main())


def test_dedup_returns_existing_active_job_then_allows_new_one():
    async def main():
        gate = asyncio.Event()

        async def work(repo, cb):
            await gate.wait()
        mgr, _store, notes, _ = make(work)
        a = await mgr.enqueue("r1", "x")
        b = await mgr.enqueue("r1", "y")
        assert a.job_id == b.job_id
        c = await mgr.enqueue("r2", "z")
        assert c.job_id != a.job_id
        starts = [n for n in notes.calls if "in background" in n["message"]]
        assert len(starts) == 2                          # no duplicate start toast for the deduped call
        gate.set()
        await mgr.wait(a.job_id, 5)
        await mgr.wait(c.job_id, 5)
        d = await mgr.enqueue("r1", "again")
        assert d.job_id != a.job_id
        await mgr.wait(d.job_id, 5)
    asyncio.run(main())


def test_progress_throttling_by_percent():
    async def main():
        async def work(repo, cb):
            for i in range(1, 101):
                cb(i, 100)
        mgr, _s, notes, _ = make(work, clock=lambda: 0.0)
        job = await mgr.enqueue("r", "why")
        await mgr.wait(job.job_id, 5)
        prog = [n for n in notes.calls if n["title"] == "Vectorize progress"]
        # 10,20,...,90 (100% is announced by the "ready" notification instead)
        assert [p["data"]["progress"] for p in prog] == [10, 20, 30, 40, 50, 60, 70, 80, 90]
        assert all(p["level"] == "INFO" for p in prog)
    asyncio.run(main())


def test_progress_throttling_by_time():
    async def main():
        t = {"now": 0.0}

        async def work(repo, cb):
            cb(1, 1000)
            cb(2, 1000)           # +0.2% and no time elapsed: silent
            t["now"] = 16.0
            cb(3, 1000)           # 15s passed: notify even though < 10%
            cb(4, 1000)           # just notified: silent
        mgr, _s, notes, _ = make(work, clock=lambda: t["now"])
        job = await mgr.enqueue("r", "why")
        await mgr.wait(job.job_id, 5)
        prog = [n for n in notes.calls if n["title"] == "Vectorize progress"]
        assert len(prog) == 1 and prog[0]["data"]["progress"] == 3
    asyncio.run(main())


def test_progress_from_worker_thread():
    async def main():
        async def work(repo, cb):
            def blocking():
                for i in range(1, 11):
                    cb(i, 10)
            await asyncio.to_thread(blocking)
        mgr, _s, notes, _ = make(work, clock=lambda: 0.0)
        job = await mgr.enqueue("r", "why")
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.COMPLETED
        assert len([n for n in notes.calls if n["title"] == "Vectorize progress"]) == 9
    asyncio.run(main())


def test_failure_path():
    async def main():
        async def work(repo, cb):
            cb(1, 10)
            raise RuntimeError("embedding server down")
        mgr, store, notes, statuses = make(work)
        job = await mgr.enqueue("r", "why")
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.FAILED and "embedding server down" in done.error
        assert mgr.status("r") == "failed" and statuses[-1] == ("r", "failed")
        assert notes.calls[-1]["level"] == "ERROR" and "embedding server down" in notes.calls[-1]["message"]
        assert (await store.get(job.job_id)).state == JobState.FAILED
        # a failed job does not block a retry
        again = await mgr.enqueue("r", "retry")
        assert again.job_id != job.job_id
        await mgr.wait(again.job_id, 5)
    asyncio.run(main())


def test_cancel_running_job():
    async def main():
        started = asyncio.Event()

        async def work(repo, cb):
            started.set()
            await asyncio.sleep(60)
        mgr, store, notes, statuses = make(work)
        job = await mgr.enqueue("r", "why")
        await started.wait()
        assert await mgr.cancel(job.job_id) is True
        assert (await store.get(job.job_id)).state == JobState.CANCELLED
        assert mgr.status("r") == "none" and statuses[-1] == ("r", "none")
        assert await mgr.cancel(job.job_id) is False
        assert await mgr.cancel("nope") is False
        nxt = await mgr.enqueue("r", "again")             # slot is free after a cancel
        assert nxt.job_id != job.job_id
        await mgr.cancel(nxt.job_id)
    asyncio.run(main())


def test_cancel_queued_job_before_it_starts():
    async def main():
        gate = asyncio.Event()

        async def work(repo, cb):
            await gate.wait()
        mgr, store, _n, _s = make(work)
        a = await mgr.enqueue("a", "x")
        b = await mgr.enqueue("b", "y")                   # waits behind a (max_concurrent=1)
        await asyncio.sleep(0.01)
        assert (await store.get(b.job_id)).state == JobState.QUEUED
        assert await mgr.cancel(b.job_id) is True
        assert (await store.get(b.job_id)).state == JobState.CANCELLED
        gate.set()
        assert (await mgr.wait(a.job_id, 5)).state == JobState.COMPLETED
    asyncio.run(main())


def test_cancel_before_first_step():
    async def main():
        async def work(repo, cb):
            await asyncio.sleep(0)
        mgr, store, _n, _s = make(work)
        job = await mgr.enqueue("r", "x")
        assert await mgr.cancel(job.job_id) is True       # task never got to run
        assert (await store.get(job.job_id)).state == JobState.CANCELLED
        assert mgr.status("r") == "none"
    asyncio.run(main())


def test_notify_typeerror_fallback():
    async def main():
        calls = []

        def old_notify(title, message, level="INFO", repo_id=None):      # no **kwargs: kind= raises TypeError
            calls.append((title, message, level, repo_id))

        async def work(repo, cb):
            cb(5, 10)
        store = InMemoryJobStore()
        mgr = VectorizeJobManager(store, work, old_notify, None, clock=lambda: 0.0)
        job = await mgr.enqueue("r", "why")
        await mgr.wait(job.job_id, 5)
        assert calls[0][2] == "INFO" and calls[0][3] == "r"
        assert calls[-1][2] == "SUCCESS"
        assert mgr.status("r") == "ready"
    asyncio.run(main())


def test_notify_and_status_setter_exceptions_do_not_break_job():
    async def main():
        def bad_notify(*a, **k):
            raise RuntimeError("sse down")

        async def bad_setter(repo, status):
            raise RuntimeError("db locked")

        async def work(repo, cb):
            cb(1, 2)
        mgr = VectorizeJobManager(InMemoryJobStore(), work, bad_notify, bad_setter)
        job = await mgr.enqueue("r", "why")
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.COMPLETED and mgr.status("r") == "ready"
    asyncio.run(main())


def test_work_does_not_block_loop():
    async def main():
        async def work(repo, cb):
            await asyncio.to_thread(lambda: __import__("time").sleep(0.2))
        mgr, _s, _n, _st = make(work)
        job = await mgr.enqueue("r", "why")
        ticks = 0
        while not (await mgr.get(job.job_id)).state == JobState.COMPLETED:
            ticks += 1
            await asyncio.sleep(0.01)
        assert ticks >= 5
    asyncio.run(main())


def test_job_roundtrip_and_db_store_adapter():
    j = Job(job_id="j1", repo_id="r", reason="because", progress=3, total=9)
    d = j.to_dict()
    assert d["state"] == "QUEUED" and d["kind"] == "vectorize" and Job.from_dict(d) == j

    class FakeDb:
        def __init__(self):
            self.rows = {}

        async def upsert_job(self, d):
            self.rows[d["job_id"]] = dict(d)

        async def get_job(self, job_id):
            return self.rows.get(job_id)

        async def list_jobs(self, repo_id=None, limit=50):
            return [r for r in self.rows.values() if repo_id in (None, r["repo_id"])][:limit]

        async def latest_job(self, repo_id, kind):
            m = [r for r in self.rows.values() if r["repo_id"] == repo_id and r["kind"] == kind]
            return m[-1] if m else None

    async def main():
        st = DbJobStore(FakeDb())
        await st.upsert(j)
        assert (await st.get("j1")) == j
        assert (await st.get("zz")) is None
        assert [x.job_id for x in await st.list("r")] == ["j1"]
        assert (await st.latest("r", "vectorize")).job_id == "j1"
        assert await st.latest("q", "vectorize") is None

        async def work(repo, cb):
            cb(1, 1)
        mgr = VectorizeJobManager(st, work, Rec())
        job = await mgr.enqueue("r2", "w")
        await mgr.wait(job.job_id, 5)
        assert (await st.get(job.job_id)).state == JobState.COMPLETED
    asyncio.run(main())
