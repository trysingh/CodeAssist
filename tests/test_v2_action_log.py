import asyncio
import contextvars
import json
import threading

import pytest

from services import action_log as al


@pytest.fixture(autouse=True)
def _clean():
    al.clear_sinks()
    al.ring.clear()
    yield
    al.clear_sinks()
    al.ring.clear()


def test_log_action_basic_fields():
    ev = al.log_action("INTAKE", "IntakeAgent", "analyze", detail="hi", data={"a": 1}, model="m", duration_ms=5)
    d = ev.to_dict()
    assert d["phase"] == "INTAKE" and d["status"] == "info" and d["data"] == {"a": 1}
    assert d["ts"].endswith("Z") and len(d["uid"]) == 32
    assert al.recent()[-1]["uid"] == ev.uid


def test_request_scope_sets_and_resets():
    assert al.current_request_id() is None
    with al.request_scope("r1", "t1") as rid:
        assert rid == "r1" and al.current_request_id() == "r1" and al.current_thread_id() == "t1"
        ev = al.log_action("LLM", "x", "y")
        assert ev.request_id == "r1" and ev.thread_id == "t1"
    assert al.current_request_id() is None and al.current_thread_id() is None


def test_request_scope_generates_id():
    with al.request_scope() as rid:
        assert rid and al.current_request_id() == rid


def test_context_propagates_to_asyncio_tasks_and_isolated():
    async def work(name):
        with al.request_scope(name):
            await asyncio.sleep(0.01)
            child = asyncio.create_task(asyncio.sleep(0, result=al.current_request_id()))
            await asyncio.sleep(0.01)
            ev = al.log_action("LLM", "a", "b")
            return ev.request_id, await child

    async def main():
        return await asyncio.gather(*[work(f"req{i}") for i in range(5)])
    res = asyncio.run(main())
    assert res == [(f"req{i}", f"req{i}") for i in range(5)]


def test_context_propagates_to_to_thread_but_not_raw_threads():
    seen = {}

    async def main():
        with al.request_scope("rq"):
            seen["to_thread"] = await asyncio.to_thread(lambda: al.current_request_id())
            ctx = contextvars.copy_context()
            out = []
            t = threading.Thread(target=lambda: out.append(ctx.run(al.current_request_id)))
            t.start(); t.join()
            seen["copied"] = out[0]
            raw = []
            t = threading.Thread(target=lambda: raw.append(al.current_request_id()))
            t.start(); t.join()
            seen["raw"] = raw[0]
    asyncio.run(main())
    assert seen == {"to_thread": "rq", "copied": "rq", "raw": None}


def test_sinks_called_and_errors_swallowed(capsys):
    got = []

    def bad(ev):
        raise RuntimeError("boom")
    al.add_sink(bad)
    al.add_sink(got.append)
    ev = al.log_action("LLM", "a", "b")        # must not raise
    assert got == [ev]
    assert "sink error" in capsys.readouterr().err


def test_set_sink_replaces_and_clear():
    a, b = [], []
    al.add_sink(a.append)
    al.set_sink(b.append)
    al.log_action("LLM", "a", "b")
    assert a == [] and len(b) == 1
    al.clear_sinks()
    al.log_action("LLM", "a", "b")
    assert len(b) == 1
    al.set_sink(None)


def test_action_ok_start_and_close():
    with al.action("TEST", "Ag", "run", detail="d", data={"x": 1}) as h:
        h.note("finished", n=3)
    evs = al.recent()
    assert [e["status"] for e in evs] == ["start", "ok"]
    assert evs[1]["parent_uid"] == evs[0]["uid"]
    assert evs[1]["detail"] == "finished" and evs[1]["data"] == {"x": 1, "n": 3}
    assert evs[1]["duration_ms"] is not None and evs[1]["duration_ms"] >= 0


def test_action_fail_reraises():
    with pytest.raises(ValueError):
        with al.action("TEST", "Ag", "run"):
            raise ValueError("nope")
    last = al.recent()[-1]
    assert last["status"] == "fail" and "nope" in last["data"]["error"] and "ValueError" in last["detail"]


def test_action_status_override():
    with al.action("TEST", "Ag", "run") as h:
        h.note("x", status="skip")
    assert al.recent()[-1]["status"] == "skip"


def test_traced_sync_and_async():
    @al.traced("TEST", "Ag")
    def add(a, b):
        return a + b

    @al.traced("TEST", "Ag", "custom")
    async def aadd(a, b):
        await asyncio.sleep(0)
        return a + b
    assert add(1, 2) == 3
    assert asyncio.run(aadd(2, 3)) == 5
    assert add.__name__ == "add" and asyncio.iscoroutinefunction(aadd)
    acts = [(e["action"], e["status"]) for e in al.recent()]
    assert acts == [("add", "start"), ("add", "ok"), ("custom", "start"), ("custom", "ok")]


def test_traced_async_failure():
    @al.traced("TEST", "Ag")
    async def boom():
        raise KeyError("k")
    with pytest.raises(KeyError):
        asyncio.run(boom())
    assert al.recent()[-1]["status"] == "fail"


def test_ring_buffer_filters_and_limit():
    for i in range(10):
        with al.request_scope(f"r{i % 2}", repo_id=f"repo{i % 3}"):
            al.log_action("LLM", "a", f"n{i}")
    assert len(al.recent(limit=3)) == 3 and al.recent(limit=3)[-1]["action"] == "n9"
    assert all(e["request_id"] == "r0" for e in al.recent(request_id="r0"))
    assert len(al.recent(request_id="r0")) == 5
    r = al.recent(repo_id="repo1")
    assert r and all(e["repo_id"] == "repo1" for e in r)
    assert al.recent(request_id="r0", repo_id="repo0", limit=100)
    assert al.recent(limit=0) == []


def test_ring_capacity_cap():
    rb = al.RingBuffer(capacity=5)
    for i in range(12):
        rb.append(al.ActionEvent(uid=str(i), ts="t", request_id=None, thread_id=None, repo_id=None,
                                 phase="LLM", agent="a", action="b", status="info"))
    assert len(rb) == 5 and rb.recent()[0]["uid"] == "7"
    assert al.RING_CAPACITY == 5000


def test_truncation_keeps_under_4kb_and_json_safe():
    big = {"text": "x" * 50000, "items": list(range(5000)), "obj": object()}
    ev = al.log_action("LLM", "a", "b", data=big)
    s = json.dumps(ev.data)
    assert len(s.encode()) <= al.MAX_DATA_BYTES
    assert ev.data.get("_truncated") is True


def test_small_data_untouched_and_non_dict():
    assert al.log_action("LLM", "a", "b", data={"k": [1, 2]}).data == {"k": [1, 2]}
    assert al.log_action("LLM", "a", "b", data=None).data == {}
    assert al.log_action("LLM", "a", "b", data=[1, 2]).data == {"value": [1, 2]}


def test_thread_safety_many_threads():
    def work():
        for _ in range(200):
            al.log_action("LLM", "a", "b")
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(al.ring) == 1600


def test_notifier_defaults_use_contextvars_only_via_action_log():
    # request_scope values are what notifier.notify() reads for its defaults
    with al.request_scope("abc", "thr"):
        assert (al.current_request_id(), al.current_thread_id()) == ("abc", "thr")
