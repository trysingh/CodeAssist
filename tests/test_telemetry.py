"""Telemetry & logs persistence: SQLite rows, general-vs-repo visibility, last-N history, live fan-out.
Run from the repo root:  pytest tests/test_telemetry.py -q   (no Ollama, Chroma, Docker or network needed)
"""
import asyncio
import json
import threading
import uuid

from config.settings import settings
from models.database import DatabaseManager
from models.schemas import GENERAL_REPO_ID
from services.notifier import NotificationDispatcher, repo_scope

run = asyncio.run


def _ev(i, scope="general", repo=None, title="T"):
    return {"uid": f"u{i}", "ts": f"2026-10-23T10:00:{i:02d}.000Z", "scope": scope, "repo_id": repo,
            "level": "INFO", "title": title, "message": f"m{i}"}


async def _db():
    db = DatabaseManager()
    await db.init_db()
    await db.prune_telemetry_events(1)          # shared test DB: start from (almost) nothing
    return db


# --------------------------------------------------------------------------- database
def test_history_is_scoped_ordered_and_limited():
    async def go():
        db = await _db()
        a, b = f"A{uuid.uuid4().hex[:6]}", f"B{uuid.uuid4().hex[:6]}"
        base = uuid.uuid4().hex[:6]
        evs = [dict(_ev(i, "general" if i % 3 == 0 else "repo", None if i % 3 == 0 else (a if i % 3 == 1 else b)),
                    uid=f"{base}{i}") for i in range(1, 13)]
        await db.save_telemetry_events(evs)

        got_a = await db.get_telemetry_events(a, 100)
        assert {e["scope"] for e in got_a} == {"general", "repo"}
        assert all(e["repo_id"] in (None, a) for e in got_a)            # never another repo's events
        assert [e["ts"] for e in got_a] == sorted(e["ts"] for e in got_a)   # oldest first

        only_general = await db.get_telemetry_events(None, 100)
        assert only_general and all(e["scope"] == "general" for e in only_general)
        assert await db.get_telemetry_events(GENERAL_REPO_ID, 100) == only_general   # placeholder = general only

        last2 = await db.get_telemetry_events(a, 2)
        assert [e["uid"] for e in last2] == [e["uid"] for e in got_a][-2:]   # LAST N, still oldest-first
        assert await db.get_telemetry_events(a, 0) == []
    run(go())


def test_duplicate_uid_is_ignored_and_prune_keeps_newest():
    async def go():
        db = await _db()
        u = uuid.uuid4().hex
        await db.save_telemetry_events([dict(_ev(1), uid=u), dict(_ev(1), uid=u)])
        assert len([e for e in await db.get_telemetry_events(None, 1000) if e["uid"] == u]) == 1
        await db.save_telemetry_events([dict(_ev(i), uid=f"{u}-{i}") for i in range(2, 8)])
        removed = await db.prune_telemetry_events(3)
        left = await db.get_telemetry_events(None, 1000)
        assert removed >= 4 and len(left) == 3 and left[-1]["uid"] == f"{u}-7"
    run(go())


# --------------------------------------------------------------------------- notifier
async def _drain(gen, n, timeout=2.0):
    out = []
    async def pull():
        async for item in gen:
            out.append(item)
            if len(out) >= n:
                return
    await asyncio.wait_for(pull(), timeout)
    return out


def test_events_are_classified_persisted_and_replayed_per_repo():
    async def go():
        db = await _db()
        n = NotificationDispatcher()
        n.bind(db)
        repo = f"R{uuid.uuid4().hex[:6]}"
        other = f"O{uuid.uuid4().hex[:6]}"
        n.notify("Startup", "general one")                                         # no repo -> general
        n.notify("Watcher Active", "repo one", repo_id=repo)                       # explicit repo
        n.notify("Telemetry Trace", {"repo_id": other, "phase": "PATCH"})          # repo from the JSON payload
        n.notify("Telemetry Trace", {"repo_id": GENERAL_REPO_ID})                  # General placeholder -> general
        with repo_scope(repo):
            n.notify("Sandbox", "scoped by context")                               # repo from repo_scope
        t = threading.Thread(target=lambda: n.notify("Thread", "from a worker thread", repo_id=repo))
        t.start(); t.join()
        await asyncio.sleep(0.05)                                                  # writer batches and commits

        # selected repo: general + own events, never the other repo's
        gen = n.subscribe(repo_id=repo, limit=50)
        hist = json.loads((await _drain(gen, 1))[0]["data"])
        await gen.aclose()
        titles = [(e["title"], e["scope"]) for e in hist["events"]]
        assert ("Startup", "general") in titles and ("Watcher Active", "repo") in titles
        assert ("Sandbox", "repo") in titles and ("Thread", "repo") in titles
        assert ("Telemetry Trace", "general") in titles                           # the placeholder one
        assert all(e["repo_id"] in (None, repo) for e in hist["events"])
        assert not any(e["repo_id"] == other for e in hist["events"])

        # nothing selected: general only
        gen = n.subscribe(repo_id=None, limit=50)
        hist = json.loads((await _drain(gen, 1))[0]["data"])
        await gen.aclose()
        assert hist["events"] and all(e["scope"] == "general" for e in hist["events"])
        await n.close()
    run(go())


def test_live_stream_filters_by_repo_and_does_not_duplicate_history():
    async def go():
        db = await _db()
        n = NotificationDispatcher()
        n.bind(db)
        repo, other = f"R{uuid.uuid4().hex[:6]}", f"O{uuid.uuid4().hex[:6]}"
        n.notify("Before", "stored before connect", repo_id=repo)
        await asyncio.sleep(0.05)

        gen = n.subscribe(repo_id=repo, limit=10)
        first = await _drain(gen, 1)
        assert first[0]["event"] == "history"
        assert any(e["title"] == "Before" for e in json.loads(first[0]["data"])["events"])

        n.notify("OtherRepo", "must not reach this tab", repo_id=other)
        n.notify("Everyone", "general live")
        n.notify("Mine", "repo live", repo_id=repo)
        live = await _drain(gen, 2)
        await gen.aclose()
        assert [json.loads(x["data"])["title"] for x in live] == ["Everyone", "Mine"]
        assert all(x["event"] == "notification" for x in live)
        await n.close()
    run(go())


def test_limit_is_clamped_and_zero_means_live_only(monkeypatch):
    async def go():
        db = await _db()
        n = NotificationDispatcher()
        n.bind(db)
        for i in range(8):
            n.notify("Bulk", f"row {i}")
        await asyncio.sleep(0.05)
        monkeypatch.setattr(settings, "telemetry_history_max", 3)
        gen = n.subscribe(repo_id=None, limit=500)
        h = json.loads((await _drain(gen, 1))[0]["data"])
        await gen.aclose()
        assert h["limit"] == 3 and len(h["events"]) == 3                      # server-side ceiling
        gen = n.subscribe(repo_id=None, limit=0)
        h = json.loads((await _drain(gen, 1))[0]["data"])
        await gen.aclose()
        assert h["events"] == []
        monkeypatch.setattr(settings, "telemetry_history_default", 2)
        gen = n.subscribe(repo_id=None)                                        # no limit sent: configured default
        h = json.loads((await _drain(gen, 1))[0]["data"])
        await gen.aclose()
        assert h["limit"] == 2 and len(h["events"]) == 2
        await n.close()
    run(go())


def test_unbound_notifier_and_persist_switch_off_still_broadcast(monkeypatch):
    async def go():
        n = NotificationDispatcher()                                           # never bound: legacy behaviour
        gen = n.subscribe(repo_id=None, limit=5)
        h = await _drain(gen, 1)
        assert json.loads(h[0]["data"])["events"] == []
        n.notify("Live", "no db")
        assert json.loads((await _drain(gen, 1))[0]["data"])["title"] == "Live"
        await gen.aclose()

        monkeypatch.setattr(settings, "telemetry_persist", False)
        n2 = NotificationDispatcher()
        n2.bind(await _db())
        assert n2._writer is None                                              # nothing is stored when switched off
        n2.notify("X", "y")
        await n2.close()
    run(go())


def test_close_flushes_events_queued_right_before_shutdown():
    async def go():
        db = await _db()
        n = NotificationDispatcher()
        n.bind(db)
        repo = f"R{uuid.uuid4().hex[:6]}"
        for i in range(25):
            n.notify("Burst", f"row {i}", repo_id=repo)
        await n.close()                                                        # no sleep: close() must drain the queue
        assert len([e for e in await db.get_telemetry_events(repo, 100) if e["title"] == "Burst"]) == 25
    run(go())
