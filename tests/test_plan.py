"""Plan-first tests: /plan, the plan step inside /change, context-budget reporting, router prefix.
Run from the repo root:  pytest tests/test_plan.py -q   (no Ollama, Chroma or network needed)
"""
import asyncio
import uuid

from config.settings import settings
from services.intent_router import COMMANDS_HELP, route_intent
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
PLAN = "FEASIBLE: yes, small edit\nFILES:\n- old.py - change Y\nNEW FILES:\n- none\nRISKS:\n- none"


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class FakeDoc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class FakeVector:
    def __init__(self, paths=()):
        self.paths = list(paths)

    def similarity_search(self, repo_id, q, k=4):
        return [FakeDoc(p) for p in self.paths]


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


async def _no_json(*a, **k):
    return None


async def _service(llm, files):
    s, rid = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(rid, files)
    svc = RepoChangeService(None, FakeVector(), s, llm_text=llm, llm_json=_no_json, emit=lambda *a: None)
    return svc, s, rid


OLD = [("old.py", b"Y = 1\n")]


def test_plan_runs_before_patch_and_is_shown(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)

    async def go():
        llm = ScriptedLLM(PLAN, block("old.py", "Y = 1", "Y = 2"))
        svc, s, rid = await _service(llm, OLD)
        r = await svc.run(rid, "change Y in old.py")
        assert r.ok and r.plan == PLAN
        assert "Do NOT write code" in llm.prompts[0]                      # first call = planner
        assert "PLAN (follow it" in llm.prompts[1] and "FEASIBLE: yes" in llm.prompts[1]
        assert "**Plan:**" in r.to_markdown()
    run(go())


def test_plan_switch_off_means_one_model_call():
    async def go():
        llm = ScriptedLLM(block("old.py", "Y = 1", "Y = 2"))
        svc, s, rid = await _service(llm, OLD)
        r = await svc.run(rid, "change Y in old.py")
        assert r.ok and r.plan == "" and len(llm.prompts) == 1
    run(go())


def test_plan_model_failure_does_not_block_the_change(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)

    async def go():
        llm = ScriptedLLM(RuntimeError("boom"), block("old.py", "Y = 1", "Y = 2"))
        svc, s, rid = await _service(llm, OLD)
        r = await svc.run(rid, "change Y in old.py")
        assert r.ok and r.plan == ""
    run(go())


def test_plan_only_saves_nothing_even_with_switch_off():
    async def go():
        llm = ScriptedLLM(PLAN)
        svc, s, rid = await _service(llm, OLD)
        out = await svc.plan_only(rid, "change Y in old.py")
        assert "Plan" in out and "old.py" in out and "/change change Y in old.py" in out
        assert await s.latest_rev(rid) == 0 and len(llm.prompts) == 1
    run(go())


def test_plan_only_reports_when_nothing_is_stored():
    async def go():
        svc, s, rid = await _service(ScriptedLLM(), [])
        out = await svc.plan_only(rid, "change Y in old.py")
        assert out.startswith("**Could not plan.**")
    run(go())


def test_no_edit_blocks_error_quotes_the_model_reply():
    async def go():
        said = "I think you should add a table for this."
        svc, s, rid = await _service(ScriptedLLM(said, said, said), OLD)
        r = await svc.run(rid, "change Y in old.py")
        assert not r.ok
        assert any("Your reply started with" in e and "add a table" in e for e in r.errors)
    run(go())


def test_oversized_file_is_reported_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(settings, "change_context_chars", 100)

    async def go():
        files = [("big.py", b"x = 1\n" * 40), ("small.py", b"y = 1\n")]
        llm = ScriptedLLM(block("small.py", "y = 1", "y = 2"))
        svc, s, rid = await _service(llm, files)
        r = await svc.run(rid, "change big.py and small.py")
        assert r.ok and any("skipped" in n and "big.py" in n for n in r.notes)
        assert "big.py" not in llm.prompts[0].split("FILES:")[-1]
    run(go())


def test_nothing_fits_budget_fails_with_clear_reason(monkeypatch):
    monkeypatch.setattr(settings, "change_context_chars", 10)

    async def go():
        llm = ScriptedLLM()
        svc, s, rid = await _service(llm, [("big.py", b"x = 1\n" * 40)])
        r = await svc.run(rid, "change big.py")
        assert not r.ok and "context budget" in r.errors[0] and not llm.prompts
    run(go())


def test_plan_prefix_routes_to_plan_and_is_listed_in_help():
    async def go():
        d = await route_intent("/plan add a table", model="")
        assert (d.intent, d.query, d.source) == ("plan", "add a table", "prefix")
        d2 = await route_intent("/change add a table", model="")
        assert d2.intent == "change"
    run(go())
    assert "/plan" in COMMANDS_HELP
