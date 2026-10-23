"""/change guards: plan coverage, budget priority, chat context, honest verification label.
Run from the repo root:  pytest tests/test_change_guards.py -q   (no Ollama, Chroma, Docker or network needed)
"""
import asyncio
import uuid

from config.settings import settings
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class FakeDoc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class FakeVector:
    def __init__(self, paths=()):
        self.paths, self.queries = list(paths), []

    def similarity_search(self, repo_id, q, k=4):
        self.queries.append(q)
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


async def _service(llm, files, vector=None):
    s, rid = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(rid, files)
    svc = RepoChangeService(None, vector or FakeVector(), s, llm_text=llm, llm_json=_no_json, emit=lambda *a: None)
    return svc, s, rid


TWO = [("a.py", b"A = 1\n"), ("b.py", b"B = 1\n")]
PLAN_AB = "FEASIBLE: yes\nFILES:\n- a.py - change A\n- b.py - change B\nNEW FILES:\n- none\nRISKS:\n- none"


def test_plan_files_not_edited_trigger_a_retry_until_covered(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)

    async def go():
        llm = ScriptedLLM(PLAN_AB, block("a.py", "A = 1", "A = 2"), block("b.py", "B = 1", "B = 2"))
        svc, s, rid = await _service(llm, TWO)
        r = await svc.run(rid, "change a.py and b.py")
        assert r.ok and r.attempts == 2 and {f["path"] for f in r.files} == {"a.py", "b.py"}
        assert "have no edits yet: b.py" in llm.prompts[2]                 # the retry says exactly what is missing
    run(go())


def test_still_incomplete_on_the_last_attempt_is_flagged_not_hidden(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)
    monkeypatch.setattr(settings, "change_max_attempts", 1)

    async def go():
        llm = ScriptedLLM(PLAN_AB, block("a.py", "A = 1", "A = 2"))
        svc, s, rid = await _service(llm, TWO)
        r = await svc.run(rid, "change a.py and b.py")
        assert r.ok and r.notes[0].startswith("INCOMPLETE against the plan") and "b.py" in r.notes[0]
        assert "INCOMPLETE" in r.to_markdown()
    run(go())


def test_a_planned_file_beats_a_big_test_file_for_the_context_budget(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)
    monkeypatch.setattr(settings, "change_context_chars", 60)

    async def go():
        files = [("tests/test_a.py", b"t = 1\n" * 8), ("zsrc.py", b"x = 1\n" * 5)]   # sorts BEFORE the source file
        plan = "FEASIBLE: yes\nFILES:\n- zsrc.py - change x\nNEW FILES:\n- none\nRISKS:\n- none"
        llm = ScriptedLLM(plan, block("zsrc.py", "x = 1\nx = 1\nx = 1\nx = 1\nx = 1", "x = 2"))
        svc, s, rid = await _service(llm, files)
        r = await svc.run(rid, "change zsrc.py and test_a.py")
        shown = llm.prompts[1].split("FILES:")[-1]
        assert r.ok and "=== zsrc.py ===" in shown and "=== tests/test_a.py ===" not in shown
        assert any("skipped" in n and "tests/test_a.py" in n for n in r.notes)
    run(go())


def test_planned_file_that_cannot_fit_fails_up_front_with_a_clear_reason(monkeypatch):
    monkeypatch.setattr(settings, "change_plan_first", True)
    monkeypatch.setattr(settings, "change_context_chars", 50)

    async def go():
        plan = "FEASIBLE: yes\nFILES:\n- big.py - change\nNEW FILES:\n- none\nRISKS:\n- none"
        llm = ScriptedLLM(plan)
        svc, s, rid = await _service(llm, [("big.py", b"x = 1\n" * 40), ("small.py", b"y = 1\n")])
        r = await svc.run(rid, "change big.py and small.py")
        assert not r.ok and "context budget" in r.errors[0] and "big.py" in r.errors[0] and "Split" in r.errors[0]
        assert len(llm.prompts) == 1                                       # only the planner ran: no wasted patch calls
    run(go())


def test_chat_context_lets_a_vague_request_find_its_files_and_reaches_the_prompts(monkeypatch):
    async def go():
        vec = FakeVector()
        llm = ScriptedLLM(block("a.py", "A = 1", "A = 2"))
        svc, s, rid = await _service(llm, TWO, vec)
        ctx = "USER: telemetry lives in a.py\nASSISTANT: agreed, persist it there"
        r = await svc.run(rid, "in the light of the discussion so far build this feature", context=ctx)
        assert r.ok and [f["path"] for f in r.files] == ["a.py"]
        assert "CONVERSATION SO FAR" in llm.prompts[0] and "telemetry lives in a.py" in llm.prompts[0]
        assert "telemetry lives in a.py" in vec.queries[0]                  # semantic search sees the conversation too

        bare = await _service(ScriptedLLM(), TWO)
        r2 = await bare[0].run(bare[2], "in the light of the discussion so far build this feature")
        assert not r2.ok                                                    # same words, no context: nothing to go on
    run(go())


def test_zero_passed_tests_do_not_earn_the_l2_label():
    v = RepoChangeService._verification
    assert v({"ran": True, "ok": True, "mode": "pytest", "passed": 0}) == "L0-static"
    assert v({"ran": True, "ok": True, "mode": "pytest", "passed": 2}) == "L2-tests"
    assert v({"ran": True, "ok": True, "mode": "smoke", "passed": 3}) == "L1-import-smoke"
