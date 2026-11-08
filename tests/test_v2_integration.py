"""v2 integration glue that needs none of langchain / fastapi / aiosqlite / pydantic_settings."""
import ast
import asyncio
from pathlib import Path

import pytest

from services import action_log as al
from services.agentic_search import AgenticSearch, RepoCorpus, needs_vectors
from services.reasoning import (BaselinePolicy, IntakeAgent, IntakeBrief, ReasoningSupervisor, VerifierAgent)
from services.vectorize_jobs import InMemoryJobStore, JobState, VectorizeJobManager

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean():
    al.clear_sinks()
    al.ring.clear()
    yield
    al.clear_sinks()
    al.ring.clear()


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ supervisor never loses the answer
class BoomIntake:
    def analyze(self, *a, **k):
        raise RuntimeError("intake exploded")


class BoomVerifier:
    def check(self, *a, **k):
        raise ValueError("verifier exploded")


async def _execute(brief, policy):
    return "THE ANSWER", {"tests_ran": False}


def _sup(intake=None, verifier=None):
    return ReasoningSupervisor(intake or IntakeAgent(), verifier or VerifierAgent(), lambda: {}, enabled=True)


def test_intake_failure_falls_back_to_plain_answer_and_logs_a_fail():
    out = run(_sup(intake=BoomIntake()).run(query="explain login", intent="ask", repo_files=[], execute=_execute))
    assert out == "THE ANSWER"
    fails = [e for e in al.recent() if e["status"] == "fail"]
    assert any("intake exploded" in e["detail"] for e in fails)


def test_verifier_failure_delivers_the_answer_without_notes():
    out = run(_sup(verifier=BoomVerifier()).run(query="explain login in auth.py", intent="ask", repo_files=[],
                                                execute=_execute))
    assert out == "THE ANSWER" and "Delivery notes" not in out
    assert any(e["status"] == "fail" and "verifier exploded" in e["detail"] for e in al.recent())


def test_healthy_supervisor_still_appends_notes():
    out = run(_sup().run(query="explain how login works in auth.py", intent="ask", repo_files=[], execute=_execute))
    assert out.startswith("THE ANSWER") and "Delivery notes" in out


def test_execute_errors_still_propagate():
    async def bad(brief, policy):
        raise KeyError("model down")
    with pytest.raises(KeyError):
        run(_sup().run(query="explain x in a.py", intent="ask", repo_files=[], execute=bad))


# ------------------------------------------------------------------ confidence threshold
FILES = {
    "src/auth/login_service.py": "class LoginService:\n    def authenticate_user(self, name, password):\n        return True\n",
    "src/api/routes.py": "def login_route(req):\n    return 1\n",
}


def test_conceptual_query_with_similarity_word_lands_strictly_below_default_threshold():
    res = AgenticSearch(RepoCorpus.from_mapping(FILES)).retrieve("places that are similar to how users get logged in everywhere")
    assert res.blocks, "the fixture must produce hits, otherwise the test proves nothing"
    assert res.confidence < 0.45, res
    need, why = needs_vectors(res)          # DEFAULT threshold
    assert need and why


def test_specific_identifier_query_does_not_need_vectors():
    res = AgenticSearch(RepoCorpus.from_mapping(FILES)).retrieve("explain LoginService.authenticate_user in login_service.py")
    assert res.confidence >= 0.45 and not needs_vectors(res)[0]


# ------------------------------------------------------------------ agentic retrieve -> needs_vectors -> job -> notify -> done
def test_weak_agentic_result_enqueues_job_notifies_and_completes():
    async def main():
        notes, statuses, worked = [], [], []

        def notify(title, message, level="INFO", repo_id=None, **kw):
            notes.append({"title": title, "message": message, "level": level, "repo_id": repo_id, **kw})

        async def setter(repo, status):
            statuses.append(status)

        async def work(repo, cb):               # stands in for EngineController.vectorize_repository
            worked.append(repo)
            cb(0, 2)
            await asyncio.sleep(0)
            cb(2, 2)

        store = InMemoryJobStore()
        mgr = VectorizeJobManager(store, work, notify, setter)
        res = AgenticSearch(RepoCorpus.from_mapping(FILES)).retrieve("code similar in meaning to password checking everywhere")
        need, why = needs_vectors(res)
        assert need
        job = await mgr.enqueue("repoA", why)           # what rag_graph does when vectorize_fallback == "auto"
        again = await mgr.enqueue("repoA", why)
        assert again.job_id == job.job_id               # deduplicated while active
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.COMPLETED and worked == ["repoA"]
        assert statuses == ["queued", "running", "ready"] and mgr.status("repoA") == "ready"
        assert notes[0]["level"] == "INFO" and why in notes[0]["message"] and notes[0]["kind"] == "job"
        assert notes[-1]["level"] == "SUCCESS" and "semantic search enabled" in notes[-1]["message"]
        assert notes[-1]["data"]["state"] == "COMPLETED"
        # the job shows up as VECTORIZE actions too
        assert any(e["phase"] == "VECTORIZE" for e in al.recent())
    run(main())


def test_failing_work_marks_job_failed_and_status_failed():
    async def main():
        statuses, notes = [], []

        async def setter(repo, status):
            statuses.append(status)

        async def work(repo, cb):
            raise RuntimeError("embedding model offline")
        mgr = VectorizeJobManager(InMemoryJobStore(), work, lambda *a, **k: notes.append((a, k)), setter)
        job = await mgr.enqueue("r", "why")
        done = await mgr.wait(job.job_id, 5)
        assert done.state == JobState.FAILED and "embedding model offline" in done.error
        assert statuses[-1] == "failed" and notes[-1][1]["level"] == "ERROR"
    run(main())


# ------------------------------------------------------------------ action sink
def test_notifier_sink_formats_action_events():
    calls = []
    al.add_sink(al.make_notifier_sink(lambda *a, **k: calls.append((a, k))))
    with al.request_scope("req1", "th1", repo_id="repoX"):
        al.log_action("RETRIEVE", "AgenticSearch", "grep", "ok", "3 hits", data={"n": 3})
        al.log_action("SANDBOX", "Sandbox", "run tests", "fail", "2 failed")
    (a1, k1), (a2, k2) = calls
    assert a1 == ("Action", "3 hits") and k1["level"] == "INFO" and k1["kind"] == "action"
    assert k1["repo_id"] == "repoX" and k1["request_id"] == "req1" and k1["thread_id"] == "th1"
    assert k1["data"]["phase"] == "RETRIEVE" and k1["data"]["data"] == {"n": 3}
    assert k2["level"] == "ERROR" and a2[1] == "2 failed"


def test_sink_without_detail_uses_agent_and_action():
    calls = []
    sink = al.make_notifier_sink(lambda *a, **k: calls.append(a))
    sink(al.log_action("LLM", "AskAgent", "llm_call"))
    assert calls[-1] == ("Action", "AskAgent: llm_call")


def test_events_from_telemetry_rows_keeps_only_action_rows():
    ev = al.log_action("INTAKE", "IntakeAgent", "analyze", repo_id="r1").to_dict()
    rows = [{"kind": "log", "data": None}, {"kind": "action", "data": ev}, {"kind": "job", "data": {"job_id": "x"}},
            {"kind": "action", "data": None}]
    assert al.events_from_telemetry_rows(rows) == [ev]
    assert al.events_from_telemetry_rows(rows, repo_id="other") == []


def test_format_trace_is_compact_and_skips_start_events():
    out = al.format_trace([
        {"status": "start", "phase": "SANDBOX", "agent": "S", "action": "run"},
        {"status": "ok", "phase": "SANDBOX", "agent": "S", "action": "run", "duration_ms": 12, "model": "m", "detail": "3 passed"},
        {"status": "fail", "phase": "LLM", "agent": "A", "action": "call"}])
    lines = out.splitlines()
    assert len(lines) == 2 and "12ms" in lines[0] and "[m]" in lines[0] and "3 passed" in lines[0] and "FAIL" in lines[1]


# ------------------------------------------------------------------ static wiring checks (controller / settings import heavy deps)
def _class_methods(path: str, cls: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_settings_declare_the_v2_flags_with_the_designed_defaults():
    tree = ast.parse((ROOT / "config/settings.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineSettings")
    defaults = {}
    for n in cls.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            v = n.value
            if isinstance(v, ast.Call) and v.keywords:
                v = next((k.value for k in v.keywords if k.arg == "default"), None)
            if isinstance(v, ast.Constant):
                defaults[n.target.id] = v.value
    assert defaults["ingest_mode"] == "agentic" and defaults["retrieval_mode"] == "agentic"
    assert defaults["vectorize_fallback"] == "auto" and defaults["vectorize_confidence_threshold"] == 0.45
    assert defaults["reasoning_supervisor"] is True and defaults["reasoning_intake_llm"] is False
    assert defaults["action_log_enabled"] is True


def test_controller_exposes_the_v2_entry_points():
    m = _class_methods("controllers/engine_controller.py", "EngineController")
    assert {"ingest_repository", "vectorize_repository", "reconcile_jobs", "start_vectorize", "search_mode",
            "list_repos_enriched", "incremental_reingest_file", "_chat_impl", "_route_chat", "_decide"} <= m


def test_api_declares_the_v2_routes():
    src = (ROOT / "views/api_view.py").read_text(encoding="utf-8")
    for route in ('"/api/v1/repos/{repo_id}/vectorize"', '"/api/v1/jobs"', '"/api/v1/jobs/{job_id}"', '"/api/v1/actions"',
                  '"/api/v1/repos/{repo_id}/search-mode"'):
        assert route in src, route
    assert src.count('"request_id": request_id') >= 3        # chat, chat/attach (+ cancelled variants), changes
