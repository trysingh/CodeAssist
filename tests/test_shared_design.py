"""Shared state in the contract: one owner, one name, every task sees it.
Run:  pytest tests/test_shared_design.py -q   (scripted models, real pytest via LocalRunner, no Docker / Ollama / network)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import contract as ctr
from services import sandbox_service as sbx
from services import tasks
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
SVC = b'class Svc:\n    def __init__(self):\n        self.name = "svc"\n'
SRC = {"svc.py": SVC}
KNOWN = ["svc.py", "other.py"]


def csym(name, file="svc.py", kind="method", deps=()):
    return ctr.CSym(file, name, f"def {name.split('.')[-1]}(self)", kind, "does it", list(deps), [], "new")


def item(**kw):
    base = {"attr": "Svc._jobs", "file": "svc.py", "type": "Dict[str, int]", "purpose": "jobs by key",
            "created_in": "Svc.__init__", "used_by": ["svc.py::Svc.add_job"]}
    return {**base, **kw}


SYMS = [csym("Svc.add_job"), csym("Svc.__init__")]


def parsed(*items, syms=SYMS):
    return ctr.parse_shared({"shared": list(items)}, syms, SRC)


# ------------------------------------------------------------------ parse / link / render
def test_a_valid_entry_is_kept_and_users_depend_on_the_creator():
    shared, probs = parsed(item())
    assert probs == [] and shared[0].attr == "Svc._jobs" and shared[0].used_by == ["svc.py::Svc.add_job"]
    ctr.link_shared(SYMS, shared)
    assert SYMS[0].deps == ["svc.py::Svc.__init__"] and SYMS[1].deps == []
    ordered, _ = tasks.plan_tasks(SYMS, [], SRC, shared)           # the designer listed add_job first
    assert [t.name for t in ordered] == ["Svc.__init__", "Svc.add_job"] and ordered[0].creates == ["_jobs"]


def test_no_shared_key_means_nothing_changes():
    assert ctr.parse_shared({"symbols": []}, SYMS, SRC) == ([], [])
    assert ctr.parse_shared({"shared": []}, SYMS, SRC) == ([], [])


@pytest.mark.parametrize("bad, text", [
    (item(attr="_jobs"), "Class.attribute"),                                        # a module-level global
    (item(file="services/cancellation.py"), "Do not create a module"),             # an invented module
    (item(attr="Ghost._jobs"), "class Ghost is not in svc.py"),
    (item(created_in="Svc.setup"), "created_in 'Svc.setup'"),                       # the creator is not in the contract
    (item(created_in="other"), "created_in 'other'"),
])
def test_entries_that_cannot_be_built_are_reported_and_dropped(bad, text):
    shared, probs = parsed(bad)
    assert shared == [] and len(probs) == 1 and text in probs[0]


def test_a_new_class_may_own_state_and_unknown_users_are_dropped_quietly():
    syms = [csym("Reg", kind="class"), csym("Reg.__init__"), csym("Reg.add")]
    shared, probs = ctr.parse_shared({"shared": [item(attr="Reg._items", created_in="Reg.__init__",
                                                       used_by=["svc.py::Reg.add", "svc.py::nope"])]}, syms, SRC)
    assert probs == [] and shared[0].used_by == ["svc.py::Reg.add"]


def test_render_marks_the_creator():
    shared, _ = parsed(item())
    mine = ctr.render_shared(shared, ("svc.py", "Svc.__init__"))
    other = ctr.render_shared(shared, ("svc.py", "Svc.add_job"))
    assert "self._jobs: Dict[str, int]" in mine and "YOU assign it" in mine
    assert "assigned in Svc.__init__" in other and "YOU" not in other
    assert ctr.render_shared([]) == ""


def test_the_creator_must_really_assign_the_attribute():
    t = tasks.Task("t1", "implement", "svc.py", "Svc.__init__", "def __init__(self)", "x", creates=["_jobs"])
    ok = b"class Svc:\n    def __init__(self):\n        self._jobs = {}\n"
    no = b"class Svc:\n    def __init__(self):\n        self.jobs = {}\n"
    assert ctr.shared_problems(t, ok) == [] and "self._jobs" in ctr.shared_problems(t, no)[0]
    assert ctr.shared_problems(tasks.Task("t2", "implement", "svc.py", "Svc.add_job", "def add_job(self)", "x"), no) == []


# ------------------------------------------------------------------ the whole loop, scripted
class Script:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class Vec:
    def similarity_search(self, repo_id, query, k=8):
        return []


async def _none(*a, **k):
    return None


DESIGN = {
    "symbols": [
        {"file": "svc.py", "name": "Svc.add_job", "signature": "def add_job(self, key: str) -> None", "behavior": "Record the job."},
        {"file": "svc.py", "name": "Svc.count", "signature": "def count(self) -> int", "behavior": "Number of jobs."},
        {"file": "svc.py", "name": "Svc.__init__", "signature": "def __init__(self) -> None", "behavior": "Keep the name and an empty job map."},
    ],
    "shared": [item(used_by=["svc.py::Svc.add_job", "svc.py::Svc.count"])],
}
INIT_OK = 'def __init__(self) -> None:\n    self.name = "svc"\n    self._jobs = {}'
INIT_FORGETS = 'def __init__(self) -> None:\n    self.name = "svc"'
ADD = "def add_job(self, key: str) -> None:\n    self._jobs[key] = 1"
COUNT = "def count(self) -> int:\n    return len(self._jobs)"


def go(llm):
    async def inner():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, SRC.items())
        emit = lambda *a: None
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
        svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=Script(DESIGN), emit=emit, sandbox=sb, choice_fn=_none)
        return await svc.run(r, "add job tracking to the Svc class in svc.py"), s, r
    return run(inner())


@pytest.fixture(autouse=True)
def loop_only(monkeypatch):
    for k, v in dict(change_contract=True, change_tasks=True, change_task_tests=False, change_author_tests=False,
                     change_tests_first=False).items():
        monkeypatch.setattr(settings, k, v)


def test_every_task_sees_the_same_name_and_the_creator_runs_first():
    llm = Script(INIT_OK, ADD, COUNT)
    res, s, r = go(llm)
    assert res.ok and len(llm.prompts) == 3
    assert "YOUR TASK: svc.py::Svc.__init__" in llm.prompts[0] and "YOU assign it in this function" in llm.prompts[0]
    for p in llm.prompts[1:]:
        assert "SHARED STATE" in p and "self._jobs: Dict[str, int]" in p and "assigned in Svc.__init__" in p
    final = run(s.get_file(r, "svc.py", res.revision)).data.decode()
    assert "self._jobs = {}" in final and "len(self._jobs)" in final and "contract:" not in final


def test_a_creator_that_forgets_the_attribute_is_sent_back():
    llm = Script(INIT_FORGETS, INIT_OK, ADD, COUNT)
    res, *_ = go(llm)
    assert res.ok and len(llm.prompts) == 4
    assert "YOUR PREVIOUS REPLY FAILED" in llm.prompts[1] and "must assign it" in llm.prompts[1]
