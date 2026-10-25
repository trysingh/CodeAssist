"""Shared state on a class that existing tests build with `Cls.__new__(Cls)` (no __init__): readers must not assume it is there.
Run:  pytest tests/test_shared_lazy.py -q   (scripted models, real pytest via LocalRunner, no Docker / Ollama / network)
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
SVC = b'class Svc:\n    def __init__(self):\n        self.name = "svc"\n\n    def hello(self):\n        return "hi"\n'
TEST = (b"from svc import Svc\n\n\ndef test_hello_without_init():\n    s = Svc.__new__(Svc)        # the way an existing test builds it\n"
        b"    assert s.hello() == \"hi\"\n")


def shared(lazy=True, default="{}"):
    x = ctr.Shared("Svc._jobs", "svc.py", "Dict[str, int]", "jobs by key", "Svc.__init__", ["svc.py::Svc.hello"], default)
    x.lazy = lazy
    return [x]


def task(name, creates=()):
    return tasks.Task("t", "implement", "svc.py", name, "def x(self)", "does it", creates=list(creates))


def src(body):
    return ("class Svc:\n    def __init__(self):\n        self.name = 'svc'\n\n    " + body.replace("\n", "\n    ") + "\n").encode()


# ------------------------------------------------------------------ detect / render / gate
def test_classes_built_without_init_are_found_in_tests_only():
    found = ctr.init_bypassed({"tests/test_a.py": b"a = Box.__new__(Box)\nb = object.__new__(Crate)\nc = Other.__new__(Box)\n",
                               "app.py": b"x = Hidden.__new__(Hidden)\n"})
    assert found == {"Box", "Crate"}


def test_the_hint_names_the_safe_way_to_read_it():
    with_default = ctr.render_shared(shared())
    no_default = ctr.render_shared(shared(default=""))
    assert "Svc.__new__" in with_default and "self.__dict__.setdefault('_jobs', {})" in with_default
    assert "getattr(self, '_jobs', None)" in no_default
    assert "__new__" not in ctr.render_shared(shared(lazy=False))


def test_a_default_that_is_not_an_expression_is_ignored():
    item = {"attr": "Svc._jobs", "file": "svc.py", "type": "dict", "purpose": "x", "created_in": "Svc.__init__", "default": "def ("}
    syms = [ctr.CSym("svc.py", "Svc.__init__", "def __init__(self) -> None", "method", "x", [], [], "behavior")]
    out, probs = ctr.parse_shared({"shared": [item]}, syms, {"svc.py": SVC})
    assert probs == [] and out[0].default == ""


@pytest.mark.parametrize("body, flagged", [
    ('def hello(self):\n    self._jobs["a"] = 1\n    return "hi"', True),                                  # direct write
    ('def hello(self):\n    return len(self._jobs)', True),                                              # direct read
    ('def hello(self):\n    self.__dict__.setdefault("_jobs", {})["a"] = 1\n    return "hi"', False),
    ('def hello(self):\n    return len(getattr(self, "_jobs", None) or {})', False),
    ('def hello(self):\n    self._jobs = {}\n    return len(self._jobs)', False),                         # it sets it itself first
    ('def hello(self):\n    return "hi"', False),
])
def test_other_methods_must_read_lazy_state_defensively(body, flagged):
    probs = ctr.shared_problems(task("Svc.hello"), src(body), shared())
    assert bool(probs) is flagged and (not flagged or "without __init__" in probs[0])


def test_the_creator_may_assign_it_plainly_and_a_normal_class_is_not_nagged():
    init = b"class Svc:\n    def __init__(self):\n        self.name = 'svc'\n        self._jobs = {}\n"
    assert ctr.shared_problems(task("Svc.__init__", creates=["_jobs"]), init, shared()) == []
    assert ctr.shared_problems(task("Svc.hello"), src('def hello(self):\n    return len(self._jobs)'), shared(lazy=False)) == []


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
        {"file": "svc.py", "name": "Svc.hello", "signature": "def hello(self)", "behavior": "Also record a job named hello."},
        {"file": "svc.py", "name": "Svc.__init__", "signature": "def __init__(self) -> None", "behavior": "Keep the name and an empty job map."},
    ],
    "shared": [{"attr": "Svc._jobs", "file": "svc.py", "type": "Dict[str, int]", "purpose": "jobs by key", "created_in": "Svc.__init__",
                "used_by": ["svc.py::Svc.hello"], "default": "{}"}],
}
INIT = 'def __init__(self) -> None:\n    self.name = "svc"\n    self._jobs = {}'
HELLO_DIRECT = 'def hello(self):\n    self._jobs["hello"] = 1\n    return "hi"'
HELLO_SAFE = 'def hello(self):\n    self.__dict__.setdefault("_jobs", {})["hello"] = 1\n    return "hi"'


@pytest.fixture(autouse=True)
def loop_only(monkeypatch):
    for k, v in dict(change_contract=True, change_tasks=True, change_task_tests=False, change_author_tests=False,
                     change_tests_first=False).items():
        monkeypatch.setattr(settings, k, v)


def go(llm):
    async def inner():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, {"svc.py": SVC, "tests/test_svc.py": TEST}.items())
        emit = lambda *a: None
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
        svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=Script(DESIGN), emit=emit, sandbox=sb, choice_fn=_none)
        return await svc.run(r, "make Svc.hello in svc.py record a job"), s, r
    return run(inner())


def test_a_direct_read_is_sent_back_and_the_existing_new_based_test_stays_green():
    llm = Script(INIT, HELLO_DIRECT, HELLO_SAFE)
    res, s, r = go(llm)
    assert res.ok and len(llm.prompts) == 3
    assert "Svc.__new__" in llm.prompts[1] and "self.__dict__.setdefault('_jobs', {})" in llm.prompts[1]      # told up front
    assert "YOUR PREVIOUS REPLY FAILED" in llm.prompts[2] and "without __init__" in llm.prompts[2]             # and again when it slipped
    final = run(s.get_file(r, "svc.py", res.revision)).data.decode()
    assert 'setdefault("_jobs", {})' in final and "self._jobs = {}" in final
