"""A rewrite keeps the calls it had; a test may not hand asyncio.run a function.
Run:  pytest tests/test_preserve_and_lint.py -q   (scripted models, real pytest via LocalRunner, no Docker / Ollama / network)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import patch_targets as pt
from services import preserve
from services import sandbox_service as sbx
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run

# ------------------------------------------------------------------ preserve
OLD = b'''
from scopes import repo_scope
import asyncio


class C:
    async def chat(self, repo_id, query, web="auto"):
        repo = await self.db.get_repo(repo_id)
        if not repo:
            raise ValueError("x")
        note = query.strip()
        with repo_scope(repo_id):
            return note + await self._route(repo_id, query, web)

    async def _route(self, repo_id, query, web):
        return "routed"
'''
BYPASS = OLD.replace(b'''        with repo_scope(repo_id):
            return note + await self._route(repo_id, query, web)
''', b'''        task = asyncio.create_task(self._run(repo_id, query, web))
        return note + await task

    async def _run(self, repo_id, query, web):
        with repo_scope(repo_id):
            return await self.rag.invoke(repo_id, query, web)
''')
WRAPPED = BYPASS.replace(b"self.rag.invoke(repo_id, query, web)", b"self._route(repo_id, query, web)")


def test_replacing_the_router_with_a_direct_call_is_reported():
    out = preserve.dropped_calls(OLD, BYPASS, "C.chat")
    assert len(out) == 1 and "`self._route(...)`" in out[0]


def test_wrapping_the_call_in_a_new_helper_is_fine():
    assert preserve.dropped_calls(OLD, WRAPPED, "C.chat") == []


def test_the_task_may_say_that_the_call_goes():
    assert preserve.dropped_calls(OLD, BYPASS, "C.chat", "Run it as a task and replace _route with the rag call") == []


def test_nothing_to_compare_means_nothing_reported():
    assert preserve.dropped_calls(OLD, OLD, "C.chat") == [] and preserve.dropped_calls(OLD, BYPASS, "C.brand_new") == []
    assert preserve.dropped_calls(OLD, b"def broken(:", "C.chat") == []


# ------------------------------------------------------------------ lint
HEAD = "import asyncio\nfrom calc import mul\n\n\n"


@pytest.mark.parametrize("src", [
    HEAD + "def run(coro):\n    return asyncio.run(coro)\n\n\ndef test_a():\n    async def go():\n        assert mul(2, 3) == 6\n    run(go)\n",
    HEAD + "run = asyncio.run\n\n\nasync def go():\n    assert mul(2, 3) == 6\n\n\ndef test_a():\n    run(go)\n",
    HEAD + "async def go():\n    pass\n\n\ndef test_a():\n    asyncio.run(go)\n",
    "from asyncio import run\n\n\nasync def go():\n    pass\n\n\ndef test_a():\n    run(go)\n",
])
def test_a_function_handed_to_asyncio_run_is_reported(src):
    out = pt.coroutine_misuse(src)
    assert len(out) == 1 and "passes the function itself" in out[0] and "go()" in out[0]


@pytest.mark.parametrize("src", [
    HEAD + "def run(coro):\n    return asyncio.run(coro)\n\n\ndef test_a():\n    async def go():\n        pass\n    run(go())\n",
    HEAD + "def run(fn):\n    return fn()\n\n\ndef test_a():\n    def go():\n        pass\n    run(go)\n",        # not an asyncio runner
    HEAD + "def test_a():\n    asyncio.run(asyncio.sleep(0))\n",
    "def broken(:\n",
])
def test_a_proper_call_or_a_different_runner_passes(src):
    assert pt.coroutine_misuse(src) == []


def test_the_runtime_error_is_a_test_side_error_with_or_without_a_contract():
    rep = sbx.RunReport(mode="pytest", ran=True, failures=[sbx.CaseResult(
        "tests.t::a", "failed", "ValueError: a coroutine was expected, got <function go at 0x7f>")])
    v = RepoChangeService._red_verdict
    assert v(rep, {"calc"}, strict=True)[0] == "bad" and v(rep, {"calc"}, strict=False)[0] == "bad"


# ------------------------------------------------------------------ in the loop
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


def service(files, llm, js=_none):
    async def inner():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, files.items())
        emit = lambda *a: None
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
        return RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=js, emit=emit, sandbox=sb, choice_fn=_none), s, r
    return inner


def tfile(body):
    return f"FILE: tests/test_change_r1_calc.py\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"


T_COROUTINE = tfile("import asyncio\nfrom calc import mul\n\n\ndef run(coro):\n    return asyncio.run(coro)\n\n\n"
                    "def test_mul():\n    async def go():\n        assert mul(2, 3) == 6\n    run(go)")
T_OK = tfile("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6")
CODE = ("FILE: calc.py\n<<<<<<< SEARCH\ndef add(a, b):\n    return a + b\n=======\n"
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n>>>>>>> REPLACE\n")


def test_a_test_that_hands_asyncio_run_a_function_is_rewritten_before_any_container_runs(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", True)
    monkeypatch.setattr(settings, "change_tests_first", True)
    llm = Script(T_COROUTINE, T_OK, CODE)

    async def go():
        svc, s, r = await service({"calc.py": b"def add(a, b):\n    return a + b\n"}, llm)()
        return await svc.run(r, "add a mul function to calc.py")
    res = run(go())
    assert res.ok and len(llm.prompts) == 3 and "`run(go)` passes the function itself" in llm.prompts[1]


SVC = b'class Svc:\n    def helper(self):\n        return "h"\n\n    def hello(self):\n        return self.helper()\n'
DESIGN = {"symbols": [
    {"file": "svc.py", "name": "Svc.hello", "signature": "def hello(self)", "behavior": "Count the call with self.bump(), then return what it returned before.",
     "depends_on": ["svc.py::Svc.bump"]},
    {"file": "svc.py", "name": "Svc.bump", "signature": "def bump(self) -> None", "behavior": "Add one to a call counter."},
]}
BUMP = 'def bump(self) -> None:\n    self.calls = getattr(self, "calls", 0) + 1'
HELLO_DROPS = 'def hello(self):\n    self.bump()\n    return "h"'
HELLO_KEEPS = 'def hello(self):\n    self.bump()\n    return self.helper()'


def test_a_rewrite_that_drops_an_existing_call_is_sent_back(monkeypatch):
    for k, v in dict(change_contract=True, change_tasks=True, change_task_tests=False, change_author_tests=False,
                     change_tests_first=False).items():
        monkeypatch.setattr(settings, k, v)
    llm = Script(BUMP, HELLO_DROPS, HELLO_KEEPS)

    async def go():
        svc, s, r = await service({"svc.py": SVC}, llm, Script(DESIGN))()
        return await svc.run(r, "make Svc.hello in svc.py count its calls"), s, r
    res, s, r = run(go())
    assert res.ok and len(llm.prompts) == 3
    assert "YOUR PREVIOUS REPLY FAILED" in llm.prompts[2] and "the old code called `self.helper(...)`" in llm.prompts[2]
    assert "return self.helper()" in run(s.get_file(r, "svc.py", res.revision)).data.decode()
