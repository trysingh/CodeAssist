"""Tests-first: tests are written before the code, must be red on the old code, and are read-only for the patcher.
Run from the repo root:  pytest tests/test_tests_first.py -q   (no Docker, Ollama or network: real pytest via LocalRunner, scripted model)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services import symbols
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
CALC = b"def add(a, b):\n    return a + b\n"
REQ = "add a mul function to calc.py"
TPATH = "tests/test_change_r1_calc.py"


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def tfile(body):
    return f"FILE: {TPATH}\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"


T_MUL = tfile("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n\n\ndef test_mul_zero():\n    assert mul(0, 5) == 0")
T_MUL_WRONG = T_MUL.replace("== 6", "== 7")
T_ADD_ONLY = tfile("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3")
CODE_MUL = block("calc.py", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b")


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


async def _svc(llm):
    s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(r, {"calc.py": CALC}.items())
    emit = lambda *a: None
    sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
    return RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=_none, emit=emit, sandbox=sb, choice_fn=_none), s, r


@pytest.fixture(autouse=True)
def tdd(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", True)
    monkeypatch.setattr(settings, "change_tests_first", True)


def test_tests_are_written_from_the_request_before_the_code_and_prove_the_change():
    async def go():
        llm = Script(T_MUL, CODE_MUL)
        svc, s, r = await _svc(llm)
        return llm, await svc.run(r, REQ)
    llm, res = run(go())
    assert res.ok and res.verification == "L3-fail-to-pass" and TPATH in {f["path"] for f in res.files}
    assert "BEFORE the code exists" in llm.prompts[0] and "def add(a, b)" in llm.prompts[0]   # signatures, not bodies
    assert "DIFF OF THE CHANGE" not in llm.prompts[0]
    assert "PRE-WRITTEN TESTS (read-only)" in llm.prompts[1] and "from calc import mul" in llm.prompts[1]


def test_tests_that_already_pass_are_rewritten_once():
    async def go():
        llm = Script(T_ADD_ONLY, T_MUL, CODE_MUL)
        svc, s, r = await _svc(llm)
        return llm, await svc.run(r, REQ)
    llm, res = run(go())
    assert res.ok and res.verification == "L3-fail-to-pass" and len(llm.prompts) == 3
    assert "already PASS" in llm.prompts[1] and "def test_add" in llm.prompts[1]


def test_tests_that_never_turn_red_are_dropped_and_the_old_author_step_takes_over():
    async def go():
        llm = Script(T_ADD_ONLY, T_ADD_ONLY, CODE_MUL, "NO_TESTS: nothing to add")
        svc, s, r = await _svc(llm)
        return await svc.run(r, REQ)
    res = run(go())
    assert res.ok and not any(f["path"].startswith("tests/") for f in res.files)
    assert any("never became red" in n for n in res.notes)


def test_the_patcher_cannot_quietly_edit_the_prewritten_tests():
    async def go():
        llm = Script(T_MUL, block(TPATH, "assert mul(2, 3) == 6", "assert mul(2, 3) == 7"), CODE_MUL)
        svc, s, r = await _svc(llm)
        res = await svc.run(r, REQ)
        return llm, res, await s.get_file(r, TPATH, res.revision)
    llm, res, f = run(go())
    assert res.ok and res.attempts == 2 and "Do not edit the pre-written test" in llm.prompts[2]
    assert b"== 6" in f.data and b"== 7" not in f.data


def test_a_wrong_test_can_be_fixed_when_the_patcher_says_so():
    async def go():
        fix = ("TEST_WRONG: " + TPATH + " - 2*3 is 6, not 7\n"
               + block(TPATH, "assert mul(2, 3) == 7", "assert mul(2, 3) == 6") + CODE_MUL)
        svc, s, r = await _svc(Script(T_MUL_WRONG, fix))
        return await svc.run(r, REQ)
    res = run(go())
    assert res.ok and any("revised by the implementer" in n for n in res.notes)


def test_a_lone_test_file_is_never_saved_as_the_change():
    async def go():
        svc, s, r = await _svc(Script(T_MUL, "NO_CHANGE: calc.py - mul already exists"))
        return await svc.run(r, REQ)
    res = run(go())
    assert not res.ok and res.revision is None and res.errors[0].startswith("Nothing to change")


def test_red_verdicts():
    tops = {"calc"}
    rep = lambda *fails, ran=True, passed=0: sbx.RunReport(mode="pytest", ran=ran, passed=passed, failures=list(fails))
    case = lambda msg: sbx.CaseResult("tests.t::a", "failed", msg)
    v = RepoChangeService._red_verdict
    assert v(rep(case("ImportError: cannot import name 'mul'")), tops)[0] == "red"
    assert v(rep(passed=3), tops)[0] == "vacuous"
    assert v(rep(case("NameError: name 'x' is not defined")), tops)[0] == "bad"
    assert v(rep(case("ModuleNotFoundError: No module named 'numpy'")), tops)[0] == "broken"
    assert v(rep(ran=False), tops)[0] == "unknown"


def test_symbols_give_signatures_and_exact_ranges():
    src = ("import x\n\n@dec\nasync def f(a: int, *, b: str = 'x') -> bool:\n    \"\"\"Do f.\"\"\"\n    return True\n\n\n"
           "class C(Base):\n    def m(self, n):\n        pass\n")
    s = {x.qualname: x for x in symbols.symbols(src)}
    assert s["f"].sig == "async def f(a: int, *, b: str='x') -> bool" and (s["f"].start, s["f"].end) == (3, 6) and s["f"].doc == "Do f."
    assert s["C"].sig == "class C(Base)" and s["C.m"].kind == "method" and (s["C.m"].start, s["C.m"].end) == (10, 11)
    assert symbols.render(list(s.values())).splitlines()[-1] == "    def m(self, n)"
    assert symbols.symbols("def broken(:") == []
