"""Contract: typed signatures designed first, applied as in-memory stubs, checked before any body exists.
Run:  pytest tests/test_contract.py -q   (real pytest via LocalRunner, scripted models, no Docker / Ollama / network)
"""
import ast
import asyncio
import uuid

import pytest

from config.settings import settings
from services import contract as ctr
from services import sandbox_service as sbx
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
CALC = b"def add(a, b):\n    return a + b\n"
APP = b"from calc import add\n\n\ndef total():\n    return add(1, 2)\n"
REQ = "add a mul function to calc.py"
TPATH = "tests/test_change_r1_calc.py"
STUB_MUL = '    raise NotImplementedError("contract: mul")'


def sym(file, name, signature, behavior="Does it.", **kw):
    return {"file": file, "name": name, "signature": signature, "behavior": behavior, **kw}


MUL = sym("calc.py", "mul", "def mul(a: int, b: int) -> int", "Multiply a and b.")


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def tfile(body):
    return f"FILE: {TPATH}\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"


T_MUL = tfile("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n\n\ndef test_mul_zero():\n    assert mul(0, 5) == 0")
T_WRONG_NAME = tfile("from calc import multiply\n\n\ndef test_mul():\n    assert multiply(2, 3) == 6")
IMPL_MUL = block("calc.py", STUB_MUL, "    return a * b")


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


async def _svc(llm, js, files=None):
    s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(r, (files or {"calc.py": CALC}).items())
    emit = lambda *a: None
    sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
    return RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=js, emit=emit, sandbox=sb, choice_fn=_none), s, r


@pytest.fixture(autouse=True)
def contract_on(monkeypatch):
    monkeypatch.setattr(settings, "change_contract", True)
    monkeypatch.setattr(settings, "change_author_tests", False)     # tests-first is switched on by the tests that need it


def tdd(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", True)
    monkeypatch.setattr(settings, "change_tests_first", True)


# ------------------------------------------------------------------ pure functions
SRC = {"svc.py": b'"""Svc."""\nimport os\n\n\nclass Svc:\n    def a(self, x):\n        return x\n\n\n@deco\ndef old(a, b):\n    return a\n\n\nif __name__ == "__main__":\n    print(1)\n',
       "crlf.py": b"def f():\r\n    return 1\r\n"}
KNOWN = list(SRC) + ["pkg/__init__.py"]


def test_stubs_cover_new_changed_and_unchanged_symbols():
    data = {"symbols": [
        sym("svc.py", "Svc.b", "def b(self, y: int) -> str", "Return y as text."),
        sym("svc.py", "new_fn", "async def new_fn(d: Dict[str, int]) -> None", imports=["from typing import Dict"]),
        sym("svc.py", "old", "def old(a)", "Changed arity.", depends_on=["svc.py::new_fn", "requests::get"]),
        sym("svc.py", "Svc.a", "def a(self, x)", "same signature, behaviour changes"),
        sym("crlf.py", "g", "def g() -> int"),
        sym("pkg/new.py", "K", "class K(Base)", "A thing."),
        sym("pkg/new.py", "K.run", "def run(self) -> None", imports=["from base import Base"]),
    ]}
    syms, probs = ctr.parse(data, SRC, KNOWN)
    assert probs == [] and {s.name: s.change for s in syms} == {
        "Svc.b": "new", "new_fn": "new", "old": "signature", "Svc.a": "behavior", "g": "new", "K": "new", "K.run": "new"}
    out = ctr.apply_stubs(syms, SRC)
    assert set(out) == {"svc.py", "crlf.py", "pkg/new.py"}                  # Svc.a keeps its body: nothing to stub
    for b in out.values():
        ast.parse(b.decode())
    svc = out["svc.py"].decode()
    assert "return x" in svc and svc.index("def b(self") < svc.index("@deco") < svc.index("def old(a):") < svc.index("def new_fn")
    assert svc.index("def new_fn") < svc.index('if __name__') and "from typing import Dict" in svc and "import os" in svc
    assert 'raise NotImplementedError("contract: old")' in svc and "return a\n" not in svc.split("def old")[1].split("if __name__")[0]
    assert b"\r\n" in out["crlf.py"] and b"\n" not in out["crlf.py"].replace(b"\r\n", b"")
    assert out["pkg/new.py"].decode().startswith("from base import Base")
    assert sorted(m for b in out.values() for m in ctr.STUB_RE.findall(b.decode())) == ["K.run", "Svc.b", "g", "new_fn", "old"]
    assert ctr.check_deps(syms, {**SRC, **out}) == []                       # new_fn resolves because the stub exists; requests::get is external


def test_the_contract_is_checked_against_the_repo():
    def probs(*items, **kw):
        return ctr.parse({"symbols": list(items)}, SRC, KNOWN, **kw)[1]
    assert "did you mean svc.py" in probs(sym("src/svc.py", "f", "def f()"))[0]
    assert "existing directory" in probs(sym("nowhere/x.py", "f", "def f()"))[0]
    assert "test file" in probs(sym("tests/test_x.py", "f", "def f()"))[0]
    assert "ONE `def`" in probs(sym("svc.py", "f", "def f(:"))[0] and "ONE `def`" in probs(sym("svc.py", "f", "@x\ndef f()"))[0]
    assert "does not match" in probs(sym("svc.py", "g", "def f()"))[0]
    assert "listed twice" in probs(sym("svc.py", "f", "def f()"), sym("svc.py", "f", "def f()"))[0]
    assert "class Nope does not exist" in probs(sym("svc.py", "Nope.m", "def m(self)"))[0]
    assert "behavior is empty" in probs(sym("svc.py", "f", "def f()", behavior=""))[0]
    assert "plain import" in probs(sym("svc.py", "f", "def f()", imports=["x = 1"]))[0]
    assert "too many" in probs(sym("svc.py", "a", "def a()"), sym("svc.py", "b", "def b()"), max_symbols=1)[0]
    assert ctr.parse({"oops": 1}, SRC, KNOWN)[1] and ctr.parse(None, SRC, KNOWN)[1]
    syms, _ = ctr.parse({"symbols": [sym("svc.py", "f", "def f()", depends_on=["svc.py::ghost", "gone.py::x"])]}, SRC, KNOWN)
    d = ctr.check_deps(syms, SRC)
    assert len(d) == 2 and "has no 'ghost'" in d[0] and "does not exist" in d[1]


# ------------------------------------------------------------------ the whole flow
def test_stubs_are_the_starting_point_tests_go_red_on_them_and_the_body_makes_them_green(monkeypatch):
    tdd(monkeypatch)

    async def go():
        llm, js = Script(T_MUL, IMPL_MUL), Script({"symbols": [MUL]})
        svc, s, r = await _svc(llm, js)
        res = await svc.run(r, REQ)
        return llm, js, res, await s.get_file(r, "calc.py", res.revision)
    llm, js, res, f = run(go())
    assert res.ok and res.verification == "L3-fail-to-pass"
    assert "def mul(a: int, b: int) -> int" in llm.prompts[0] and "NotImplementedError" in llm.prompts[0]     # tests see the contract
    assert "CONTRACT (signatures are fixed)" in llm.prompts[1] and "calc.py::mul [new]" in llm.prompts[1] and STUB_MUL in llm.prompts[1]
    assert b"return a * b" in f.data and b"contract:" not in f.data


def test_a_stub_left_behind_is_never_saved(monkeypatch):
    tdd(monkeypatch)
    untouched = block("calc.py", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b  # sum")

    async def go():
        llm = Script(T_MUL, untouched, IMPL_MUL)
        svc, s, r = await _svc(llm, Script({"symbols": [MUL]}))
        return llm, await svc.run(r, REQ)
    llm, res = run(go())
    assert res.ok and res.attempts == 2 and "Unimplemented contract stub(s): mul" in llm.prompts[2]


def test_a_test_that_ignores_the_contract_is_rewritten(monkeypatch):
    tdd(monkeypatch)

    async def go():
        llm = Script(T_WRONG_NAME, T_MUL, IMPL_MUL)
        svc, s, r = await _svc(llm, Script({"symbols": [MUL]}))
        return llm, await svc.run(r, REQ)
    llm, res = run(go())
    assert res.ok and "breaks the contract" in llm.prompts[1] and "def test_mul_zero" in llm.prompts[1] + T_MUL


def test_a_contract_with_an_invented_dependency_is_repaired_once():
    bad = sym("calc.py", "mul", "def mul(a: int, b: int) -> int", depends_on=["calc.py::nope"])

    async def go():
        js = Script({"symbols": [bad]}, {"symbols": [MUL]})
        svc, s, r = await _svc(Script(IMPL_MUL), js)
        return js, await svc.run(r, REQ)
    js, res = run(go())
    assert res.ok and len(js.prompts) == 2 and "has no 'nope'" in js.prompts[1]


def test_without_a_valid_contract_the_change_runs_as_before():
    async def go():
        js = Script({"oops": 1}, None)
        svc, s, r = await _svc(Script(block("calc.py", "def add(a, b):\n    return a + b",
                                            "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b")), js)
        return await svc.run(r, REQ)
    res = run(go())
    assert res.ok and any("contract skipped" in n for n in res.notes)


def test_callers_broken_by_a_new_signature_are_shown_and_fixed():
    change = sym("calc.py", "add", "def add(values: list) -> int", "Sum the values.")
    fix = (block("calc.py", '    raise NotImplementedError("contract: add")', "    return sum(values)")
           + block("app.py", "return add(1, 2)", "return add([1, 2])"))

    async def go():
        llm = Script(fix)
        svc, s, r = await _svc(llm, Script({"symbols": [change]}), {"calc.py": CALC, "app.py": APP})
        return llm, await svc.run(r, "make add in calc.py take a list")
    llm, res = run(go())
    assert res.ok
    assert "CALLERS THESE SIGNATURES BREAK" in llm.prompts[0] and "app.py" in llm.prompts[0] and "def total" in llm.prompts[0]


def test_the_outline_keeps_every_file_and_adds_parameters_where_they_matter():
    big = {f"m{i}.py": f"def fn_{i}(alpha, beta, gamma=1):\n    return 1\n".encode() for i in range(6)}

    async def go(limit):
        svc, s, r = await _svc(Script(), Script(), big)
        state = {"repo_id": r, "base_rev": 0, "working": {}, "targets": ["m4.py"], "known": sorted(big)}
        return await svc._outline(state, limit=limit)
    tight, roomy = run(go(90)), run(go(5000))
    assert len(tight.splitlines()) == 6                              # nothing dropped
    assert "fn_4(alpha, beta, gamma=)" in tight and "fn_0(alpha" not in tight and "fn_0" in tight    # the file in play is upgraded first
    assert all("(alpha, beta, gamma=)" in row for row in roomy.splitlines())
