"""Task loop: the contract is implemented one symbol at a time, with function-level edits and small prompts.
Run:  pytest tests/test_tasks.py -q   (real pytest via LocalRunner, scripted models, no Docker / Ollama / network)
"""
import ast
import asyncio
import uuid

import pytest

from config.settings import settings
from services import contract as ctr
from services import sandbox_service as sbx
from services import splice, tasks
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
CALC = b"def add(a, b):\n    return a + b\n"
APP = b"from calc import add\n\n\ndef total():\n    return add(1, 2)\n"


def sym(file, name, signature, behavior="Does it.", **kw):
    return {"file": file, "name": name, "signature": signature, "behavior": behavior, **kw}


MUL = sym("calc.py", "mul", "def mul(a: int, b: int) -> int", "Multiply a and b.")
POW = sym("calc.py", "power", "def power(base: int, exp: int) -> int", "base to the power exp, using mul.", depends_on=["calc.py::mul"])


def tf(path, body):
    return f"FILE: {path}\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


T_ACC = tf("tests/test_change_r1_calc.py", "from calc import mul, power\n\n\ndef test_both():\n    assert power(mul(2, 3), 2) == 36")
T_U_MUL = tf("tests/x.py", "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n\n\ndef test_mul_zero():\n    assert mul(0, 5) == 0")
T_U_MUL_WRONG = T_U_MUL.replace("== 6", "== 7")
T_U_VACUOUS = tf("tests/x.py", "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3")
T_U_POW = tf("tests/x.py", "from calc import power\n\n\ndef test_power():\n    assert power(2, 3) == 8\n\n\ndef test_power_zero():\n    assert power(5, 0) == 1")
IMPL_MUL = "def mul(a: int, b: int) -> int:\n    return a * b"
IMPL_POW = "def power(base: int, exp: int) -> int:\n    result = 1\n    for _ in range(exp):\n        result = mul(result, base)\n    return result"


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
def full_pipeline(monkeypatch):
    for k, v in dict(change_contract=True, change_tasks=True, change_task_tests=True, change_author_tests=True, change_tests_first=True).items():
        monkeypatch.setattr(settings, k, v)


def go_calc(llm, symbols=(MUL, POW), request="add mul and power functions to calc.py", files=None):
    async def go():
        svc, s, r = await _svc(llm, Script({"symbols": list(symbols)}), files)
        res = await svc.run(r, request)
        return res, s, r
    return run(go())


# ------------------------------------------------------------------ splice
SRC = (b'"""M."""\nimport os\n\n\nclass Svc:\n    @property\n    def a(self):\n        return 1\n\n    async def b(self, y: int) -> str:\n'
       b'        """Old."""\n        raise NotImplementedError("contract: Svc.b")\n\n    def c(self):\n        return 3\n\n\ndef top(x):\n    return x\n')


def test_splice_replaces_exactly_one_symbol_and_keeps_the_rest():
    new, err = splice.replace_symbol(
        SRC, "Svc.b", "```python\nimport json\n\nasync def b(self, y: int) -> str:\n    return _f(json.dumps(y))\n\ndef _f(v):\n    return v\n```",
        keep_sig="async def b(self, y: int) -> str")
    assert err == ""
    text = new.decode()
    ast.parse(text)
    assert "import os\nimport json" in text and "    async def b(self, y: int) -> str:\n        return _f(json.dumps(y))" in text
    assert "    def _f(v):" in text and text.count("def c(self)") == 1 and "@property" in text and "contract:" not in text
    deco, _ = splice.replace_symbol(SRC, "Svc.a", "def a(self):\n    return 2")           # no decorator in the reply: the existing one stays
    assert b"@property\n    def a(self):\n        return 2" in deco
    crlf, _ = splice.replace_symbol(SRC.replace(b"\n", b"\r\n"), "top", "def top(x):\n    return x * 2")
    assert b"return x * 2\r\n" in crlf and crlf.count(b"\n") == crlf.count(b"\r\n")


def test_splice_rejects_what_would_damage_the_file():
    def err(qn, code, sig=""):
        new, e = splice.replace_symbol(SRC, qn, code, sig)
        assert new is None
        return e
    assert "must stay exactly `def top(x)`" in err("top", "def top(x, y):\n    return x", "def top(x)")
    assert "must define `top`" in err("top", "x = 1")
    assert "no module-level statements" in err("top", "def top(x):\n    return x\nprint(1)")
    assert "does not parse" in err("top", "def top(x:\n")
    assert "already exist" in err("Svc.c", "def c(self):\n    return 1\n\ndef a(self):\n    return 2")
    assert "does not exist" in err("nope", "def nope(): pass")
    assert "empty" in err("top", "   \n")


def test_splice_slices_and_finds_the_enclosing_function():
    assert splice.slice_of(SRC, "Svc.b").startswith("async def b(self, y: int) -> str:") and "def c(" not in splice.slice_of(SRC, "Svc.b")
    assert splice.symbol_at(SRC, 12).qualname == "Svc.b" and splice.symbol_at(SRC, 5) is None and splice.symbol_at(SRC, 18).qualname == "top"


# ------------------------------------------------------------------ task plan
def csym(name, deps=(), file="m.py", kind="function", change="new"):
    return ctr.CSym(file, name, f"def {name}()", kind, "does it", list(deps), [], change)


def test_tasks_run_callees_first_then_caller_fixes():
    src = {"m.py": b"def low():\n    pass\n\n\ndef mid():\n    pass\n\n\ndef top():\n    pass\n",
           "app.py": b"def run():\n    a()\n    a()\n\n\ndef other():\n    a()\n\n\nVALUE = a()\n"}
    syms = [csym("top", deps=["m.py::mid"]), csym("mid", deps=["m.py::low", "requests::get"]), csym("low"), csym("K", kind="class")]
    ripple = ["app.py:2: `a()` does not fit a(x)", "app.py:3: `a()` takes no x",        # two calls in run(): ONE task
              "app.py:10: module level call", "app.py:4: blank line",                  # not inside any function: left to the single patch
              "m.py:6: inside a contract symbol"]                                       # mid() is rewritten by its own task anyway
    ordered, left = tasks.plan_tasks(syms, ripple, src)
    assert [t.name for t in ordered] == ["low", "mid", "top", "run"] and [t.id for t in ordered] == ["t1", "t2", "t3", "t4"]
    fix = ordered[-1]
    assert fix.kind == "fix_caller" and "does not fit" in fix.behavior and "takes no x" in fix.behavior
    assert len(left) == 3 and any("app.py:10" in x for x in left) and any("new class with no method" in x for x in left)
    assert [t.name for t in tasks.plan_tasks([csym("x", deps=["m.py::y"]), csym("y", deps=["m.py::x"])], [], src)[0]] == ["x", "y"]   # a cycle keeps the order


# ------------------------------------------------------------------ the loop
def test_each_symbol_is_built_from_a_small_prompt_with_only_what_it_needs():
    llm = Script(T_ACC, T_U_MUL, IMPL_MUL, T_U_POW, IMPL_POW)
    res, s, r = go_calc(llm)
    assert res.ok and res.verification == "L3-fail-to-pass" and len(llm.prompts) == 5          # no single-patch call at all
    power = llm.prompts[4]
    assert "calc.py::mul    def mul(a: int, b: int) -> int" in power                           # the signature of what it calls
    assert "return a * b" not in power and "def add(a, b)" not in power                       # not its body, not unrelated code
    assert "TESTS YOUR CODE MUST PASS" in power and "def test_power" in power
    paths = {f["path"] for f in res.files}
    assert {"tests/test_change_r1_unit_mul.py", "tests/test_change_r1_unit_power.py", "tests/test_change_r1_calc.py"} <= paths
    final = run(s.get_file(r, "calc.py", res.revision)).data.decode()
    assert "return a * b" in final and "result = mul(result, base)" in final and "contract:" not in final and "def add(a, b)" in final


def test_a_failing_unit_test_gets_a_scoped_repair():
    llm = Script(T_ACC, T_U_MUL, "def mul(a: int, b: int) -> int:\n    return a + b", IMPL_MUL, T_U_POW, IMPL_POW)
    res, *_ = go_calc(llm)
    assert res.ok and len(llm.prompts) == 6
    repair = llm.prompts[3]
    assert "YOUR PREVIOUS REPLY FAILED" in repair and "test_mul" in repair and "return a + b" in repair
    assert "def power" not in repair.split("YOUR PREVIOUS REPLY FAILED")[0].split("CURRENT CODE")[0]    # still only this task's context


def test_a_task_that_cannot_finish_hands_over_with_its_dependents():
    still_stub = 'def mul(a: int, b: int) -> int:\n    raise NotImplementedError("contract: mul")'
    wrong_sig = "def mul(a, b, c):\n    return a * b"
    patch = (block("calc.py", '    raise NotImplementedError("contract: mul")', "    return a * b")
             + block("calc.py", '    raise NotImplementedError("contract: power")', "    return base ** exp"))
    llm = Script(T_ACC, T_U_MUL, still_stub, wrong_sig, patch)                                   # power is skipped: no model call for it
    res, *_ = go_calc(llm)
    assert res.ok and len(llm.prompts) == 5
    assert "TASK LOOP RESULT" in llm.prompts[4] and "STILL TO IMPLEMENT (still stubs): calc.py::mul, calc.py::power" in llm.prompts[4]
    assert any("not done after 2 attempts" in n for n in res.notes) and any("depends on calc.py::mul" in n for n in res.notes)
    assert "is still there" in llm.prompts[3]                                                    # attempt 2 was told why attempt 1 failed


def test_callers_broken_by_a_new_signature_get_their_own_small_task(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", False)         # this scenario is about the callers, not the tests
    monkeypatch.setattr(settings, "change_task_tests", False)
    change = sym("calc.py", "add", "def add(values: list) -> int", "Sum the values.")
    llm = Script("def add(values: list) -> int:\n    return sum(values)", "def total():\n    return add([1, 2])")
    res, s, r = go_calc(llm, symbols=[change], request="make add in calc.py take a list", files={"calc.py": CALC, "app.py": APP})
    assert res.ok and len(llm.prompts) == 2
    assert "WHAT TO FIX: " in llm.prompts[1] and "does not fit add(values: list)" in llm.prompts[1] and "app.py::total" in llm.prompts[1]
    assert b"add([1, 2])" in run(s.get_file(r, "app.py", res.revision)).data


def test_a_single_symbol_takes_the_single_patch_fast_path():
    llm = Script(tf("tests/test_change_r1_calc.py", "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6"),
                 block("calc.py", '    raise NotImplementedError("contract: mul")', "    return a * b"))
    res, *_ = go_calc(llm, symbols=[MUL], request="add a mul function to calc.py")
    assert res.ok and len(llm.prompts) == 2 and "CONTRACT (signatures are fixed)" in llm.prompts[1] and "YOUR TASK" not in llm.prompts[1]


def test_unit_tests_that_never_turn_red_are_skipped_not_forced():
    llm = Script(T_ACC, T_U_VACUOUS, T_U_VACUOUS, IMPL_MUL, T_U_POW, IMPL_POW)
    res, *_ = go_calc(llm)
    paths = {f["path"] for f in res.files}
    assert res.ok and "tests/test_change_r1_unit_mul.py" not in paths and "tests/test_change_r1_unit_power.py" in paths


def test_a_wrong_unit_test_is_dropped_loudly_and_the_final_run_judges_the_code():
    llm = Script(T_ACC, T_U_MUL_WRONG, IMPL_MUL, IMPL_MUL, T_U_POW, IMPL_POW)
    res, *_ = go_calc(llm)
    paths = {f["path"] for f in res.files}
    assert res.ok and "tests/test_change_r1_unit_mul.py" not in paths and "tests/test_change_r1_unit_power.py" in paths
    assert any("unit tests still fail after 2 attempts" in n and "tests dropped, code kept" in n for n in res.notes)


def test_a_reply_that_keeps_the_stub_is_sent_back(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", False)
    monkeypatch.setattr(settings, "change_task_tests", False)
    llm = Script('def mul(a: int, b: int) -> int:\n    raise NotImplementedError("contract: mul")', IMPL_MUL, IMPL_POW)
    res, *_ = go_calc(llm)
    assert res.ok and len(llm.prompts) == 3 and "is still there" in llm.prompts[1]
