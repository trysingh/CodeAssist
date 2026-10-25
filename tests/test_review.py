"""Review: a fresh-context reviewer checks coverage per requirement and unrequested edits, and routes findings back.
Run:  pytest tests/test_review.py -q   (real pytest via LocalRunner, scripted models, no Docker / Ollama / network)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
CALC = b"def add(a, b):\n    return a + b\n"
APP = b"from calc import add\n\n\ndef total():\n    return add(1, 2)\n"
REQ = "add mul and power functions to calc.py"


def sym(file, name, signature, behavior="Does it.", **kw):
    return {"file": file, "name": name, "signature": signature, "behavior": behavior, **kw}


MUL = sym("calc.py", "mul", "def mul(a: int, b: int) -> int", "Multiply a and b.")
POW = sym("calc.py", "power", "def power(base: int, exp: int) -> int", "base to the power exp, using mul.", depends_on=["calc.py::mul"])
IMPL_MUL = "def mul(a: int, b: int) -> int:\n    return a * b"
IMPL_POW = "def power(base: int, exp: int) -> int:\n    result = 1\n    for _ in range(exp):\n        result = mul(result, base)\n    return result"
PASS = {"coverage": [{"requirement": "mul exists", "status": "covered", "evidence": "calc.py::mul"},
                     {"requirement": "power exists", "status": "covered", "evidence": "calc.py::power"}], "unrequested": []}


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class Script:
    def __init__(self, *replies):
        self.replies, self.prompts, self.models = list(replies), [], []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        self.models.append(model)
        return self.replies.pop(0)


class Vec:
    def similarity_search(self, repo_id, query, k=8):
        return []


async def _none(*a, **k):
    return None


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    for k, v in dict(change_contract=True, change_tasks=True, change_review=True, change_task_tests=False,
                     change_author_tests=False, change_tests_first=False).items():
        monkeypatch.setattr(settings, k, v)


def flow(llm_replies, review, symbols=(MUL, POW), files=None, request=REQ, **run_kw):
    llm, js = Script(*llm_replies), Script({"symbols": list(symbols)}, review)

    async def go():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, (files or {"calc.py": CALC}).items())
        emit = lambda *a: None
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
        svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=js, emit=emit, sandbox=sb, choice_fn=_none)
        res = await svc.run(r, request, **run_kw)
        final = await s.get_file(r, "calc.py", res.revision) if res.revision else None
        return res, (final.data.decode() if final else "")
    res, final = run(go())
    return res, final, llm, js


def test_the_reviewer_sees_only_the_request_the_scope_and_the_final_diff():
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW], PASS)
    assert res.ok and len(llm.prompts) == 2 and len(js.prompts) == 2
    review = js.prompts[1]
    assert REQ in review and "calc.py::mul" in review and "+    return a * b" in review and "+def power" in review
    assert "You implement ONE function" not in review and "YOUR TASK" not in review         # nothing from the build conversation


def test_a_finding_pinned_on_one_symbol_goes_back_to_that_symbol_only():
    partial = {"coverage": [{"requirement": "mul rejects non-integers", "status": "partial", "evidence": "calc.py::mul"}], "unrequested": []}
    fix = ("def mul(a: int, b: int) -> int:\n    if not isinstance(a, int) or not isinstance(b, int):\n"
           "        raise TypeError('ints only')\n    return a * b")
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW, fix], partial)
    assert res.ok and len(llm.prompts) == 3 and len(js.prompts) == 2                    # one fix call, no second review
    assert "WHAT TO FIX: requirement not met (partial): mul rejects non-integers" in llm.prompts[2]
    assert "def power" not in llm.prompts[2].split("CURRENT CODE")[1].split("IMPORTS")[0]    # only the pinned symbol is shown
    assert "raise TypeError('ints only')" in final and "result = mul(result, base)" in final


def test_a_finding_that_belongs_to_no_symbol_goes_to_the_single_patch():
    missing = {"coverage": [{"requirement": "a sub function", "status": "missing", "evidence": ""}], "unrequested": []}
    patch = block("calc.py", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b")
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW, patch], missing)
    assert res.ok and res.attempts == 2 and "def sub(a, b)" in final
    assert "A reviewer who saw only the request and the final diff found" in llm.prompts[2]
    assert "requirement not met (missing): a sub function" in llm.prompts[2] and len(js.prompts) == 2


def test_a_reviewer_cannot_block_on_something_that_is_not_in_the_change():
    ghost = {"coverage": [], "unrequested": [{"file": "ghost.py", "symbol": "x", "why": "extra"}]}
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW], ghost)
    assert res.ok and len(llm.prompts) == 2
    assert any("review (advisory)" in n and "not part of the change" in n for n in res.notes)


def test_with_no_attempts_left_the_review_only_annotates(monkeypatch):
    monkeypatch.setattr(settings, "change_max_attempts", 1)
    partial = {"coverage": [{"requirement": "mul rejects non-integers", "status": "partial", "evidence": "calc.py::mul"}], "unrequested": []}
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW], partial)
    assert res.ok and len(llm.prompts) == 2 and "TypeError" not in final
    assert any(n.startswith("review (not fixed, no attempts left): calc.py::mul") for n in res.notes)


def test_a_reviewer_that_cannot_answer_never_blocks():
    res, final, llm, js = flow([IMPL_MUL, IMPL_POW], None)
    assert res.ok and any("review skipped" in n for n in res.notes)


def test_files_the_plan_did_not_list_are_pointed_out_to_the_reviewer():
    patch = (block("calc.py", '    raise NotImplementedError("contract: mul")', "    return a * b")
             + block("app.py", "def total():\n    return add(1, 2)", "def total():\n    return add(1, 2)  # tidy"))
    res, final, llm, js = flow([patch], PASS, symbols=[MUL], files={"calc.py": CALC, "app.py": APP}, request="add mul to calc.py",
                               plan="FILES\n- calc.py: add mul\n")
    assert res.ok and "FILES CHANGED THAT THE PLAN DID NOT LIST: app.py" in js.prompts[1]


def test_each_role_uses_its_own_model(monkeypatch):
    for role, m in dict(designer_model="m-design", implementer_model="m-impl", tester_model="m-test", reviewer_model="m-review").items():
        monkeypatch.setattr(settings, role, m)
    monkeypatch.setattr(settings, "change_task_tests", True)
    monkeypatch.setattr(settings, "change_author_tests", True)
    monkeypatch.setattr(settings, "change_tests_first", True)

    def tf(path, body):
        return f"FILE: {path}\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"
    acc = tf("tests/a.py", "from calc import mul, power\n\n\ndef test_both():\n    assert power(mul(2, 3), 2) == 36")
    umul = tf("tests/b.py", "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6")
    upow = tf("tests/c.py", "from calc import power\n\n\ndef test_power():\n    assert power(2, 3) == 8")
    res, final, llm, js = flow([acc, umul, IMPL_MUL, upow, IMPL_POW], PASS)
    assert res.ok
    assert llm.models == ["m-test", "m-test", "m-impl", "m-test", "m-impl"]
    assert js.models == ["m-design", "m-review"]


def test_a_review_fix_after_a_single_patch_does_not_replay_the_old_patch():
    wrong_sig = "def mul(a, b, c):\n    return a * b"
    patch = (block("calc.py", '    raise NotImplementedError("contract: mul")', "    return a * b")
             + block("calc.py", '    raise NotImplementedError("contract: power")', "    return base ** exp"))
    fix = "def mul(a: int, b: int) -> int:\n    if not isinstance(a, int):\n        raise TypeError('ints only')\n    return a * b"
    partial = {"coverage": [{"requirement": "mul rejects non-integers", "status": "partial", "evidence": "calc.py::mul"}], "unrequested": []}
    res, final, llm, js = flow([wrong_sig, wrong_sig, patch, fix], partial)
    assert res.ok and "raise TypeError('ints only')" in final and "return base ** exp" in final
