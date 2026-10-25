"""Generated tests may only patch what exists; a setup error is not 'red'; per-task test runs are capped.
Run:  pytest tests/test_test_side_guards.py -q   (real pytest via LocalRunner, scripted model, no Docker / Ollama / network)
"""
import asyncio
import uuid

import pytest

from config.settings import settings
from services import patch_targets as pt
from services import sandbox_service as sbx
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
SRC = {
    "calc.py": b"import os\n\n\ndef add(a, b):\n    return a + b\n\n\nclass Box:\n    size = 1\n\n    def open(self):\n        pass\n",
    "pkg/__init__.py": b"",
    "pkg/mod.py": b"def f():\n    pass\n",
}
HEAD = "from unittest import mock\nimport calc as c\nfrom calc import Box\n\n\ndef test_x(monkeypatch):\n    "


def flagged(line):
    return pt.missing_patch_targets(HEAD + line + "\n", SRC)


@pytest.mark.parametrize("line", [
    'mock.patch("calc.nope")',
    'mock.patch("calc.Box.close")',
    'mock.patch("pkg.mod.g")',
    'mock.patch("pkg.gone.f")',
    'monkeypatch.setattr("calc.nope", 1)',
    'monkeypatch.setattr(c, "nope", 1)',
    'mock.patch.object(Box, "close")',
])
def test_a_target_that_does_not_exist_is_reported(line):
    assert len(flagged(line)) == 1 and "does not exist" in flagged(line)[0]


@pytest.mark.parametrize("line", [
    'mock.patch("calc.add")',
    'mock.patch("calc.os")',                                    # an imported name counts
    'mock.patch("calc.Box.open")',
    'mock.patch("calc.Box.size")',
    'mock.patch("pkg.mod.f")',
    'mock.patch("httpx.AsyncClient")',                          # third party: not ours to judge
    'mock.patch("calc.nope", create=True)',
    'monkeypatch.setattr("calc.nope", 1, raising=False)',
    'monkeypatch.setattr(c, "add", lambda: 1)',
    'mock.patch.object(Box, "open")',
    'monkeypatch.setenv("A.B", "1")',                           # not a patch of code
])
def test_a_target_that_exists_or_is_not_ours_passes(line):
    assert flagged(line) == []


def test_the_contract_stub_counts_as_existing():
    stubbed = {**SRC, "calc.py": SRC["calc.py"] + b"\n\ndef mul(a, b):\n    raise NotImplementedError('contract: mul')\n"}
    assert pt.missing_patch_targets('from unittest import mock\n\n\ndef test_m():\n    mock.patch("calc.mul")\n', stubbed) == []


def test_a_module_level_getattr_or_star_import_is_left_alone():
    open_src = {"m.py": b"from os import *\n"}
    assert pt.missing_patch_targets('from unittest import mock\n\n\ndef test_m():\n    mock.patch("m.anything")\n', open_src) == []


def test_a_setup_error_on_a_missing_patch_target_is_not_red_when_a_contract_exists():
    rep = sbx.RunReport(mode="pytest", ran=True, failures=[sbx.CaseResult(
        "tests.t::a", "error", "failed on setup with \"AttributeError: <module 'views.api_view' from '/work/views/api_view.py'> "
        "does not have the attribute 'engine_controller'\"")])
    v = RepoChangeService._red_verdict
    assert v(rep, {"views"}, strict=True)[0] == "bad"
    assert v(rep, {"views"}, strict=False)[0] == "red"          # without a contract the name may be one the change adds


# ------------------------------------------------------------------ in the loop
CALC = b"def add(a, b):\n    return a + b\n"


def tfile(body):
    return f"FILE: tests/test_change_r1_calc.py\n<<<<<<< SEARCH\n=======\n{body}\n>>>>>>> REPLACE\n"


T_BAD = tfile("from unittest import mock\nfrom calc import mul\n\n\ndef test_mul():\n    with mock.patch(\"calc.helper_that_is_not_there\"):\n        assert mul(2, 3) == 6")
T_OK = tfile("from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6")
CODE = ("FILE: calc.py\n<<<<<<< SEARCH\ndef add(a, b):\n    return a + b\n=======\n"
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n>>>>>>> REPLACE\n")


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


class SpyRunner(sbx.LocalRunner):
    def __init__(self):
        super().__init__()
        self.timeouts = []

    async def run(self, requirements, tar, argv, timeout):
        self.timeouts.append(timeout)
        return await super().run(requirements, tar, argv, timeout)


def go(llm, runner):
    async def inner():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, {"calc.py": CALC}.items())
        emit = lambda *a: None
        sb = sbx.SandboxService(s, runner, enabled=True, emit=emit)
        svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=_none, emit=emit, sandbox=sb, choice_fn=_none)
        return await svc.run(r, "add a mul function to calc.py")
    return run(inner())


def test_a_test_that_patches_a_missing_name_is_rewritten_before_any_container_runs(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", True)
    monkeypatch.setattr(settings, "change_tests_first", True)
    monkeypatch.setattr(settings, "change_task_test_timeout_s", 17)
    llm, runner = Script(T_BAD, T_OK, CODE), SpyRunner()
    res = go(llm, runner)
    assert res.ok and len(llm.prompts) == 3
    assert "patch target `calc.helper_that_is_not_there` does not exist" in llm.prompts[1]
    assert runner.timeouts[0] == 17 and len(runner.timeouts) >= 2       # the red check runs with the per-task cap
