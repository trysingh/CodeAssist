"""Why /change used up its attempts, and the controls that fix it (scripted models, no Docker / Ollama / network).
Run:  pytest tests/test_reasoning_repair.py -q

  * apply_edits(partial=True): the blocks that matched are kept, the failed ones are reported by index
  * a bad edit block is a FREE repair (change_repair_attempts); the same problem twice is not
  * a failing test or a review finding is diagnosed by a separate step before the next patch
  * a plan file the patcher skipped twice gets one targeted CHANGE / NO_CHANGE check
  * the last counted attempt can use a stronger model; every run leaves an attempt log
"""
import asyncio
import uuid
from pathlib import Path

import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services.patch_service import apply_edits, parse_edits
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run

M_PY = b"def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n"
GOOD = b"def add(a, b):\n    return a + b\n"
T_CALC = b"from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


A_OK = block("m.py", "def a():\n    return 1", "def a():\n    return 10")
B_BAD = block("m.py", "def b():\n    return 999", "def b():\n    return 20")        # the model misremembered this line
B_OK = block("m.py", "def b():\n    return 2", "def b():\n    return 20")
C_OK = block("m.py", "def c():\n    return 3", "def c():\n    return 30")
DIAG = ("ROOT CAUSE: m.py::b returns 2 and the request needs 20.\nFAULT IN: code - the assertion shows 2\n"
        "FIX: 1. m.py::b - return 20\nDO NOT: quote `return 999`, that line does not exist")


class Script:
    def __init__(self, *replies):
        self.replies, self.prompts, self.models = list(replies), [], []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        self.models.append(model)
        return self.replies.pop(0)


class Doc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class Vec:
    def __init__(self, paths):
        self.paths = paths

    def similarity_search(self, repo_id, q, k=4):
        return [Doc(p) for p in self.paths]


async def _none(*a, **k):
    return None


@pytest.fixture(autouse=True)
def on(monkeypatch):
    """The legacy suite runs with these off (conftest); here they are on, as in production."""
    for k, v in dict(change_partial_apply=True, change_repair_attempts=3, change_diagnose=True, change_gap_decide=True,
                     change_max_attempts=3).items():
        monkeypatch.setattr(settings, k, v)


def pipeline(replies, files, request="update the functions", **run_kw):
    llm = Script(*replies)

    async def go():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, files)
        svc = RepoChangeService(None, Vec([p for p, _ in files]), s, llm_text=llm, llm_json=_none,
                                emit=lambda *a: None, choice_fn=_none)
        res = await svc.run(r, request, **run_kw)
        got = {p: (await s.get_file(r, p, res.revision)).data.decode() for p, _ in files} if res.revision else {}
        return res, got

    res, got = run(go())
    return res, got, llm


# --------------------------------------------------------------------------- partial apply
def test_default_is_all_or_nothing_and_partial_keeps_the_blocks_that_matched():
    edits, _ = parse_edits(A_OK + B_BAD + C_OK)
    atomic = apply_edits(edits, {"m.py": M_PY})
    assert "m.py" not in atomic.new_files and atomic.partial == {}             # the old rule: one bad block loses all three
    part = apply_edits(edits, {"m.py": M_PY}, partial=True)
    assert part.partial == {"m.py": {"applied": [1, 3], "failed": [2]}} and "edit 2" in part.errors[0]
    text = part.new_files["m.py"].decode()
    assert "return 10" in text and "return 30" in text and "return 2\n" in text


def test_a_new_file_is_never_partial():
    edits, _ = parse_edits(block("n.py", "", "x = 1") + block("n.py", "", "y = 2"))
    res = apply_edits(edits, {"n.py": None}, partial=True)
    assert res.errors and "n.py" not in res.new_files and res.partial == {}


def test_the_retry_resends_only_the_failed_block_and_nothing_is_lost():
    res, got, llm = pipeline([A_OK + B_BAD + C_OK, B_OK], [("m.py", M_PY)])
    assert res.ok and res.attempts == 2
    assert "return 10" in got["m.py"] and "return 20" in got["m.py"] and "return 30" in got["m.py"]    # all three changes
    second = llm.prompts[1]
    assert "edit block(s) 1, 3 applied" in second and "block(s) 2 FAILED" in second
    assert "ALREADY in the file contents shown below" in second
    assert "    return 10" in second and "    return 30" in second            # the file view already holds the applied blocks


def test_old_behaviour_tells_the_truth_about_a_file_that_was_left_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "change_partial_apply", False)
    res, got, llm = pipeline([A_OK + B_BAD + C_OK, A_OK + B_OK + C_OK], [("m.py", M_PY)])
    assert res.ok and "return 10" in got["m.py"] and "return 30" in got["m.py"]
    assert "left UNCHANGED" in llm.prompts[1] and "re-send ALL its blocks" in llm.prompts[1]
    assert "    return 10" not in llm.prompts[1]                                # nothing of the first reply was kept


# --------------------------------------------------------------------------- free repairs
def test_a_bad_edit_block_does_not_use_up_an_attempt(monkeypatch):
    monkeypatch.setattr(settings, "change_max_attempts", 1)
    res, got, llm = pipeline([B_BAD, B_OK], [("m.py", M_PY)])
    assert res.ok and res.attempts == 2 and res.repairs == 1 and "return 20" in got["m.py"]
    monkeypatch.setattr(settings, "change_repair_attempts", 0)
    res, got, llm = pipeline([B_BAD, B_OK], [("m.py", M_PY)])
    assert not res.ok and res.attempts == 1 and res.repairs == 0                # the old rule: the single attempt is gone


def test_the_same_problem_twice_is_not_free_and_ends_the_run(monkeypatch):
    monkeypatch.setattr(settings, "change_max_attempts", 2)
    monkeypatch.setattr(settings, "change_diagnose", False)
    res, got, llm = pipeline([B_BAD, B_BAD, B_BAD], [("m.py", M_PY)])
    assert not res.ok and res.attempts == 3 and res.repairs == 1
    assert "could not produce a safe change** after 3 attempt(s) (1 of them were cheap edit repairs)" in res.to_markdown()


def test_a_repeated_edit_failure_is_diagnosed_before_the_next_try():
    res, got, llm = pipeline([B_BAD, B_BAD, DIAG, B_OK], [("m.py", M_PY)])
    assert res.ok and "return 20" in got["m.py"]
    assert "debugging analyst" in llm.prompts[2] and "SEARCH text not found" in llm.prompts[2]
    assert "DIAGNOSIS OF THAT FAILURE" in llm.prompts[3] and "ROOT CAUSE: m.py::b" in llm.prompts[3]
    assert "EARLIER ATTEMPTS THAT DID NOT WORK" in llm.prompts[3]


# --------------------------------------------------------------------------- diagnosis after a failing test
def test_a_failing_test_is_diagnosed_from_the_traceback_before_the_patcher_retries():
    files = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC)]
    diag = ("ROOT CAUSE: calc.py::add subtracts, the assertion shows -1 != 5.\nFAULT IN: code - assert add(2, 3) == 5\n"
            "FIX: 1. calc.py::add - return a + b\nDO NOT: change the test")
    llm = Script(block("calc.py", "    return a + b", "    return a - b"), diag,
                 block("calc.py", "    return a - b", "    return a + b + 0"))

    async def go():
        s, r = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
        await s.upsert_base_files(r, files)
        emit = lambda *a: None
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=emit)
        svc = RepoChangeService(None, Vec([p for p, _ in files]), s, llm_text=llm, llm_json=_none, emit=emit,
                                sandbox=sb, choice_fn=_none)
        return await svc.run(r, "refactor calc.py add")

    res = run(go())
    assert res.ok and res.attempts == 2 and res.verification == "L2-tests"
    assert "debugging analyst" in llm.prompts[1] and "tests.test_calc::test_add" in llm.prompts[1]
    assert "+    return a - b" in llm.prompts[1]                                # it saw what the first attempt changed
    assert "DIAGNOSIS OF THAT FAILURE" in llm.prompts[2] and "calc.py::add subtracts" in llm.prompts[2]


# --------------------------------------------------------------------------- plan gap
PLAN = ("FEASIBLE: yes\nFILES:\n- a.py - add the field\n- b.js - show the field in the UI\nNEW FILES:\n- none\nRISKS:\n- none")
GAP_FILES = [("a.py", b"X = 1\nY = 1\n"), ("b.js", b"const y = 1;\nrender(y);\n")]
A1 = block("a.py", "X = 1", "X = 2")
A2 = block("a.py", "Y = 1", "Y = 2")                  # valid, but still not the file the plan needs


def test_a_plan_file_skipped_twice_is_checked_once_and_accepted_when_the_quoted_line_exists():
    gap = "VERDICT: NO_CHANGE\nEVIDENCE: render(y);\nWHY: the UI already renders y\nINSTRUCTION: none"
    res, got, llm = pipeline([A1, A2, gap], GAP_FILES, request="add the field", plan=PLAN)
    assert res.ok and {f["path"] for f in res.files} == {"a.py"}
    assert "You decide whether ONE file needs an edit" in llm.prompts[2] and "const y = 1;" in llm.prompts[2]
    assert any("no change needed in b.js" in n and "already renders" in n for n in res.notes)
    assert len(llm.prompts) == 3                                                # no third patch call was needed


def test_a_plan_file_skipped_twice_gets_an_explicit_instruction_when_the_check_says_change():
    gap = "VERDICT: CHANGE\nEVIDENCE: none\nWHY: the field is not shown\nINSTRUCTION: in render(), show y.field next to y"
    res, got, llm = pipeline([A1, A2, gap, block("b.js", "render(y);", "render(y, y.field);")], GAP_FILES,
                             request="add the field", plan=PLAN)
    assert res.ok and {f["path"] for f in res.files} == {"a.py", "b.js"}
    assert "WHAT TO CHANGE IN b.js: in render(), show y.field next to y" in llm.prompts[3]
    assert "FOCUSED PASS" in llm.prompts[3]


def test_a_no_change_answer_with_a_made_up_line_is_not_believed():
    gap = "VERDICT: NO_CHANGE\nEVIDENCE: renderField(y);\nWHY: already done\nINSTRUCTION: none"
    res, got, llm = pipeline([A1, A2, gap, block("b.js", "render(y);", "render(y, 1);")], GAP_FILES,
                             request="add the field", plan=PLAN)
    assert res.ok and {f["path"] for f in res.files} == {"a.py", "b.js"}
    assert "the line quoted as proof is not in the file" in llm.prompts[3]


# --------------------------------------------------------------------------- escalation + log
def test_the_last_counted_attempt_uses_the_escalation_model(monkeypatch):
    monkeypatch.setattr(settings, "change_max_attempts", 2)
    monkeypatch.setattr(settings, "change_repair_attempts", 0)
    monkeypatch.setattr(settings, "change_escalation_model", "big-model:cloud")
    res, got, llm = pipeline([B_BAD, B_OK], [("m.py", M_PY)])
    assert res.ok and llm.models[0] != "big-model:cloud" and llm.models[1] == "big-model:cloud"


def test_every_run_leaves_a_log_with_the_raw_replies_and_the_problems():
    res, got, llm = pipeline([B_BAD, B_BAD, B_BAD, B_BAD, B_BAD, B_BAD], [("m.py", M_PY)])
    assert not res.ok
    note = next(n for n in res.notes if n.startswith("attempt log"))
    log = Path(note.split(": ", 1)[1]).read_text(encoding="utf-8")
    assert "REQUEST (base revision 0)" in log and "return 999" in log          # the raw model reply
    assert "SEARCH text not found" in log and "[same problem again]" in log


def test_old_logs_are_pruned(monkeypatch):
    import time
    monkeypatch.setattr(settings, "change_logs_keep", 2)
    svc = RepoChangeService(None, Vec([]), RepoStore(), llm_text=Script(), llm_json=_none, emit=lambda *a: None, choice_fn=_none)
    repo = f"r_{uuid.uuid4().hex[:8]}"
    paths = []
    for _ in range(4):
        paths.append(svc._open_log(repo, "req", 0))
        time.sleep(0.01)
    assert len(list(Path(paths[-1]).parent.glob("*.log"))) == 2
    assert Path(paths[-1]).exists() and not Path(paths[0]).exists()
