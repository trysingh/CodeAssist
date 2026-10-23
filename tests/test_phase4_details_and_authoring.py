"""Phase 4: what the sandbox reports (not just counts) and tests written for a proposed change.
Run from the repo root:  pytest tests -q
No Docker, Ollama or network needed: real pytest runs in a subprocess through LocalRunner, the model is scripted.
"""
import asyncio
import os
import uuid

import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run

GOOD = b"def add(a, b):\n    return a + b\n"
T_CALC = b"from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
OFFLINE_MODULE = (b"raise ConnectionError(\"HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443): "
                  b"Max retries exceeded (Caused by NameResolutionError: Failed to resolve 'openaipublic.blob.core.windows.net')\")\n")
T_BROKEN_IMPORT = b"import vocab\nfrom calc import add\n\n\ndef test_needs_vocab():\n    assert add(1, 1) == 2\n"


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def new_file(path, text):
    return f"FILE: {path}\n<<<<<<< SEARCH\n=======\n{text}\n>>>>>>> REPLACE\n"


def rid():
    return f"r_{uuid.uuid4().hex[:8]}"


class Script:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


class Vec:
    def similarity_search(self, repo_id, query, k=8):
        return []


async def _no_json(*a, **k):
    return None


async def _no_choice(*a, **k):
    return None


async def _svc(llm, files, runner=None):
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, files)
    emit = lambda *a: None
    sb = sbx.SandboxService(s, runner or sbx.LocalRunner(), enabled=True, emit=emit)
    svc = RepoChangeService(None, Vec(), s, llm_text=llm, llm_json=_no_json, emit=emit, sandbox=sb, choice_fn=_no_choice)
    return svc, s, r


@pytest.fixture
def authoring(monkeypatch):
    monkeypatch.setattr(settings, "change_author_tests", True)


# ----------------------------------------------------------------------------- reading the container's report
JUNIT_COLLECT = """<testsuites><testsuite name="pytest" errors="1" failures="0" skipped="0" tests="1">
<testcase classname="" name="tests.test_x"><error message="collection failure">tests/test_x.py:9: in &lt;module&gt;
    import vocab
vocab.py:1: in &lt;module&gt;
    raise ConnectionError("boom")
E   ConnectionError: Failed to resolve 'openaipublic.blob.core.windows.net'</error></testcase></testsuite></testsuites>"""


def test_a_collection_error_says_what_failed_not_just_that_it_did():
    passed, failed, _, cases = sbx.parse_junit(JUNIT_COLLECT)
    c = cases[0]
    assert (passed, failed) == (0, 1) and c.kind == "collect"
    assert "openaipublic.blob.core.windows.net" in c.message and c.message.startswith("collection failure:")
    assert "import vocab" in c.detail                                  # the traceback is kept, cause at the end


def test_long_tracebacks_keep_the_cause_at_the_end():
    body = "\n".join(f"frame {i}" for i in range(2000)) + "\nE   ValueError: the real cause"
    xml = f'<testsuite><testcase classname="t" name="a"><failure message="">{body}</failure></testcase></testsuite>'
    _, _, _, cases = sbx.parse_junit(xml)
    assert cases[0].message == "ValueError: the real cause" and cases[0].detail.endswith("the real cause")


def test_hints_name_the_usual_sandbox_only_causes():
    net = sbx.hint_for("requests.exceptions.ConnectionError: host='openaipublic.blob.core.windows.net' tiktoken")
    assert "openaipublic.blob.core.windows.net" in net and "--network none" in net and "tiktoken" in net
    assert "jinja2" in sbx.hint_for("ModuleNotFoundError: No module named 'jinja2'", {"services"})
    assert sbx.hint_for("No module named 'services'", {"services"}) == ""            # the repo's own module: not an image problem
    assert "read-only" in sbx.hint_for("OSError: [Errno 30] Read-only file system: '/x'")
    assert sbx.hint_for("assert 1 == 2") == ""


def test_docker_that_cannot_start_the_container_is_infrastructure_not_a_failing_test():
    rep = sbx.parse_run(sbx.RawRun(stdout="docker: Error response from daemon: pull access denied", exit_code=125),
                        "pytest", "docker")
    assert not rep.ran and "could not start the sandbox container" in rep.note and "pull access denied" in rep.note


def test_pytest_keeps_going_after_a_file_that_cannot_be_imported():
    assert "--continue-on-collection-errors" in sbx.pytest_argv(["tests/a.py"])


# ----------------------------------------------------------------------------- offline import failures (the tiktoken case)
def _offline_repo(extra=()):
    return {"calc.py": GOOD, "vocab.py": OFFLINE_MODULE, "tests/test_calc.py": T_BROKEN_IMPORT, **dict(extra)}


def test_when_no_test_can_even_be_imported_the_run_is_not_a_pass():
    """Every test file broken the same way before and after the change used to read as 'ok, failures pre-existed'."""
    async def go():
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, list(_offline_repo().items()))
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        v = await sb.verify(r, 0, {"calc.py": b"def add(a, b):\n    return a + b + 0\n"})
        assert not v.ran and not v.ok
        assert "no test produced a result" in v.report.note and "openaipublic.blob.core.windows.net" in v.report.note
        assert "--network none" in v.report.note and v.preexisting == 1 and v.preexisting_cases[0].kind == "collect"
    run(go())


def test_one_broken_file_does_not_hide_the_other_tests():
    async def go():
        files = _offline_repo({"tests/test_ok.py": T_CALC})
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, list(files.items()))
        sb = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        v = await sb.verify(r, 0, {"calc.py": b"def add(a, b):\n    return a + b + 0\n"})
        assert v.ran and v.ok and v.report.passed == 1                 # the healthy file really ran
        assert v.preexisting == 1 and v.hints and "--network none" in v.hints[0]
        d = v.to_dict()
        assert d["preexisting_details"][0]["kind"] == "collect" and os.path.exists(d["log"])
    run(go())


def test_the_chat_answer_lists_failing_tests_and_the_likely_cause():
    async def go():
        files = list(_offline_repo({"tests/test_ok.py": T_CALC}).items())
        llm = Script(block("calc.py", "    return a + b", "    return a + b + 0"))
        svc, s, r = await _svc(llm, files)
        res = await svc.run(r, "tidy add in calc.py")
        md = res.to_markdown()
        assert res.ok and "**Sandbox details:**" in md
        assert "also fails on the untouched base" in md and "openaipublic.blob.core.windows.net" in md
        assert "likely cause" in md and "full container output" not in md or "sandbox_logs" in md
    run(go())


def test_a_change_with_a_failing_sandbox_does_not_crash_the_pipeline():
    """Regression: _sandbox called self.notifier.notify(...) whenever the verdict was not ok."""
    async def go():
        llm = Script(block("calc.py", "    return a + b", "    return a - b"),            # breaks test_add
                     block("calc.py", "    return a - b", "    return a + b + 0"))
        svc, s, r = await _svc(llm, [("calc.py", GOOD), ("tests/test_calc.py", T_CALC)])
        res = await svc.run(r, "refactor add in calc.py")
        assert res.ok and res.attempts == 2 and not any("AttributeError" in e for e in res.errors)
    run(go())


# ----------------------------------------------------------------------------- image recipe
def test_image_prefetches_what_the_code_downloads_because_the_run_has_no_network(monkeypatch):
    r = sbx.DockerRunner()
    with_tk = r.dockerfile("fastapi>=0.1\ntiktoken>=0.7.0\n")
    assert "ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken" in with_tk and "tiktoken.get_encoding" in with_tk
    assert with_tk.index("pip install") < with_tk.index("get_encoding") < with_tk.index("ENTRYPOINT")
    assert "tiktoken" not in r.dockerfile("fastapi>=0.1\n")
    assert r.image_tag("tiktoken\n") != r.image_tag("tiktoken==0\n")
    monkeypatch.setattr(settings, "sandbox_build_run", ["python -m nltk.downloader punkt"])
    assert "RUN python -m nltk.downloader punkt" in r.dockerfile("x\n")


# ----------------------------------------------------------------------------- tests written for the change
REPO = [("calc.py", GOOD), ("tests/test_calc.py", T_CALC)]
ADD_MUL = block("calc.py", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b")
MUL_TEST = "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n\n\ndef test_mul_zero():\n    assert mul(5, 0) == 0\n"


def test_generated_tests_run_and_must_fail_on_the_old_code(authoring):
    async def go():
        llm = Script(ADD_MUL, new_file("tests/test_whatever.py", MUL_TEST))
        svc, s, r = await _svc(llm, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and res.verification == "L3-fail-to-pass" and len(llm.prompts) == 2
        assert "add a mul function" in llm.prompts[1] and "def mul" in llm.prompts[1]     # the author saw request, diff, source
        assert "from calc import add" in llm.prompts[1]                                    # and how this repo's tests import
        gen = [f["path"] for f in res.files if f["path"].startswith("tests/")]
        assert gen == ["tests/test_change_r1_calc.py"]
        assert (await s.get_file(r, gen[0])) is not None                                   # part of the saved revision
        assert res.tests["proof"]["proves"] == 1 and res.tests["proof"]["tests"] == 2
        md = res.to_markdown()
        assert "fail->pass" in md and "L3-fail-to-pass" in md
    run(go())


def test_generated_tests_that_also_pass_on_old_code_are_rewritten_once(authoring):
    async def go():
        weak = "from calc import add\n\n\ndef test_add_still_works():\n    assert add(2, 2) == 4\n"
        llm = Script(ADD_MUL, new_file("tests/t.py", weak), new_file("tests/t.py", MUL_TEST))
        svc, s, r = await _svc(llm, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and res.verification == "L3-fail-to-pass" and len(llm.prompts) == 3
        assert "also PASS on the old code" in llm.prompts[2] and "test_add_still_works" in llm.prompts[2]
    run(go())


def test_tests_that_never_fail_on_old_code_are_kept_but_not_called_proof(authoring):
    async def go():
        weak = "from calc import add\n\n\ndef test_add_still_works():\n    assert add(2, 2) == 4\n"
        llm = Script(ADD_MUL, new_file("tests/t.py", weak), new_file("tests/t.py", weak))
        svc, s, r = await _svc(llm, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and res.verification == "L2-tests" and res.tests["proof"]["proves"] == 0
        assert any("regression guards only" in n for n in res.notes) and "do not prove the change" in res.to_markdown()
    run(go())


def test_a_wrong_generated_test_is_left_out_loudly_not_allowed_to_sink_the_change(authoring):
    async def go():
        wrong = "from calc import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 7\n"
        again = block("calc.py", "    return a * b", "    return a * b  # unchanged behaviour")
        llm = Script(ADD_MUL, new_file("tests/t.py", wrong), again)
        svc, s, r = await _svc(llm, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and not [f for f in res.files if f["path"].startswith("tests/")]      # not saved with the revision
        assert res.tests["generated_dropped"] and "left out" in res.to_markdown()
        assert "GENERATED for this change" in llm.prompts[2]                               # the patcher was told whose fault it may be
    run(go())


def test_unusable_test_author_output_never_blocks_the_change(authoring):
    async def go():
        llm = Script(ADD_MUL, "sorry, I cannot do that", "still no file")
        svc, s, r = await _svc(llm, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and res.verification == "L2-tests" and any("no usable tests" in n for n in res.notes)
        llm2 = Script(ADD_MUL, "NO_TESTS: needs a browser")
        svc, s, r = await _svc(llm2, REPO)
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and any("NO_TESTS" in n for n in res.notes)
    run(go())


def test_no_test_author_call_when_the_sandbox_cannot_run_anything(authoring):
    class Down:
        name = "docker"

        async def available(self):
            return False, "Docker CLI not found on PATH"

    async def go():
        llm = Script(ADD_MUL)
        svc, s, r = await _svc(llm, REPO, runner=Down())
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and len(llm.prompts) == 1 and any("no tests were written" in n for n in res.notes)
    run(go())


def test_the_switch_turns_test_writing_off():
    async def go():
        llm = Script(ADD_MUL)
        svc, s, r = await _svc(llm, REPO)          # conftest: ENGINE_CHANGE_AUTHOR_TESTS=false
        res = await svc.run(r, "add a mul function to calc.py")
        assert res.ok and len(llm.prompts) == 1 and res.verification == "L2-tests"
    run(go())
