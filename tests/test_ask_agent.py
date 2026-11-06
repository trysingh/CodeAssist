"""Ask mode: default routing, find/grep/ls/read tools, the investigation loop and the sandbox check of proposed edits.
No Ollama, Docker or network: the model is scripted and the sandbox is faked."""
import asyncio
import uuid

import pytest

from config.settings import settings
from services.code_agent import (CodeAgent, RepoTools, _strip_line_numbers, leads_from_context, parse_action)
from services.intent_router import route_intent
from services.patch_service import Edit
from services.repo_store import RepoStore
from services.sandbox_service import CaseResult, RunReport, SandboxVerdict

run = asyncio.run

APP = (
    "import os\n\n\n"
    "def total(items):\n"
    "    return sum(i.price for i in items)\n\n\n"
    "def discount(amount, pct):\n"
    "    return amount - amount * pct\n"
)
FILES = {
    "app.py": APP,
    "services/cart.py": "from app import total\n\n\ndef checkout(items):\n    return total(items)\n",
    "services/util.py": "def helper():\n    return 1\n",
    "util.py": "X = 1\n",
    "README.md": "# demo\n",
}


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def act(tool, **args):
    import json
    return "ACTION: " + json.dumps({"tool": tool, "args": args})


class Scripted:
    """Stands in for the chat model: returns the next scripted reply and remembers what it was shown."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    async def __call__(self, messages, repo_id=""):
        self.seen.append([str(m.content) for m in messages])
        return self.replies.pop(0)


class FakeSandbox:
    def __init__(self, verdict=None):
        self.verdict, self.calls = verdict, []

    async def verify(self, repo_id, base_rev, working, new_tests=()):
        self.calls.append((base_rev, sorted(working), list(new_tests)))
        return self.verdict or SandboxVerdict(RunReport(mode="pytest", ran=True, passed=3, backend="docker", strategy="affected",
                                                        selected=["tests/test_app.py"]))


async def make(files=None):
    store, rid = RepoStore(), uuid.uuid4().hex[:12]
    await store.upsert_base_files(rid, [(p, t.encode()) for p, t in (files or FILES).items()])
    return store, rid


def agent_for(llm, sandbox=None, store=None):
    return CodeAgent(store, sandbox or FakeSandbox(), llm=llm, emit=lambda *a, **k: None)


# ----------------------------------------------------------------------------- routing
def test_no_prefix_is_ask_by_default(monkeypatch):
    monkeypatch.setattr(settings, "router_auto", False)

    async def go():
        async def boom(*a, **k):
            raise AssertionError("no model call is needed when nothing can change the mode")
        for text in ("fix the bug in a.py", "add pagination to the repos endpoint", "implement login", "debug this traceback"):
            d = await route_intent(text, noul_fn=boom, model="m")
            assert (d.intent, d.query, d.source) == ("ask", text, "default")
        # explicit prefixes still switch mode
        assert (await route_intent("/change add a table", noul_fn=boom, model="m")).intent == "change"
        assert (await route_intent("/fix the bug", noul_fn=boom, model="m")).intent == "change"
        assert (await route_intent("/plan add a table", noul_fn=boom, model="m")).intent == "plan"
        assert (await route_intent("/help", noul_fn=boom, model="m")).intent == "help"
    run(go())


def test_auto_router_still_available_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "router_auto", True)
    assert run(route_intent("fix the bug in a.py", model="")).intent == "change"


# ----------------------------------------------------------------------------- tools
def tools():
    return RepoTools({p: t.encode() for p, t in FILES.items()})


def test_find_ls_and_resolve():
    t = tools()
    assert "services/cart.py" in t.find("cart")
    assert "app.py" in t.find("*.py") and "README.md" not in t.find("*.py")
    assert "No file matches" in t.find("zzz")
    ls = t.ls("services")
    assert "cart.py" in ls and "util.py" in ls
    assert "services/  (2 files)" in t.ls("")
    assert t.resolve("app.py")[0] == "app.py"
    assert t.resolve("cart.py")[0] == "services/cart.py"          # unique basename
    p, err = t.resolve("util.py")                                  # exact root file wins over services/util.py
    assert p == "util.py"
    assert "ERROR" in t.resolve("nope.py")[1]


def test_grep_lines_context_and_absence():
    t = tools()
    out = t.grep("def total")
    assert "1 match(es) in 1 file(s)" in out and "app.py:4: def total(items):" in out
    assert "app.py-5-" in out                                      # context line, marked with dashes
    miss = t.grep("does_not_exist")
    assert "No match" in miss and "5 file(s) searched" in miss     # absence is stated with its scope
    assert "services/cart.py" in t.grep("total", path="services/")
    assert "app.py" not in t.grep("total", path="services/")
    assert "searched as plain text" in t.grep("total(")            # invalid regex falls back to literal
    assert "too long or too complex" in t.grep("(a+)+")            # backtracking guard
    assert "ERROR" in t.grep("")


def test_read_numbers_lines_truncates_and_tracks_coverage(monkeypatch):
    t = tools()
    out = t.read("app.py")
    assert "(COMPLETE FILE)" in out and "4| def total(items):" in out
    assert t.fully_read("app.py")

    big = RepoTools({"big.py": "\n".join(f"x{i} = {i}" for i in range(1, 501)).encode()})
    monkeypatch.setattr(settings, "ask_read_chars", 400)
    first = big.read("big.py")
    assert "TRUNCATED" in first and "start=" in first
    assert not big.fully_read("big.py")
    nxt = int(first.split("start=")[1].split()[0])
    while not big.fully_read("big.py"):                            # following the hints reads the whole file
        out = big.read("big.py", nxt)
        if "TRUNCATED" not in out:
            break
        nxt = int(out.split("start=")[1].split()[0])
    assert big.fully_read("big.py")
    assert big.covers("big.py", 250)


def test_run_dedupes_counts_and_never_raises():
    t = tools()
    first = t.run("grep", {"pattern": "total"})
    assert "match" in first
    assert "already ran exactly this call" in t.run("grep", {"pattern": "total"})
    assert t.calls == {"grep": 1}
    assert "unknown tool" in t.run("rm", {"path": "/"})
    assert "ERROR" in t.run("read", {"path": "../../etc/passwd"})   # only repo keys can be read
    assert "ERROR" in t.run("read", "not a dict")


def test_parse_action_variants():
    assert parse_action("The answer is 42.") is None
    assert parse_action('ACTION: {"tool": "grep", "args": {"pattern": "x"}}') == ("grep", {"pattern": "x"}, "")
    assert parse_action('I will look.\nACTION: {"tool": "read", "args": {"path": "a.py"}}')[0] == "read"
    assert parse_action('```json\n{"tool": "find", "args": {"pattern": "a"}}\n```')[0] == "find"
    assert parse_action('ACTION: {"tool": "ls", "path": "x"}')[1] == {"path": "x"}      # flat args accepted
    assert parse_action("ACTION: not json")[2] != ""                                      # bad action: told how to fix


def test_leads_keep_named_files_as_read_me_lines_and_respect_cap():
    ctx = ("--- [FULL_FILE: COMPLETE] app.py (3 chunks) ---\nBIG BODY " + "x" * 5000 +
           "\n\n--- [PRIMARY_MATCH] File: b.py (Chunk #1) ---\n" + "y" * 5000)
    out = leads_from_context(ctx, 3000)
    assert "app.py" in out and "read it with the read tool" in out and "BIG BODY" not in out
    assert "b.py" in out and len(out) < 3000
    assert "search with the tools" in leads_from_context("", 100)


def test_line_number_prefix_is_stripped_only_when_every_line_has_it():
    e = _strip_line_numbers(Edit("a.py", " 4| def total(items):\n 5|     return 1", " 4| def total(items):\n 5|     return 2", 1))
    assert e.search == "def total(items):\n    return 1" and e.replace == "def total(items):\n    return 2"
    keep = _strip_line_numbers(Edit("a.py", "x = '1| a'\ny = 2", "z", 1))
    assert keep.search == "x = '1| a'\ny = 2"


# ----------------------------------------------------------------------------- agent loop
def test_investigates_with_tools_then_answers_without_validation():
    async def go():
        store, rid = await make()
        llm = Scripted(act("grep", pattern="def total"), act("read", path="app.py"),
                       "`total` sums prices (app.py:4). `checkout` calls it.")
        sb = FakeSandbox()
        out = await agent_for(llm, sb, store).answer(rid, "SYSTEM", [])
        assert "app.py:4" in out
        assert "grep x1" in out and "read x1" in out and "Files read: app.py" in out
        assert sb.calls == []                                       # an explanation has nothing to validate
        # the model was shown the real tool output, with line numbers
        assert "OBSERVATION (read)" in llm.seen[2][-1] and "4| def total(items):" in llm.seen[2][-1]
        assert llm.seen[0][0].startswith("SYSTEM") and "HOW TO WORK" in llm.seen[0][0]
    run(go())


def test_answer_without_any_tool_is_labelled():
    async def go():
        store, rid = await make()
        out = await agent_for(Scripted("It does things."), store=store).answer(rid, "S", [])
        assert "No search tools were used" in out
    run(go())


def test_tool_budget_forces_an_answer(monkeypatch):
    monkeypatch.setattr(settings, "ask_max_steps", 2)

    async def go():
        store, rid = await make()
        llm = Scripted(act("find", pattern="py"), act("ls", path=""), "Best effort answer.")
        out = await agent_for(llm, store=store).answer(rid, "S", [])
        assert out.startswith("Best effort answer.")
        assert "tool budget is used up" in llm.seen[-1][-1]

        # a model that keeps asking for tools after the budget is gone still gets a clear reply, not a loop
        llm = Scripted(act("find", pattern="py"), act("ls", path=""), act("ls", path="services"))
        out = await agent_for(llm, store=store).answer(rid, "S", [])
        assert out.startswith("I ran out of tool calls")
    run(go())


def test_edit_to_a_file_that_was_not_read_is_sent_back_then_passes():
    fix = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct",
                "def discount(amount, pct):\n    return amount - amount * pct / 100")

    async def go():
        store, rid = await make()
        sb = FakeSandbox()
        llm = Scripted("Root cause: pct is a percentage.\n" + fix,        # 1st answer: never read app.py -> rejected
                       act("read", path="app.py"),                         # model reads the whole file
                       "Root cause: pct is a percentage (app.py:9).\n" + fix)
        out = await agent_for(llm, sb, store).answer(rid, "S", [])
        assert "not read it completely" in llm.seen[1][-1]               # the model was told why
        assert len(sb.calls) == 1 and sb.calls[0][0] == 0 and sb.calls[0][1] == ["app.py"]
        assert "Sandbox check of the proposed edit(s)" in out and "3 passed" in out
        assert "did NOT pass" not in out
    run(go())


def test_search_text_that_does_not_exist_is_sent_back():
    bad = block("app.py", "def discount(amount, rate):\n    return 0", "pass")
    good = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct",
                 "def discount(amount, pct):\n    return amount * (1 - pct)")

    async def go():
        store, rid = await make()
        llm = Scripted(act("read", path="app.py"), bad, good)
        sb = FakeSandbox()
        out = await agent_for(llm, sb, store).answer(rid, "S", [])
        assert "not found" in llm.seen[-1][-1]                           # exact-match failure is reported to the model
        assert len(sb.calls) == 1 and "did NOT pass" not in out
    run(go())


def test_line_numbers_copied_into_search_are_tolerated():
    numbered = block("app.py", " 8| def discount(amount, pct):\n 9|     return amount - amount * pct",
                     " 8| def discount(amount, pct):\n 9|     return amount * (1 - pct)")

    async def go():
        store, rid = await make()
        sb = FakeSandbox()
        out = await agent_for(Scripted(act("read", path="app.py"), numbered), sb, store).answer(rid, "S", [])
        assert len(sb.calls) == 1 and "did NOT pass" not in out
    run(go())


def test_new_failing_test_goes_back_and_unresolved_failures_are_reported(monkeypatch):
    monkeypatch.setattr(settings, "ask_validate_rounds", 1)
    fix = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct", "def discount(amount, pct):\n    return amount")
    fail = CaseResult("tests/test_app.py::test_discount", "failed", "assert 100 == 90", "E assert 100 == 90")
    sb = FakeSandbox(SandboxVerdict(RunReport(mode="pytest", ran=True, passed=1, failed=1, failures=[fail], backend="docker"),
                                    new_failures=[fail]))

    async def go():
        store, rid = await make()
        llm = Scripted(act("read", path="app.py"), fix, fix)               # the model repeats itself after the report
        out = await agent_for(llm, sb, store).answer(rid, "S", [])
        assert "test tests/test_app.py::test_discount fails" in llm.seen[-1][-1]
        assert len(sb.calls) == 2
        assert "did NOT pass" in out and "unverified" in out             # the user is told it is NOT verified
    run(go())


def test_generated_repro_test_is_passed_as_new_test():
    code = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct",
                 "def discount(amount, pct):\n    return amount * (1 - pct)")
    test = ("FILE: tests/test_discount.py\n<<<<<<< SEARCH\n=======\n"
            "from app import discount\n\n\ndef test_discount():\n    assert discount(100, 0.1) == 90\n>>>>>>> REPLACE\n")

    async def go():
        store, rid = await make()
        sb = FakeSandbox()
        out = await agent_for(Scripted(act("read", path="app.py"), code + test), sb, store).answer(rid, "S", [])
        assert sb.calls[0][1] == ["app.py", "tests/test_discount.py"]
        assert sb.calls[0][2] == ["tests/test_discount.py"]              # runs, and is proven against the OLD code by the sandbox
        assert "did NOT pass" not in out
    run(go())


def test_sandbox_not_available_is_reported_not_hidden():
    fix = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct", "def discount(amount, pct):\n    return amount")
    down = SandboxVerdict(RunReport(mode="none", ran=False, note="Docker not reachable", backend="docker"), not_run="docker_down")

    async def go():
        store, rid = await make()
        out = await agent_for(Scripted(act("read", path="app.py"), fix), FakeSandbox(down), store).answer(rid, "S", [])
        assert "Tests not run" in out and "Docker not reachable" in out
        assert "did NOT pass" not in out
    run(go())


def test_edit_that_breaks_syntax_is_caught_before_the_sandbox():
    broken = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct", "def discount(amount, pct):\n    return (")

    async def go():
        store, rid = await make()
        sb = FakeSandbox()
        llm = Scripted(act("read", path="app.py"), broken, "I could not produce a valid edit.")
        await agent_for(llm, sb, store).answer(rid, "S", [])
        assert sb.calls == []                                            # never reached the container
        assert "static check" in llm.seen[-1][-1]
    run(go())


def test_validation_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "ask_validate", False)
    fix = block("app.py", "def discount(amount, pct):\n    return amount - amount * pct", "def discount(amount, pct):\n    return amount")

    async def go():
        store, rid = await make()
        sb = FakeSandbox()
        out = await agent_for(Scripted(fix), sb, store).answer(rid, "S", [])
        assert sb.calls == [] and "Sandbox check" not in out
    run(go())


# ----------------------------------------------------------------------------- wiring
def test_rag_node_uses_agent_and_falls_back_when_it_fails(monkeypatch):
    pytest.importorskip("langchain_chroma")
    from langchain_core.messages import AIMessage, HumanMessage
    from services import rag_graph

    class Mgr(rag_graph.RAGWorkflowManager):
        def __init__(self):                                              # no DB / vectors needed for this node
            self.web = None
            self.code_agent = None

    class Agent:
        def __init__(self, fail=False):
            self.fail, self.got = fail, None

        async def answer(self, repo_id, system, history):
            self.got = (repo_id, system, history)
            if self.fail:
                raise RuntimeError("boom")
            return "agent answer"

    class FakeLLM:
        def __init__(self, *a, **k): pass
        async def ainvoke(self, msgs): return AIMessage(content="single-shot answer")

    monkeypatch.setattr(settings, "ask_agent", True)
    monkeypatch.setattr(rag_graph, "ChatOllama", FakeLLM)
    state = {"repo_id": "r1", "messages": [HumanMessage(content="why?")], "context_str": "--- [PRIMARY_MATCH] File: a.py (Chunk #0) ---\ncode"}

    async def go():
        m = Mgr()
        m.code_agent = Agent()
        out = await m._generate_answer_node(state)
        assert out["messages"][0].content == "agent answer"
        assert "a.py" in m.code_agent.got[1] and m.code_agent.got[0] == "r1"
        m.code_agent = Agent(fail=True)
        assert (await m._generate_answer_node(state))["messages"][0].content == "single-shot answer"
        m.code_agent = None
        assert (await m._generate_answer_node(state))["messages"][0].content == "single-shot answer"
    run(go())


# ----------------------------------------------------------------------------- real sandbox service (local runner)
def test_suggested_fix_runs_through_the_real_sandbox_with_fail_to_pass_proof(monkeypatch):
    """Agent proposes a fix + repro test for a real bug; SandboxService (LocalRunner = same logic as Docker, minus the container)
    runs the repo's test and the new one, and proves the new test fails on the OLD code."""
    from services import sandbox_service as sbx
    monkeypatch.setattr(settings, "sandbox_boot_check", False)       # no FastAPI app in this tiny repo

    files = {
        "shop.py": "def discount(amount, pct):\n    return amount - amount * pct\n",             # BUG: pct is a percentage (10 = 10%)
        "tests/test_shop.py": "from shop import discount\n\n\ndef test_zero():\n    assert discount(50, 0) == 50\n",
    }
    fix = block("shop.py", "def discount(amount, pct):\n    return amount - amount * pct",
                "def discount(amount, pct):\n    return amount - amount * pct / 100")
    repro = ("FILE: tests/test_discount_repro.py\n<<<<<<< SEARCH\n=======\n"
             "from shop import discount\n\n\ndef test_ten_percent():\n    assert discount(200, 10) == 180\n>>>>>>> REPLACE\n")

    async def go():
        store, rid = await make(files)
        sb = sbx.SandboxService(store, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        llm = Scripted(act("grep", pattern="def discount"), act("read", path="shop.py"), "Root cause: shop.py:2 treats pct as a fraction.\n" + fix + repro)
        out = await agent_for(llm, sb, store).answer(rid, "S", [])
        assert "did NOT pass" not in out, out
        assert "Tests: **" in out and "passed" in out                 # tests really ran
        assert "fail->pass" in out                                    # and the repro test fails on the old code, passes on the new
    run(go())


def test_a_wrong_fix_is_caught_by_the_real_sandbox(monkeypatch):
    from services import sandbox_service as sbx
    monkeypatch.setattr(settings, "sandbox_boot_check", False)
    monkeypatch.setattr(settings, "ask_validate_rounds", 0)
    files = {
        "shop.py": "def discount(amount, pct):\n    return amount - amount * pct / 100\n",
        "tests/test_shop.py": "from shop import discount\n\n\ndef test_ten():\n    assert discount(200, 10) == 180\n",
    }
    wrong = block("shop.py", "    return amount - amount * pct / 100", "    return amount")

    async def go():
        store, rid = await make(files)
        sb = sbx.SandboxService(store, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
        out = await agent_for(Scripted(act("read", path="shop.py"), "Drop the discount.\n" + wrong), sb, store).answer(rid, "S", [])
        assert "did NOT pass" in out and "test_ten" in out            # the existing test catches it; the user is told
    run(go())
