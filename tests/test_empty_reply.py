"""Ask mode must never answer with a blank message.

Seen in the chat UI: the answer was ONLY the footer line "No search tools were used ...". The model had returned an empty
`content` (a thinking model that spent its whole output budget thinking; langchain-ollama drops `thinking` unless reasoning=True),
and CodeAgent turned "" into a final answer. No Ollama, Docker or network here: the model is scripted.
"""
import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from services import model_calls
from services.code_agent import (CodeAgent, EmptyModelReply, ModelText, clean_reply, describe_meta, reply_meta)
from services.repo_store import RepoStore
from services.sandbox_service import RunReport, SandboxVerdict

run = asyncio.run

FILES = {"app.py": "def total(items):\n    return sum(items)\n\n\ndef discount(a, p):\n    return a - a * p\n", "README.md": "# demo\n"}


class Scripted:
    """Stands in for the chat model; each reply is a str or a ModelText (reply + server details)."""
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    async def __call__(self, messages, repo_id=""):
        self.seen.append([str(m.content) for m in messages])
        return self.replies.pop(0)


class FakeSandbox:
    async def verify(self, repo_id, base_rev, working, new_tests=()):
        return SandboxVerdict(RunReport(mode="pytest", ran=True, passed=1, backend="docker", strategy="affected", selected=[]))


async def make():
    store, rid = RepoStore(), uuid.uuid4().hex[:12]
    await store.upsert_base_files(rid, [(p, t.encode()) for p, t in FILES.items()])
    return store, rid


def agent_for(llm, store):
    return CodeAgent(store, FakeSandbox(), llm=llm, emit=lambda *a, **k: None)


# ----------------------------------------------------------------------------- helpers
def test_clean_reply_removes_only_a_leading_think_block():
    assert clean_reply("<think>hmm</think>\n\nThe answer.") == "The answer."
    assert clean_reply("<thinking>a</thinking><think>b</think> Done.") == "Done."
    assert clean_reply("<|channel>thought\nstep 1<channel|>\nFinal.") == "Final."
    assert clean_reply("  plain answer \n") == "plain answer"
    assert clean_reply(None) == "" and clean_reply("   ") == ""
    # opened and never closed: the output budget ended INSIDE the thinking, nothing was answered
    assert clean_reply("<think>still thinking about the zip feature") == ""
    # an answer that merely talks about the tag is left alone
    talk = "Strip the `</think>` marker with a regex, e.g. re.sub(r'<think>.*?</think>', '', s)."
    assert clean_reply(talk) == talk


def test_reply_meta_and_describe():
    resp = AIMessage(content="", response_metadata={"model": "m", "done_reason": "length", "eval_count": 8192, "prompt_eval_count": 900},
                     additional_kwargs={"reasoning_content": "x" * 50})
    meta = reply_meta(resp)
    assert meta == {"model": "m", "done_reason": "length", "eval_count": 8192, "prompt_eval_count": 900, "thinking_chars": 50}
    d = describe_meta(meta)
    assert "done_reason=length" in d and "8192 tokens generated" in d and "50 chars of thinking" in d
    assert describe_meta({}) == "no details from the server"
    assert reply_meta(AIMessage(content="hi")) == {}


def test_reasoning_arg_mapping():
    f = model_calls.reasoning_arg
    assert f("") is None and f("  ") is None and f("whatever") is None
    assert f("off") is False and f("FALSE") is False and f("0") is False
    assert f("on") is True and f("true") is True
    assert f("low") == "low" and f("High") == "high"


# ----------------------------------------------------------------------------- agent
def test_empty_reply_is_asked_again_once_and_the_answer_is_not_blank():
    async def go():
        store, rid = await make()
        llm = Scripted(ModelText("", {"done_reason": "stop", "eval_count": 1}), "Add zip_tool.py next to engine.py.")
        out = await agent_for(llm, store).answer(rid, "SYS", [HumanMessage(content="add a zip feature?")])
        assert out.startswith("Add zip_tool.py next to engine.py.")
        assert len(llm.seen) == 2
        assert "Your last reply was empty" in llm.seen[1][-1]               # the model was told, not just re-sent the same prompt
        assert llm.seen[1][-2] == "(empty reply)"                           # roles keep alternating: user, assistant, user
    run(go())


def test_whitespace_and_unclosed_think_count_as_empty():
    async def go():
        store, rid = await make()
        llm = Scripted("   \n", "<think>never finished", "Real answer.")
        with pytest.raises(EmptyModelReply):
            await agent_for(llm, store).answer(rid, "SYS", [])
        assert len(llm.seen) == 2                                           # one retry only, no endless loop
        llm = Scripted("<think>plan</think>", "Real answer.")
        assert (await agent_for(llm, store).answer(rid, "SYS", [])).startswith("Real answer.")
    run(go())


def test_empty_twice_raises_with_the_server_details():
    async def go():
        store, rid = await make()
        llm = Scripted(ModelText("", {"done_reason": "length", "eval_count": 8192}), ModelText("", {"done_reason": "stop", "eval_count": 1}))
        with pytest.raises(EmptyModelReply) as ei:
            await agent_for(llm, store).answer(rid, "SYS", [])
        msg = str(ei.value)
        assert "done_reason=length" in msg and "8192 tokens generated" in msg and "done_reason=stop" in msg
    run(go())


def test_think_block_never_leaks_and_an_action_inside_it_is_not_run():
    async def go():
        store, rid = await make()
        llm = Scripted('<think>maybe ACTION: {"tool": "read", "args": {"path": "app.py"}}</think>The answer, no tools needed.')
        out = await agent_for(llm, store).answer(rid, "SYS", [])
        assert out.startswith("The answer, no tools needed.")
        assert "<think>" not in out and "read x1" not in out and "No search tools were used" in out
    run(go())


def test_empty_after_a_tool_call_and_empty_at_the_budget_are_both_covered(monkeypatch):
    monkeypatch.setattr(settings, "ask_max_steps", 1)

    async def go():
        store, rid = await make()
        act = 'ACTION: {"tool": "ls", "args": {"path": ""}}'
        # budget used up -> forced final answer is empty -> retried -> real answer
        llm = Scripted(act, "", "Best effort answer.")
        out = await agent_for(llm, store).answer(rid, "SYS", [])
        assert out.startswith("Best effort answer.") and "ls x1" in out
    run(go())


def test_default_model_call_retries_without_thinking_when_the_budget_ran_out(monkeypatch):
    """The real _ollama path: first call as configured, retry sends reasoning=False only after done_reason == length."""
    calls = []

    class FakeChat:
        def __init__(self, reasoning):
            self.reasoning = reasoning

        async def ainvoke(self, messages):
            calls.append(self.reasoning)
            if len(calls) == 1:     # a thinking model: whole budget spent thinking, content empty, thinking dropped by langchain-ollama
                return AIMessage(content="", response_metadata={"model": "gemma4:31b-cloud", "done_reason": "length", "eval_count": 8192})
            return AIMessage(content="Here is the answer.", response_metadata={"done_reason": "stop"})

    def fake_build(model, **kw):
        return FakeChat(kw.get("reasoning"))

    monkeypatch.setattr(model_calls, "_build", fake_build)
    monkeypatch.setattr(settings, "ask_reasoning", "")

    async def go():
        store, rid = await make()
        out = await CodeAgent(store, FakeSandbox(), emit=lambda *a, **k: None).answer(rid, "SYS", [HumanMessage(content="q")])
        assert out.startswith("Here is the answer.")
    run(go())
    assert calls == [None, False]

    calls.clear()               # an explicit setting is honoured on every call, and an empty "stop" does not switch thinking off
    monkeypatch.setattr(settings, "ask_reasoning", "on")

    class StopEmpty(FakeChat):
        async def ainvoke(self, messages):
            calls.append(self.reasoning)
            return AIMessage(content="" if len(calls) == 1 else "ok", response_metadata={"done_reason": "stop", "eval_count": 1})

    monkeypatch.setattr(model_calls, "_build", lambda model, **kw: StopEmpty(kw.get("reasoning")))

    async def go2():
        store, rid = await make()
        out = await CodeAgent(store, FakeSandbox(), emit=lambda *a, **k: None).answer(rid, "SYS", [HumanMessage(content="q")])
        assert out.startswith("ok")
    run(go2())
    assert calls == [True, True]


def test_a_model_that_goes_silent_while_fixing_does_not_wipe_the_first_answer(monkeypatch):
    monkeypatch.setattr(settings, "ask_validate", True)
    monkeypatch.setattr(settings, "ask_validate_rounds", 1)
    fix = ("FILE: app.py\n<<<<<<< SEARCH\ndef discount(a, p):\n    return a - a * p\n=======\n"
           "def discount(a, p):\n    return a * (1 - p)\n>>>>>>> REPLACE\n")

    async def go():
        store, rid = await make()
        llm = Scripted("Root cause: p.\n" + fix, "", "")          # app.py never read -> rejected -> model goes silent twice
        out = await agent_for(llm, store).answer(rid, "SYS", [])
        assert "return a * (1 - p)" in out                         # the earlier proposal is still shown
        assert "stopped replying" in out and "did NOT pass" in out  # ... clearly marked as unverified
    run(go())


# ----------------------------------------------------------------------------- graph node (what the chat UI calls)
def _node_parts(monkeypatch):
    pytest.importorskip("langchain_chroma")
    from services import rag_graph

    class Mgr(rag_graph.RAGWorkflowManager):
        def __init__(self):
            self.web = None
            self.code_agent = None

    monkeypatch.setattr(settings, "ask_agent", True)
    state = {"repo_id": "r1", "messages": [HumanMessage(content="can we add a zip feature?")], "context_str": ""}
    return rag_graph, Mgr, state


def test_chat_never_shows_a_blank_answer_agent_gives_up_then_single_shot_is_empty_too(monkeypatch):
    rag_graph, Mgr, state = _node_parts(monkeypatch)

    class Agent:
        async def answer(self, repo_id, system, history):
            raise EmptyModelReply("empty reply twice. 1st: done_reason=length, 8192 tokens generated")

    class EmptyLLM:
        def __init__(self, *a, **k): pass
        async def ainvoke(self, msgs): return AIMessage(content="", response_metadata={"done_reason": "length", "eval_count": 8192})

    monkeypatch.setattr(rag_graph, "ChatOllama", EmptyLLM)

    async def go():
        m = Mgr()
        m.code_agent = Agent()
        msg = (await m._generate_answer_node(state))["messages"][0]
        text = str(msg.content)
        assert text.strip() and "empty reply" in text and "ENGINE_ASK_REASONING=off" in text
        assert "done_reason=length" in text
    run(go())


def test_agent_gives_up_then_single_shot_answers_and_says_so(monkeypatch):
    rag_graph, Mgr, state = _node_parts(monkeypatch)

    class Agent:
        async def answer(self, repo_id, system, history):
            raise EmptyModelReply("empty reply twice")

    class LLM:
        def __init__(self, *a, **k): pass
        async def ainvoke(self, msgs): return AIMessage(content="<think>x</think>Yes, add zip_tool.py.")

    monkeypatch.setattr(rag_graph, "ChatOllama", LLM)

    async def go():
        m = Mgr()
        m.code_agent = Agent()
        text = str((await m._generate_answer_node(state))["messages"][0].content)
        assert text.startswith("Yes, add zip_tool.py.") and "<think>" not in text
        assert "nothing was searched" in text and "empty reply twice" in text
    run(go())


def test_single_shot_reasoning_flag_is_passed_only_when_configured(monkeypatch):
    rag_graph, Mgr, state = _node_parts(monkeypatch)
    seen = []

    class LLM:
        def __init__(self, *a, **k): seen.append(k)
        async def ainvoke(self, msgs): return AIMessage(content="fine")

    monkeypatch.setattr(rag_graph, "ChatOllama", LLM)

    async def go():
        monkeypatch.setattr(settings, "ask_reasoning", "")
        await Mgr()._generate_answer_node(state)
        monkeypatch.setattr(settings, "ask_reasoning", "off")
        await Mgr()._generate_answer_node(state)
    run(go())
    assert "reasoning" not in seen[0] and seen[1]["reasoning"] is False
