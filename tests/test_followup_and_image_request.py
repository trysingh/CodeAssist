"""Follow-up web searches in one thread + requests written inside uploaded images. No network, no Ollama."""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from config.settings import settings
from models.schemas import GENERAL_REPO_ID
from services import rag_graph
from services.attachment_intent import (derive_request, is_vague, looks_actionable, rules_extract, transcript_of)
from services.rag_graph import RAGWorkflowManager
from services.web_search_service import WebContext, WebSource


def run(coro):
    return asyncio.run(coro)


# ============================================================================= same thread, new search
class _DB:
    async def get_thread_attachments(self, thread_id):
        return []


class _Att:
    async def build_context(self, *a):
        return ""


class _Web:
    """Records every gather() and returns a different 'fresh result' each time."""
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    async def gather(self, question, *, repo_scoped, history=None, avoid=None, now=None):
        self.calls.append({"question": question, "history": history, "avoid": avoid, "repo_scoped": repo_scoped})
        n = len(self.calls)
        if self.fail:
            return WebContext(query=f"q{n}", note="search unavailable (ConnectError)")
        return WebContext(text=f'<web_source n="1" url="https://x.io/{n}" title="T">fresh result {n}</web_source>',
                          sources=[WebSource(1, f"https://x.io/{n}", "T", text="fresh")], query=f"q{n}",
                          date="Thursday, 22 October 2026")


@pytest.fixture
def llm(monkeypatch):
    seen = []

    class FakeChat:
        def __init__(self, **kw):
            pass

        async def ainvoke(self, messages):
            seen.append(messages)
            return AIMessage(content="ok")
    monkeypatch.setattr(rag_graph, "ChatOllama", FakeChat)
    return seen


def _mgr(web):
    return RAGWorkflowManager(_DB(), None, attachment_service=_Att(), web_service=web)


def test_new_request_in_same_thread_runs_a_fresh_search(llm):
    """The reported bug: 'do a new search and try checking again' was answered from the previous answer."""
    web, m = _Web(), None

    async def go():
        m = _mgr(web)
        await m.invoke_chat(GENERAL_REPO_ID, "t1", "is there an india vs new zealand T20 match today?")
        out = await m.invoke_chat(GENERAL_REPO_ID, "t1", "do a new search and try checking again")
        await m.invoke_chat(GENERAL_REPO_ID, "t1", "what about tomorrow?")
        return out
    out = run(go())

    assert len(web.calls) == 3                                           # every one of the three messages searched
    second = web.calls[1]
    assert second["question"] == "do a new search and try checking again"
    assert any("india vs new zealand" in t for _, t in second["history"])     # query can be built from the first question
    assert second["avoid"] == ["q1"]                                     # ...and must differ from the one just run
    assert web.calls[2]["avoid"] == ["q2"]
    sys2 = llm[1][0].content
    assert "fresh result 2" in sys2 and "fresh result 1" not in sys2     # turn 2 sees ONLY its own results
    assert "CURRENT DATE AND TIME" in sys2 and "query: q2" in sys2
    assert "Web sources" in out and "*Searched: q2" in out               # the reply shows THIS turn's own search


def test_no_search_claim_when_nothing_was_attached(llm):
    web = _Web(fail=True)

    async def go():
        m = _mgr(web)
        return await m.invoke_chat(GENERAL_REPO_ID, "t2", "do a new search")
    out = run(go())
    sys1 = llm[0][0].content
    assert "Do not claim you cannot search the web" in sys1             # the sentence the bug report quoted is ruled out
    assert "search problem: search unavailable (ConnectError)" in sys1
    assert "answered without it" in out                                  # user is told the search failed, not that none exists


def test_web_chip_off_is_explained_to_the_model(llm):
    web = _Web()

    async def go():
        await _mgr(web).invoke_chat(GENERAL_REPO_ID, "t3", "do a new search", web="off")
    run(go())
    assert web.calls == [] and "turned off for this chat" in llm[0][0].content


# ============================================================================= image -> request
CHAT_SHOT = ("can u confirm if there is a T20 match scheduled today between them\n"
             "I cannot confirm if there is a T20 match between India and New Zealand today.\n"
             "do a new search and try checking again\n"
             "I do not have the ability to perform a new live web search.\n"
             "Description: dark-theme chat UI with two user bubbles and two bot replies")
TRACE = "Traceback (most recent call last):\n  File \"app.py\", line 3, in <module>\nKeyError: 'repo_id'\nDescription: terminal"


def _img(text, name="shot.png", kind="image"):
    return {"filename": name, "kind": kind, "text": text}


def _noul(p):
    async def f(*a, **k):
        return p
    return f


async def _boom(*a, **k):
    raise AssertionError("must not be called")


def _ext(reply):
    async def f(text, typed):
        f.seen = (text, typed)
        return reply
    return f


def test_vague_detection():
    for t in ("", "see this", "please review the attached content", "fix this", "do what the screenshot says", "ok"):
        assert is_vague(t), t
    for t in ("what does this function do in services/repo_store.py?", "summarize the T20 schedule for the week please",
              "latest fastapi", "why is the sandbox failing"):
        assert not is_vague(t), t


def test_image_only_message_becomes_the_request():
    ext = _ext("Confirm whether a T20 match between India and New Zealand is scheduled today, using a new web search.")
    found = run(derive_request("", [_img(CHAT_SHOT)], noul_fn=_noul(0.9), extract_fn=ext, model="m"))
    assert found.source == "image" and "India and New Zealand" in found.request
    assert "Description:" in ext.seen[0] and ext.seen[1] == ""           # the model sees the whole OCR, incl. the description
    assert "`shot.png`" in found.note() and "Send the request as text" in found.note()


def test_typed_message_that_stands_alone_wins():
    found = run(derive_request("explain how decide_web works", [_img(CHAT_SHOT)], noul_fn=_boom, extract_fn=_boom, model="m"))
    assert found.source == "typed" and found.request == "explain how decide_web works" and found.note() == ""


def test_vague_typed_text_is_kept_and_prefixes_survive():
    ext = _ext("Explain and fix this error: KeyError: 'repo_id'")
    found = run(derive_request("fix this", [_img(TRACE)], noul_fn=_boom, extract_fn=ext, model="m"))   # explicit: no gate
    assert found.request == "fix this: Explain and fix this error: KeyError: 'repo_id'"
    found = run(derive_request("/fix", [_img(TRACE)], noul_fn=_noul(0.8), extract_fn=ext, model="m"))
    assert found.request.startswith("/fix Explain and fix this error")


def test_image_without_a_request_changes_nothing():
    plain = "Q3 revenue\nNorth 12\nSouth 9\nDescription: bar chart"
    assert run(derive_request("", [_img(plain)], noul_fn=_noul(0.1), extract_fn=_boom, model="m")).source == "typed"
    # gate says yes but the model finds nothing -> still unchanged
    assert run(derive_request("", [_img(plain)], noul_fn=_noul(0.9), extract_fn=_ext("NONE"), model="m")).source == "typed"


def test_only_images_are_read_and_the_switch_works(monkeypatch):
    assert run(derive_request("", [_img(CHAT_SHOT, "a.pdf", "pdf")], noul_fn=_boom, extract_fn=_boom, model="m")).source == "typed"
    assert run(derive_request("", [_img("[No text could be extracted from this file.]")], noul_fn=_boom,
                              extract_fn=_boom, model="m")).source == "typed"
    monkeypatch.setattr(settings, "attachment_request_enabled", False)
    assert run(derive_request("", [_img(CHAT_SHOT)], noul_fn=_boom, extract_fn=_boom, model="m")).source == "typed"


def test_models_down_falls_back_to_rules():
    async def down(text, typed):
        raise RuntimeError("ollama down")
    found = run(derive_request("", [_img(TRACE)], noul_fn=_noul(None), extract_fn=down, model="m"))
    assert found.source == "image" and found.request == "Explain and fix this error: KeyError: 'repo_id'"
    assert found.how == "rules" and found.method == "rules"
    assert looks_actionable(transcript_of(CHAT_SHOT)) and not looks_actionable("Q3 revenue\nNorth 12")
    # speakers are not labelled in OCR text: the assistant's lines are skipped, and a topic-less last line is joined to the one before
    assert rules_extract(transcript_of(CHAT_SHOT)) == ("can u confirm if there is a T20 match scheduled today between them; "
                                                       "do a new search and try checking again")
    assert "Description" not in transcript_of(CHAT_SHOT)


def test_image_request_is_routed_like_typed_text(monkeypatch):
    """chat_with_repo: the request read from the image is what gets routed, and the answer says so."""
    try:
        from controllers.engine_controller import EngineController
    except Exception as e:  # noqa: BLE001 - e.g. tiktoken cannot download its vocabulary offline
        pytest.skip(f"controller not importable here: {type(e).__name__}")
    from services import attachment_intent as ai

    async def fake_extract(text, typed):
        return "Is a T20 match between India and New Zealand scheduled today?"
    routed = {}

    class DB:
        async def init_db(self): pass
        async def get_repo(self, rid): return object()

    async def route(self, repo_id, thread_id, query, web):
        routed["query"] = query
        return "answer"
    monkeypatch.setattr(ai.derive_request, "__kwdefaults__", {**ai.derive_request.__kwdefaults__, "extract_fn": fake_extract})
    monkeypatch.setattr(EngineController, "_route_chat", route)
    monkeypatch.setattr(settings, "router_model", "")
    ctrl = EngineController.__new__(EngineController)
    ctrl.db = DB()
    out = run(ctrl.chat_with_repo(GENERAL_REPO_ID, "t", "", attachments=[_img(CHAT_SHOT)]))
    assert routed["query"].startswith("Is a T20 match") and out.endswith("answer") and "Request read from your image" in out
    out = run(ctrl.chat_with_repo(GENERAL_REPO_ID, "t", "", attachments=[_img("Q3 revenue\nNorth 12")]))
    assert routed["query"] == "Please review the attached content." and out == "answer"
