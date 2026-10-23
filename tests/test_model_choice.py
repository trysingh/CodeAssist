import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from config.settings import settings
from services import rag_graph
from services.model_calls import model_scope, reasoning_model

def test_chat_uses_picked_model_else_default(monkeypatch):
    seen = []
    class FakeChat:
        def __init__(self, **kw): seen.append(kw["model"])
        async def ainvoke(self, messages): return AIMessage(content="ok")
    monkeypatch.setattr(rag_graph, "ChatOllama", FakeChat)
    m = rag_graph.RAGWorkflowManager(None, None, attachment_service=object(), web_service=object())
    state = {"messages": [HumanMessage(content="hi")], "repo_id": "__general__"}
    async def go():
        await m._generate_answer_node(state)
        with model_scope("qwen3:32b"):
            await m._generate_answer_node(state)
    asyncio.run(go())
    assert seen == [settings.llm_model, "qwen3:32b"]

def test_patcher_default_and_override(monkeypatch):
    monkeypatch.setattr(settings, "patcher_model", "patcher-x")
    assert reasoning_model(settings.patcher_model) == "patcher-x"
    with model_scope("qwen3:32b"):
        assert reasoning_model(settings.patcher_model) == "qwen3:32b"
    with model_scope(""):
        assert reasoning_model(settings.patcher_model) == "patcher-x"