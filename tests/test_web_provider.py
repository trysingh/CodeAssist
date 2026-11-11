"""Runner-aware web search: choose_web_provider matrix, CLI argv for native web, make_chat forwarding, supports / status, and the rag_graph
web node on the native path. No real CLI, no network; pure logic runs without langchain (stubbed_imports)."""
import asyncio
import importlib
import json
import types
from pathlib import Path

import pytest

from stubbed_imports import stubbed_imports


@pytest.fixture(scope="module")
def m():
    with stubbed_imports():
        yield types.SimpleNamespace(
            wp=importlib.import_module("services.web_provider"), core=importlib.import_module("services.runners.cli_core"),
            rn=importlib.import_module("services.runners"), settings=importlib.import_module("config.settings").settings)


@pytest.fixture(autouse=True)
def _cfg(m, monkeypatch):
    for k, v in dict(llm_runner="ollama", embedding_runner="ollama", claude_code_bin="claude", claude_code_model="", codex_bin="codex",
                     codex_model="", llm_model="gemma4:31b-cloud", web_provider="auto").items():
        monkeypatch.setattr(m.settings, k, v, raising=False)


# ---------------------------------------------------------------- the pure decision
@pytest.mark.parametrize("setting,supports,scoped,want", [
    ("app", True, False, "app"), ("app", True, True, "app"), ("app", False, False, "app"), ("app", False, True, "app"),
    ("native", True, False, "native"), ("native", True, True, "native"), ("native", False, False, "app"), ("native", False, True, "app"),
    ("auto", True, False, "native"), ("auto", True, True, "app"), ("auto", False, False, "app"), ("auto", False, True, "app"),
    ("bogus", True, False, "native"), ("bogus", True, True, "app"), ("", True, False, "native"), (None, True, True, "app"),
    (" NATIVE ", True, True, "native"),
])
def test_choose_web_provider_matrix(m, setting, supports, scoped, want):
    assert m.wp.choose_web_provider(setting, supports, scoped) == want


def test_web_provider_setting_declared(m):
    src = (Path(__file__).resolve().parent.parent / "config" / "settings.py").read_text(encoding="utf-8")
    assert 'web_provider: str = Field(default="auto", validation_alias=AliasChoices("ENGINE_WEB_PROVIDER", "WEB_PROVIDER"))' in src


# ---------------------------------------------------------------- argv
def test_claude_argv_web(m):
    off = m.core.build_claude_argv("claude", "", "")
    assert off[off.index("--tools") + 1] == "" and "--allowedTools" not in off
    on = m.core.build_claude_argv("claude", "opus", "/w/s.txt", web=True)
    assert on[on.index("--tools") + 1] == "WebSearch,WebFetch" and on[on.index("--allowedTools") + 1] == "WebSearch,WebFetch"
    assert not {"Bash", "Edit", "Read"} & set(",".join(on).split(","))
    assert on[on.index("--model") + 1] == "opus" and on[on.index("--system-prompt-file") + 1] == "/w/s.txt"


def test_codex_argv_web(m):
    off = m.core.build_codex_argv("codex", "", "/t/l.txt")
    assert off[off.index("-c") + 1] == 'web_search="disabled"' and off[-1] == "-" and "read-only" in off
    on = m.core.build_codex_argv("codex", "gpt-5", "/t/l.txt", web=True)
    assert on[on.index("-c") + 1] == 'web_search="live"' and on[-1] == "-" and on[on.index("-m") + 1] == "gpt-5"


def test_adapters_carry_web(m, tmp_path):
    ca = m.core.ClaudeCodeAdapter("claude", "", web=True)
    argv, _ = ca.build("sys", "q", str(tmp_path))
    assert "--allowedTools" in argv
    argv, _ = m.core.ClaudeCodeAdapter("claude", "").build("sys", "q", str(tmp_path))
    assert "--allowedTools" not in argv
    argv, body = m.core.CodexAdapter("codex", "", web=True).build("sys", "q", str(tmp_path))
    assert 'web_search="live"' in argv and "may use web search" in body and "must cite" in body and "do not run commands" in body
    argv, body = m.core.CodexAdapter("codex", "").build("sys", "q", str(tmp_path))
    assert 'web_search="disabled"' in argv and body.startswith(m.core.CodexAdapter.PREAMBLE)


def test_codex_parser_ignores_web_search_events(m):
    lines = [{"type": "item.started", "item": {"id": "i0", "type": "web_search", "query": "x", "text": "NOT THE REPLY"}},
             {"type": "item.completed", "item": {"id": "i0", "type": "web_search", "query": "x", "text": "NOT THE REPLY"}},
             {"type": "web_search_end", "call_id": "c", "query": "x"},
             {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "Answer https://e.com"}},
             {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}}]
    text, inp, out, _c, known, err = m.core.parse_codex_events("\n".join(json.dumps(x) for x in lines))
    assert text == "Answer https://e.com" and (inp, out, known, err) == (3, 2, True, "")


def test_refused_model_retry_keeps_web(m):
    seen = []

    def run(argv, stdin, timeout_s, cwd):
        seen.append(argv)
        if len(seen) == 1:
            return m.core.ExecResult(1, "", "The 'gpt-5' model is not supported")
        return m.core.ExecResult(0, "", "")

    class A(m.core.Adapter):
        def build(self, system, prompt, workdir):
            return m.core.build_codex_argv("codex", self.model, "", self.web), prompt

        def parse(self, res, workdir):
            if res.returncode:
                raise m.core.failure("x", res.returncode, res.stderr)
            return m.core.CLIReply(text="ok")
    _reply, used = m.core.call_cli(A("codex", "gpt-5", web=True), run, "", "q", 5)
    assert used == "default" and len(seen) == 2 and all('web_search="live"' in a for a in seen)


# ---------------------------------------------------------------- capability, status, make_chat
def test_supports_and_status(m):
    for name in m.rn.available_runners():
        assert "web_search" in m.rn.get_runner(name).supports, name
    for name in ("ollama", "cloudflare", "aistudio", "openai"):
        assert m.rn.get_runner(name).supports["web_search"] is False
    for name in ("claude_code", "chatgpt_codex"):
        assert m.rn.get_runner(name).supports["web_search"] is True
    by = {r["name"]: r for r in m.rn.runner_status()}
    assert all(isinstance(r["native_web"], bool) for r in by.values())
    assert by["claude_code"]["native_web"] and by["chatgpt_codex"]["native_web"] and not by["ollama"]["native_web"]


class Rec:
    def __init__(self, **kw):
        self.kw = kw


def test_make_chat_forwards_web_only_to_runners_that_search(m, monkeypatch):
    seen = {}

    class NoWeb:
        name, label, remote, max_input_tokens = "noweb", "No web", False, 0
        supports = {"web_search": False}

        def map_model(self, model, *, vision=False):
            return model

        def chat(self, model, *, temperature=0, callbacks=None, chat_cls=None):      # no `web` parameter: must not receive it
            seen["noweb"] = (model, temperature)
            return "noweb-llm"

    m.rn.register_runner("noweb", NoWeb)
    try:
        with m.rn.runner_scope("noweb"):
            assert m.rn.make_chat("x", callbacks=[], web=True) == "noweb-llm"
    finally:
        m.rn._REGISTRY.pop("noweb", None)

    for name, flag in (("claude_code", "--allowedTools"), ("chatgpt_codex", 'web_search="live"')):
        with m.rn.runner_scope(name):
            on = m.rn.make_chat("", callbacks=["cb"], chat_cls=Rec, web=True).kw
            off = m.rn.make_chat("", callbacks=["cb"], chat_cls=Rec).kw
        assert on["web"] is True and off["web"] is False
        assert on["adapter"].web is True and off["adapter"].web is False
        argv, _ = on["adapter"].build("", "q", "/tmp")
        assert flag in argv or flag in " ".join(argv)
        argv_off, _ = off["adapter"].build("", "q", "/tmp")
        assert flag not in argv_off


# ---------------------------------------------------------------- rag_graph web node (native path)
@pytest.fixture
def rg(m):
    return importlib.import_module("services.rag_graph")      # `m` keeps the langchain stand-ins installed, so this runs without langchain


def _node(rg, scoped):
    class Msg:
        content = "what is the latest python release"

    class Web:
        called = False

        async def gather(self, *a, **k):
            Web.called = True
            raise AssertionError("gather must not run on the native path")

    mgr = object.__new__(rg.RAGWorkflowManager)
    mgr.web, mgr.code_agent = Web(), None
    state = {"messages": [Msg()], "repo_id": "some/repo" if scoped else rg.GENERAL_REPO_ID, "web_mode": "on"}
    return mgr, state, Web


def test_web_node_native_path_skips_gather(m, rg, monkeypatch):
    mgr, state, Web = _node(rg, scoped=False)
    with m.rn.runner_scope("claude_code"):
        out = asyncio.run(mgr._web_context_node(state))
    assert out["web_native"] is True and out["web_context"] == "" and out["web_note"] == "native" and not Web.called
    assert "native search" in out["web_footer"] and "Claude" in out["web_footer"] and out["web_query"].startswith("what is")


def test_web_node_repo_scoped_auto_keeps_app_path(m, rg):
    mgr, state, Web = _node(rg, scoped=True)
    with m.rn.runner_scope("claude_code"):
        out = asyncio.run(mgr._web_context_node(state))
    assert Web.called and out["web_native"] is False      # gather raised -> the node's own fallback, never an exception
