"""Subscription runners (claude_code, chatgpt_codex): official CLIs executed through a fake executor, no real CLI, no login, no network.

The pure logic (argv builders, parsers, flattening, error classification, registry) runs without langchain (tests/stubbed_imports.py
stands in for missing packages); the tests that build the LangChain model need langchain_core and are skipped without it.
"""
import asyncio
import importlib.util
import json
import types

import pytest

from stubbed_imports import stubbed_imports

# Decided at import time, before the stub finder is installed (the stubs would make any import succeed).
needs_langchain = pytest.mark.skipif(importlib.util.find_spec("langchain_core") is None, reason="langchain_core not installed")


@pytest.fixture(scope="module")
def m():
    with stubbed_imports():
        import importlib
        yield types.SimpleNamespace(
            core=importlib.import_module("services.runners.cli_core"), rn=importlib.import_module("services.runners"),
            errs=importlib.import_module("services.llm_errors"), base=importlib.import_module("services.runners.base"),
            cc=importlib.import_module("services.runners.claude_code_runner"), cx=importlib.import_module("services.runners.codex_runner"),
            settings=importlib.import_module("config.settings").settings)


@pytest.fixture(autouse=True)
def _cfg(m, monkeypatch):
    for k, v in dict(llm_runner="ollama", embedding_runner="ollama", claude_code_bin="claude", claude_code_model="", claude_code_timeout_s=300.0,
                     claude_code_max_input_tokens=150000, codex_bin="codex", codex_model="", codex_timeout_s=300.0,
                     codex_max_input_tokens=100000, llm_model="gemma4:31b-cloud").items():
        monkeypatch.setattr(m.settings, k, v, raising=False)


CLAUDE_OK = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "Hello there",
                        "total_cost_usd": 0.01, "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
                                                          "cache_creation_input_tokens": 20}})


# ---------------------------------------------------------------- registry / status / config
def test_registry_and_status_listing(m, monkeypatch):
    assert {"claude_code", "chatgpt_codex"} <= set(m.rn.available_runners())
    monkeypatch.setattr(m.settings, "claude_code_bin", "/nonexistent/claude")
    by = {r["name"]: r for r in m.rn.runner_status()}
    assert by["claude_code"]["label"] == "Claude (subscription via Claude Code CLI)"
    assert by["chatgpt_codex"]["label"] == "ChatGPT (subscription via Codex CLI)"
    assert by["claude_code"]["remote"] is True and by["claude_code"]["configured"] is False
    assert (by["claude_code"]["timeout_s"], by["claude_code"]["max_retries"]) == (300.0, 0)
    assert "claude" in by["claude_code"]["error"] and "/login" in by["claude_code"]["error"]


@pytest.mark.parametrize("name,setting,hint", [("claude_code", "claude_code_bin", "/login"), ("chatgpt_codex", "codex_bin", "codex login")])
def test_check_config_missing_binary(m, monkeypatch, name, setting, hint):
    monkeypatch.setattr(m.settings, setting, "/nonexistent/bin")
    with pytest.raises(m.base.RunnerConfigError) as e:
        m.rn.get_runner(name).check_config()
    assert hint in str(e.value)


def test_check_config_ok_with_existing_binary(m, monkeypatch):
    monkeypatch.setattr(m.settings, "codex_bin", "python3")
    m.rn.get_runner("chatgpt_codex").check_config()


def test_runner_attributes(m):
    cc, cx = m.rn.get_runner("claude_code"), m.rn.get_runner("chatgpt_codex")
    assert (cc.max_input_tokens, cx.max_input_tokens) == (150000, 100000)
    assert cc.supports == cx.supports == {"vision": False, "json": True, "reasoning": False, "num_ctx": False, "embeddings": False,
                                           "web_search": True}
    assert cc.remote and cx.remote and cc.max_retries() == 0
    assert asyncio.run(cc.list_models()) == ["sonnet", "opus", "haiku"]
    m.settings.claude_code_model = "claude-sonnet-4-5"
    assert asyncio.run(cc.list_models())[-1] == "claude-sonnet-4-5"
    assert "gpt-5.5" in asyncio.run(cx.list_models())


def test_map_model(m):
    cc, cx = m.rn.get_runner("claude_code"), m.rn.get_runner("chatgpt_codex")
    assert cc.map_model("gemma4:31b-cloud") == "default" and cx.map_model("") == "default"      # "default" = omit the flag
    assert cc.map_model("opus") == "opus" and cc.map_model("claude-opus-4-1") == "claude-opus-4-1"
    assert cx.map_model("gpt-5") == "gpt-5" and cx.map_model("o3") == "o3" and cx.map_model("codex-mini") == "codex-mini"
    m.settings.claude_code_model, m.settings.codex_model = "sonnet", "gpt-5-codex"
    assert cc.map_model("gemma4:31b-cloud") == "sonnet" and cx.map_model("gemma4:31b-cloud") == "gpt-5-codex"
    assert cc.map_model("ollama-thing") == "sonnet"         # "o*" must not swallow ollama names (codex: o\d only)
    assert cx.map_model("ollama-thing") == "gpt-5-codex"


def test_embeddings_refused_and_embedding_runner_fails_clearly(m, monkeypatch):
    for n in ("claude_code", "chatgpt_codex"):
        with pytest.raises(m.base.RunnerConfigError, match="no embeddings; keep EMBEDDING_RUNNER=ollama"):
            m.rn.get_runner(n).embeddings()
    monkeypatch.setattr(m.settings, "embedding_runner", "claude_code")
    with pytest.raises(m.base.RunnerConfigError, match="EMBEDDING_RUNNER=ollama"):
        m.rn.check_runner_config()
    monkeypatch.setattr(m.settings, "embedding_runner", "ollama")
    m.rn.check_runner_config()                               # unaffected for the default setup


# ---------------------------------------------------------------- argv builders
def test_claude_argv(m):
    argv = m.core.build_claude_argv("claude", "", "")
    assert argv[:4] == ["claude", "-p", "--output-format", "json"]
    assert argv[argv.index("--tools") + 1] == "" and "--model" not in argv and "--system-prompt-file" not in argv
    argv = m.core.build_claude_argv("claude", "opus", "/w/system_prompt.txt")
    assert argv[argv.index("--model") + 1] == "opus" and argv[argv.index("--system-prompt-file") + 1] == "/w/system_prompt.txt"
    assert "--model" not in m.core.build_claude_argv("claude", "default", "")


def test_codex_argv(m):
    argv = m.core.build_codex_argv("codex", "", "/tmp/x/last.txt")
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "-" and "-m" not in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only" and "--skip-git-repo-check" in argv and "--json" in argv
    assert argv[argv.index("--output-last-message") + 1] == "/tmp/x/last.txt"
    argv = m.core.build_codex_argv("codex", "gpt-5", "/tmp/x/last.txt")
    assert argv[argv.index("-m") + 1] == "gpt-5" and argv[-1] == "-"


def test_prompt_goes_to_stdin_not_argv(m, tmp_path):
    a = m.core.ClaudeCodeAdapter("claude", "")
    argv, stdin = a.build("multi\nline \"sys\" & | <x>", "SECRET-PROMPT", str(tmp_path))
    assert stdin == "SECRET-PROMPT" and not any("SECRET-PROMPT" in x or "multi" in x for x in argv)
    f = argv[argv.index("--system-prompt-file") + 1]                # system prompt goes through a file, never argv
    assert open(f, encoding="utf-8").read() == "multi\nline \"sys\" & | <x>"
    argv, _ = a.build("", "q", str(tmp_path))
    assert "--system-prompt-file" not in argv
    argv, stdin = m.core.CodexAdapter("codex", "").build("sys", "SECRET-PROMPT", "/tmp/w")
    assert "SECRET-PROMPT" in stdin and "sys" in stdin and not any("SECRET-PROMPT" in x for x in argv)


# ---------------------------------------------------------------- parsers
def test_parse_claude_output(m):
    reply, is_err = m.core.parse_claude_output(CLAUDE_OK)
    assert (reply.text, is_err, reply.output_tokens, reply.cache_read_tokens, reply.cost_usd) == ("Hello there", False, 5, 100, 0.01)
    assert reply.input_tokens == 130 and reply.usage_known
    reply, is_err = m.core.parse_claude_output(json.dumps([{"type": "system"}, json.loads(CLAUDE_OK)]))
    assert reply.text == "Hello there"
    reply, is_err = m.core.parse_claude_output("plain words")        # tolerant
    assert reply.text == "plain words" and not reply.usage_known
    _, is_err = m.core.parse_claude_output(json.dumps({"is_error": True, "result": "x"}))
    assert is_err


def test_parse_codex_events(m):
    lines = ["warning: not json", json.dumps({"type": "thread.started", "thread_id": "t"}), json.dumps({"type": "something.new", "x": 1}),
             json.dumps({"type": "item.completed", "item": {"id": "i", "type": "reasoning", "text": "hm"}}),
             json.dumps({"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "Final answer"}}),
             json.dumps({"type": "turn.completed", "usage": {"input_tokens": 50, "cached_input_tokens": 20, "output_tokens": 7}})]
    text, inp, out, cached, known, err = m.core.parse_codex_events("\n".join(lines))
    assert (text, inp, out, cached, known, err) == ("Final answer", 50, 7, 20, True, "")
    legacy = json.dumps({"id": "1", "msg": {"type": "token_count", "info": {"total_token_usage": {"input_tokens": 3, "output_tokens": 4}}}})
    assert m.core.parse_codex_events(legacy)[1:3] == (3, 4)
    assert m.core.parse_codex_events(json.dumps({"type": "turn.failed", "error": {"message": "boom"}}))[5] == "boom"


def test_codex_adapter_prefers_last_message_file(m, tmp_path):
    (tmp_path / "last_message.txt").write_text("from file\n")
    ev = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "from jsonl"}})
    reply = m.core.CodexAdapter("codex").parse(m.core.ExecResult(0, ev, ""), str(tmp_path))
    assert reply.text == "from file"
    (tmp_path / "last_message.txt").unlink()
    assert m.core.CodexAdapter("codex").parse(m.core.ExecResult(0, ev, ""), str(tmp_path)).text == "from jsonl"


# ---------------------------------------------------------------- flattening / json fences (plain objects, no langchain)
def test_flatten_messages(m):
    O = types.SimpleNamespace
    system, body = m.core.flatten_messages([O(type="system", content="SYS"), O(type="human", content="hi")])
    assert (system, body) == ("SYS", "hi")
    system, body = m.core.flatten_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "q1"},
                                            {"role": "assistant", "content": "a1"}, {"role": "tool", "content": "obs"},
                                            O(type="human", content=[{"type": "text", "text": "q2"},
                                                                     {"type": "image_url", "image_url": {"url": "data:..."}}])])
    assert system == "S" and "[USER]\nq1" in body and "[ASSISTANT]\na1" in body and "[TOOL RESULT]\nobs" in body
    assert "q2" in body and m.core.IMAGE_NOTE in body and "data:" not in body


def test_json_mode_instruction_and_fences(m):
    system, _ = m.core.compose_prompt("S", "q", json_mode=True, max_tokens=100)
    assert "single JSON object" in system and "100 tokens" in system and system.startswith("S")
    assert m.core.strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert m.core.strip_json_fences('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------------------- error classification
def test_failure_classification(m):
    c = m.core
    assert c.failure("Claude Code CLI", 1, "Not logged in · Please run /login").cli_kind == "auth"
    assert c.failure("Codex CLI", 1, "", "Error: please run `codex login`").cli_kind == "auth"
    for txt in ("Claude AI usage limit reached|1700000000", "You hit your 5-hour limit", "rate limit exceeded", "HTTP 429 too many requests"):
        assert c.failure("Claude Code CLI", 1, txt).cli_kind == "quota", txt
    e = c.failure("Codex CLI", 2, "x" * 2000)
    assert e.cli_kind == "other" and len(str(e)) < 600


def test_llm_errors_mapping(m):
    errs, c = m.errs, m.core
    quota = c.failure("Claude Code CLI", 1, "usage limit reached")
    auth = c.failure("Claude Code CLI", 1, "Not logged in")
    with m.rn.runner_scope("claude_code"):
        q = errs.as_quota_error(quota, "sonnet")
        a = errs.as_auth_error(auth, "sonnet")
    assert isinstance(q, errs.ModelQuotaError) and "subscription" in str(q)
    assert isinstance(a, errs.ModelAuthError) and "/login" in str(a)
    assert errs.as_quota_error(auth, "sonnet") is None and errs.as_auth_error(quota, "sonnet") is None
    assert errs.as_quota_error(c.failure("Codex CLI", 1, "segfault"), "x") is None
    assert errs.is_timeout(c.CLITimeoutError("claude did not finish within 5s"))


def test_resolve_command_windows_npm_shim(m, tmp_path, monkeypatch):
    # npm on Windows installs `codex.cmd`; CreateProcess cannot start it by bare name (WinError 2): run `node <entry.js>` instead
    js = tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    js.parent.mkdir(parents=True)
    js.write_text("// entry")
    shim = tmp_path / "codex.cmd"
    shim.write_text('@ECHO off\r\nSET dp0=%~dp0\r\n"%_prog%"  "%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*\r\n')
    node = tmp_path / "node.exe"
    node.write_text("")
    monkeypatch.setattr(m.core.shutil, "which", lambda b: str(shim) if b == "codex" else None)
    cmd = m.core.resolve_command("codex", windows=True)
    assert cmd[0] == str(node) and m.core.os.path.normpath(cmd[1]) == m.core.os.path.normpath(str(js).replace("\\", m.core.os.sep))
    shim.write_text("@echo off\r\nsomething-else %*\r\n")         # unreadable shim: run it by full path
    assert m.core.resolve_command("codex", windows=True) == [str(shim)]
    monkeypatch.setattr(m.core.shutil, "which", lambda b: "/usr/bin/codex")
    assert m.core.resolve_command("codex", windows=False) == ["/usr/bin/codex"]
    monkeypatch.setattr(m.core.shutil, "which", lambda b: None)
    with pytest.raises(m.base.RunnerConfigError):
        m.core.resolve_command("codex", windows=True)


def test_run_subprocess_timeout_and_missing_binary(m):
    with pytest.raises(m.core.CLITimeoutError):
        m.core.run_subprocess(["python3", "-c", "import time; time.sleep(5)"], "", 0.3, "/tmp")
    with pytest.raises(m.base.RunnerConfigError):
        m.core.run_subprocess(["/nonexistent/cli"], "", 1, "/tmp")
    res = m.core.run_subprocess(["python3", "-c", "import sys; print(sys.stdin.read().upper())"], "abc", 10, "/tmp")
    assert res.returncode == 0 and res.stdout.strip() == "ABC"


def test_chat_passes_settings_to_chat_cls_seam(m):
    class Rec:
        def __init__(self, **kw):
            self.kw = kw
    cc = m.rn.get_runner("claude_code")
    got = cc.chat("opus", max_tokens=200, json_mode=True, callbacks=["cb"], chat_cls=Rec)
    assert got.kw["model_name"] == "opus" and got.kw["runner_name"] == "claude_code" and got.kw["json_mode"] is True
    assert got.kw["max_tokens"] == 200 and got.kw["callbacks"] == ["cb"] and got.kw["timeout_s"] == 300.0
    assert got.kw["adapter"].binary == "claude" and got.kw["adapter"].model == "opus"
    got = m.rn.get_runner("chatgpt_codex").chat("gemma4:31b-cloud", chat_cls=Rec)
    assert got.kw["model_name"] == "default" and "callbacks" not in got.kw


# ---------------------------------------------------------------- the LangChain model (needs langchain_core)
class FakeExec:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def __call__(self, argv, stdin, timeout_s, cwd):
        self.calls.append((argv, stdin, timeout_s, cwd))
        if self.exc:
            raise self.exc
        return self.result


def _model(m, adapter, ex, **kw):
    from services.runners.cli_chat import CLIChatModel
    return CLIChatModel(adapter=adapter, executor=ex, model_name="sonnet", runner_name="claude_code", timeout_s=7, **kw)


@needs_langchain
def test_cli_chat_model_claude(m):
    from langchain_core.messages import HumanMessage, SystemMessage
    with stubbed_imports():
        ex = FakeExec(m.core.ExecResult(0, CLAUDE_OK, ""))
        llm = _model(m, m.core.ClaudeCodeAdapter("claude", "sonnet"), ex)
        out = llm.invoke([SystemMessage(content="sys"), HumanMessage(content="question")])
        assert out.content == "Hello there"
        assert out.usage_metadata["input_tokens"] == 130 and out.usage_metadata["output_tokens"] == 5 and out.usage_metadata["total_tokens"] == 135
        assert out.response_metadata == {"model_name": "sonnet", "runner": "claude_code"}
        argv, stdin, timeout, cwd = ex.calls[0]
        assert stdin == "question" and timeout == 7 and "--system-prompt-file" in argv
        assert asyncio.run(llm.ainvoke([HumanMessage(content="again")])).content == "Hello there"


@needs_langchain
def test_cli_chat_model_json_mode_codex_and_errors(m):
    from langchain_core.messages import HumanMessage
    with stubbed_imports():
        fenced = json.dumps({"is_error": False, "result": '```json\n{"ok": true}\n```'})
        llm = _model(m, m.core.ClaudeCodeAdapter("claude"), FakeExec(m.core.ExecResult(0, fenced, "")), json_mode=True)
        assert llm.invoke([HumanMessage(content="q")]).content == '{"ok": true}'
        bad = FakeExec(m.core.ExecResult(1, json.dumps({"is_error": True, "result": "Not logged in · Please run /login"}), ""))
        with pytest.raises(m.core.CLIRunError) as e:
            _model(m, m.core.ClaudeCodeAdapter("claude"), bad).invoke([HumanMessage(content="PROMPT-TEXT")])
        assert e.value.cli_kind == "auth" and "PROMPT-TEXT" not in str(e.value)
        with pytest.raises(m.core.CLITimeoutError):
            _model(m, m.core.CodexAdapter("codex"), FakeExec(exc=m.core.CLITimeoutError("t"))).invoke([HumanMessage(content="q")])
        ev = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 2}})
        res = m.core.ExecResult(0, json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "pong"}}) + "\n" + ev, "")
        out = _model(m, m.core.CodexAdapter("codex"), FakeExec(res)).invoke([HumanMessage(content="ping")])
        assert out.content == "pong" and out.usage_metadata["total_tokens"] == 11


# ---------------------------------------------------------------- refused model -> one retry on the CLI's default
def test_unsupported_model_detection(m):
    assert m.core.is_unsupported_model("Codex CLI failed (exit 1): The 'gpt-5' model is not supported when using Codex with a ChatGPT account.")
    assert not m.core.is_unsupported_model("You've hit your usage limit")
    assert m.core.classify_failure("The 'gpt-5' model is not supported when using Codex with a ChatGPT account.") == "other"


def test_call_cli_retries_without_model_flag_when_model_refused(m):
    calls = []
    refused = '{"type":"error","message":"The \'gpt-5\' model is not supported when using Codex with a ChatGPT account."}'

    def run(argv, stdin, timeout, cwd):
        calls.append(list(argv))
        if "-m" in argv:
            return m.core.ExecResult(1, refused, "")
        with open(m.core.os.path.join(cwd, "last_message.txt"), "w", encoding="utf-8") as fh:
            fh.write("ok from default")
        return m.core.ExecResult(0, "", "")

    reply, used = m.core.call_cli(m.core.CodexAdapter("codex", "gpt-5"), run, "", "q", 5)
    assert reply.text == "ok from default" and used == m.core.DEFAULT_MODEL
    assert "-m" in calls[0] and "-m" not in calls[1] and len(calls) == 2
    with pytest.raises(m.core.CLIRunError):         # no model flag to drop: the refusal is raised as is
        m.core.call_cli(m.core.CodexAdapter("codex", ""), lambda *a: m.core.ExecResult(1, refused, ""), "", "q", 5)
    quota = lambda *a: m.core.ExecResult(1, "", "You've hit your usage limit")
    with pytest.raises(m.core.CLIRunError) as e:    # other failures are not retried
        m.core.call_cli(m.core.CodexAdapter("codex", "gpt-5.5"), quota, "", "q", 5)
    assert e.value.cli_kind == "quota"


def test_model_list_setting_replaces_static_list(m, monkeypatch):
    import asyncio as _a
    from config.settings import settings
    monkeypatch.setattr(settings, "codex_models", " gpt-5.6-sol, gpt-5.5 ,", raising=False)
    monkeypatch.setattr(settings, "codex_model", "", raising=False)
    r = m.rn.get_runner("chatgpt_codex")
    assert _a.run(r.list_models()) == ["gpt-5.6-sol", "gpt-5.5"]
    monkeypatch.setattr(settings, "codex_models", "", raising=False)
    assert "gpt-5" not in _a.run(r.list_models()) and "gpt-5.5" in _a.run(r.list_models())


# ---------------------------------------------------------------- patch 0005: Codex web mode, telemetry label
def test_codex_web_mode_cached_or_live(m, monkeypatch):
    argv = m.core.build_codex_argv("codex", "", "/w/l.txt", web=True, web_mode="cached")
    assert argv[argv.index("-c") + 1] == 'web_search="cached"' and argv[-1] == "-"
    assert 'web_search="live"' in m.core.build_codex_argv("codex", "", "/w/l.txt", web=True, web_mode="bogus")
    assert 'web_search="disabled"' in m.core.build_codex_argv("codex", "", "/w/l.txt", web=False, web_mode="cached")
    monkeypatch.setattr(m.settings, "codex_web_mode", "Cached", raising=False)
    a = m.rn.get_runner("chatgpt_codex").make_adapter("", web=True)
    argv, _ = a.build("", "q", "/w")
    assert 'web_search="cached"' in argv


def test_refused_model_retry_keeps_web_mode(m):
    seen = []

    def run(argv, stdin, timeout, cwd):
        seen.append(list(argv))
        if "-m" in argv:
            return m.core.ExecResult(1, "", "The 'gpt-5' model is not supported when using Codex with a ChatGPT account.")
        with open(m.core.os.path.join(cwd, "last_message.txt"), "w", encoding="utf-8") as fh:
            fh.write("ok")
        return m.core.ExecResult(0, "", "")

    a = m.core.CodexAdapter("codex", "gpt-5", True)
    a.web_mode = "cached"
    reply, used = m.core.call_cli(a, run, "", "q", 5)
    assert reply.text == "ok" and 'web_search="cached"' in seen[1] and "-m" not in seen[1]


def test_telemetry_requested_model_says_default_when_nothing_picked(m, monkeypatch):
    class Cb:                                   # stands in for telemetry.StdoutTelemetryCallback (matched by class name)
        def __init__(self, req):
            self.model_name, self.requested_model = "", req
    Cb.__name__ = "StdoutTelemetryCallback"

    class R:
        name, label, remote, max_input_tokens = "fakecli", "Fake", True, 0
        supports = {"web_search": False}
        def check_config(self): return None
        def map_model(self, model, *, vision=False): return "default"
        def chat(self, model, **kw): return kw
    m.rn.register_runner("fakecli", R)
    monkeypatch.setattr(m.settings, "llm_model", "gemma4:31b-cloud", raising=False)
    with m.rn.runner_scope("fakecli"):
        cb = Cb("gemma4:31b-cloud")
        m.rn.make_chat("gemma4:31b-cloud", callbacks=[cb])
        assert cb.requested_model == "Default" and cb.model_name == "default"
        cb2 = Cb("")
        m.rn.make_chat("gpt-5.5", callbacks=[cb2])
        assert cb2.requested_model == "gpt-5.5"
    m.rn._REGISTRY.pop("fakecli", None)


# ---------------------------------------------------------------- patch 0006: which model really answered
CODEX_HI = (
    '{"type":"thread.started","thread_id":"01a1ca18-5a7f-7751-9ca3-c027fe8576a5"}\n{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hi! What can I help you with?"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":14217,"cached_input_tokens":11648,"cache_write_input_tokens":0,'
    '"output_tokens":13,"reasoning_output_tokens":0}}\n')


def test_claude_actual_model_from_model_usage(m):
    data = {"result": "x", "usage": {"input_tokens": 1, "output_tokens": 2},
            "modelUsage": {"claude-haiku-4-5": {"outputTokens": 3}, "claude-sonnet-4-5": {"outputTokens": 400}}}
    reply, _ = m.core.parse_claude_output(json.dumps(data))
    assert reply.actual_model == "claude-sonnet-4-5"
    reply, _ = m.core.parse_claude_output(json.dumps({"result": "x"}))
    assert reply.actual_model == ""


def test_codex_actual_model_from_session_log_then_config(m, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert m.core.codex_thread_id(CODEX_HI) == "01a1ca18-5a7f-7751-9ca3-c027fe8576a5"
    a = m.core.CodexAdapter("codex", "")
    work = tmp_path / "w"
    work.mkdir()
    assert a.parse(m.core.ExecResult(0, CODEX_HI, ""), str(work)).actual_model == ""      # nothing recorded anywhere
    (tmp_path / "config.toml").write_text('model = "gpt-5.4"\n[profiles.x]\nmodel = "other"\n')
    r = a.parse(m.core.ExecResult(0, CODEX_HI, ""), str(work))
    assert r.actual_model == "gpt-5.4" and r.text.startswith("Hi!") and r.input_tokens == 14217 and r.cache_read_tokens == 11648
    day = tmp_path / "sessions" / "2026" / "11" / "11"
    day.mkdir(parents=True)
    (day / "rollout-2026-11-11T15-43-00-01a1ca18-5a7f-7751-9ca3-c027fe8576a5.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"01a1ca18"}}\n{"type":"response_item","payload":{"text":"hi"}}\n'
        '{"type":"turn_context","payload":{"cwd":"/x","model":"gpt-5.5"}}\n')
    assert a.parse(m.core.ExecResult(0, CODEX_HI, ""), str(work)).actual_model == "gpt-5.5"           # session log wins
    assert m.core.CodexAdapter("codex", "gpt-5.4-mini").parse(m.core.ExecResult(0, CODEX_HI, ""), str(work)).actual_model == "gpt-5.4-mini"
    assert m.core.codex_session_model("../../etc") == ""                                                  # odd ids never reach glob


def test_telemetry_relabels_with_actual_model(m):
    with stubbed_imports():
        from services.telemetry import StdoutTelemetryCallback
        cb = StdoutTelemetryCallback(model_name="default", repo_id="r", phase="RAG_GENERATION", requested_model="Default")
        cb.start_time = __import__("time").perf_counter()
        msg = types.SimpleNamespace(usage_metadata={"input_tokens": 5, "output_tokens": 1},
                                    response_metadata={"model_name": "gpt-5.5", "actual_model": "gpt-5.5"})
        resp = types.SimpleNamespace(generations=[[types.SimpleNamespace(message=msg, generation_info={})]], llm_output={})
        try:
            cb.on_llm_end(resp)
        except Exception:       # emit / ledger side effects are not under test
            pass
        assert cb.model_name == "gpt-5.5" and cb.requested_model == "Default"
