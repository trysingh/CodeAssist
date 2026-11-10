"""Token mode (services/token_budget.py), lean ask agent, input cap, Cloudflare session affinity, token telemetry and footer.
Fakes only: no network, no Ollama, no langchain_openai."""
import asyncio
import json
import re
import types
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import settings
from services import model_calls
from services.code_agent import PROTOCOL, CodeAgent, RepoTools, compact_observations, observation_stub
from services.reasoning.baseline import guidance_block
from services.repo_store import RepoStore
from services.runners import (RunnerConfigError, current_runner_name, get_embedding_runner, get_runner, make_chat,
                              register_runner, require_runner, resolve_model, runner_scope, runner_status, session_key, thread_scope)
from services.runners import _REGISTRY
from services.runner_info import check_chat_choice, models_payload, runners_payload
from services.telemetry import StdoutTelemetryCallback
from services.token_budget import (budget, effective_mode, estimate_tokens, footer_text, is_lean, ledger_scope, max_input_tokens,
                                   token_mode_scope, trim_history)

run = asyncio.run
CF_DEFAULT = "@cf/google/gemma-4-26b-a4b-it"


def big(tag, lines=60):
    return "\n".join(f"{tag} line {i}: " + "x" * 40 for i in range(1, lines + 1)) + "\n"


FILES = {"a.py": big("a"), "b.py": big("b"), "c.py": big("c"), "d.py": big("d"), "e.py": big("e"), "README.md": "# demo\n"}


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    for k, v in dict(llm_runner="ollama", embedding_runner="ollama", token_mode="auto", max_input_tokens=None, show_token_footer=True,
                     ask_validate=False, cf_account_id="acc", cf_api_token="tok", cf_ai_gateway_url="", cf_session_affinity=True,
                     cf_max_input_tokens=60000, cf_llm_model=CF_DEFAULT, cf_list_all_models=False, cf_models=CF_DEFAULT,
                     lean_keep_full_obs=2, lean_history_turns=6).items():
        monkeypatch.setattr(settings, k, v)
    yield
    _REGISTRY.pop("fake", None)
    _REGISTRY.pop("fakeremote", None)


class Scripted:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    async def __call__(self, messages, repo_id=""):
        self.seen.append([(type(m).__name__, str(m.content)) for m in messages])
        return self.replies.pop(0)


def act(tool, **args):
    return "ACTION: " + json.dumps({"tool": tool, "args": args})


async def ask(llm, history=None, system="SYS", files=FILES):
    store, rid = RepoStore(), uuid.uuid4().hex[:12]
    await store.upsert_base_files(rid, [(p, t.encode()) for p, t in files.items()])
    events = []
    agent = CodeAgent(store, None, llm=llm, emit=lambda title, msg, level="INFO": events.append((title, msg, level)))
    out = await agent.answer(rid, system, history if history is not None else [HumanMessage(content="what does a.py do?")])
    return out, events


class FakeRunner:
    def __init__(self, name="fake", remote=False, ok=True):
        self.name, self.remote, self.ok, self.supports, self.max_input_tokens = name, remote, ok, {}, 0
        self.built = []

    def check_config(self):
        if not self.ok:
            raise RunnerConfigError(f"{self.name} is not set up")

    def map_model(self, model, *, vision=False):
        return f"{self.name}:{model}"

    def chat(self, model, **kw):
        self.built.append((model, kw))
        return ("chat", self.name, model)

    def embeddings(self, model=None, **kw):
        return ("emb", self.name)

    async def list_models(self):
        return [f"{self.name}-m1", f"{self.name}-m2"]


def add_runner(name="fake", **kw):
    r = FakeRunner(name, **kw)
    register_runner(name, lambda: r)
    return r


# ------------------------------------------------------------------ modes and budget()
def test_budget_is_normal_or_lean_and_never_above_normal(monkeypatch):
    assert budget("ask_max_steps") == 10 and budget("ask_obs_chars") == 90000 and budget("ask_read_chars") == 30000
    with token_mode_scope("lean"):
        assert [budget(n) for n in ("ask_max_steps", "ask_obs_chars", "ask_read_chars", "ask_seed_chars", "skill_pack_max_chars",
                                    "change_context_chars")] == [6, 40000, 12000, 4000, 6000, 40000]
        monkeypatch.setattr(settings, "ask_max_steps", 3)           # a lower normal value wins: lean only ever reduces
        assert budget("ask_max_steps") == 3
        monkeypatch.setattr(settings, "lean_ask_read_chars", 500)   # configurable
        assert budget("ask_read_chars") == 500
        assert budget("ask_grep_max") == settings.ask_grep_max      # not a lean budget
    assert budget("ask_read_chars") == 30000


def test_auto_follows_the_runner_remote_flag(monkeypatch):
    assert get_runner("ollama").remote is False and get_runner("cloudflare").remote is True
    assert effective_mode() == "normal"                              # ollama
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    assert effective_mode() == "lean" and is_lean()
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    add_runner("fakeremote", remote=True)
    with runner_scope("fakeremote"):                                 # a custom remote runner is lean too, without any special casing
        assert effective_mode() == "lean"
    assert effective_mode() == "normal"


def test_setting_and_per_request_override(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "token_mode", "normal")
    assert effective_mode() == "normal" and budget("ask_max_steps") == 10
    with token_mode_scope("lean"):
        assert effective_mode() == "lean"
        with token_mode_scope("bogus"):                              # invalid = keep the current pick
            assert effective_mode() == "lean"
        with token_mode_scope("auto"):                               # auto resolves by runner again (cloudflare -> lean)
            assert effective_mode() == "lean"
    assert effective_mode() == "normal"
    monkeypatch.setattr(settings, "token_mode", "lean")
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    assert effective_mode() == "lean"
    with token_mode_scope("normal"):
        assert effective_mode() == "normal"
    monkeypatch.setattr(settings, "token_mode", "nonsense")          # a broken setting counts as auto
    assert effective_mode() == "normal"


# ------------------------------------------------------------------ normal mode = today's prompts
def test_normal_mode_prompts_are_byte_identical_to_the_pre_token_mode_agent():
    replies = [act("read", path="a.py"), act("grep", pattern="line 3", path="b.py"), act("find", pattern="*.md"), "FINAL ANSWER"]
    hist = [HumanMessage(content="q%d" % i) if i % 2 == 0 else AIMessage(content="a%d" % i) for i in range(12)] + [HumanMessage(content="now?")]
    llm = Scripted(*replies)
    out, _ = run(ask(llm, list(hist), system="SYSTEM PROMPT"))

    # reference: how the agent built its prompts before token modes (full history, every observation in full)
    tools = RepoTools({p: t.encode() for p, t in FILES.items()})
    msgs = [("SystemMessage", "SYSTEM PROMPT" + guidance_block() + PROTOCOL)] + [(type(m).__name__, m.content) for m in hist]
    expected = [list(msgs)]
    for left, rep in zip(range(10, 7, -1), replies[:3]):
        d = json.loads(rep[len("ACTION: "):])
        obs = tools.run(d["tool"], d["args"])
        msgs += [("AIMessage", rep), ("HumanMessage", f"OBSERVATION ({d['tool']}):\n{obs}\n\n[{left - 1} tool call(s) left]")]
        expected.append(list(msgs))
    assert llm.seen == expected
    assert out.startswith("FINAL ANSWER")


def test_normal_mode_never_applies_the_input_cap_unless_set_and_remote(monkeypatch):
    assert max_input_tokens() == 0
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    monkeypatch.setattr(settings, "token_mode", "normal")
    assert max_input_tokens() == 0                                   # Cloudflare's 60000 is a LEAN default, not a surprise in normal mode
    monkeypatch.setattr(settings, "max_input_tokens", 1234)
    assert max_input_tokens() == 1234                                # explicit safety net on a remote runner
    monkeypatch.setattr(settings, "llm_runner", "ollama")
    assert max_input_tokens() == 0                                   # local runner in normal mode: never
    with token_mode_scope("lean"):
        assert max_input_tokens() == 1234
        monkeypatch.setattr(settings, "max_input_tokens", 0)         # 0 = off
        assert max_input_tokens() == 0
        monkeypatch.setattr(settings, "max_input_tokens", None)
        assert max_input_tokens() == 0                               # ollama has no cap of its own
        monkeypatch.setattr(settings, "llm_runner", "cloudflare")
        assert max_input_tokens() == 60000


# ------------------------------------------------------------------ lean: observation compaction, history
def test_lean_compacts_old_observations_into_one_line_stubs():
    reads = [act("read", path=f"{c}.py") for c in "abcde"]
    llm = Scripted(*reads, "DONE")
    with token_mode_scope("lean"):
        out, _ = run(ask(llm, system="SYS"))
    last = llm.seen[-1]                                              # the 6th call: 5 observations exist, 2 stay in full
    obs = [t for kind, t in last if t.startswith("OBSERVATION")]
    assert len(obs) == 5
    tools = RepoTools({p: t.encode() for p, t in FILES.items()})
    for c, got in zip("abc", obs[:3]):
        full = tools.run("read", {"path": f"{c}.py"})
        assert got == f"OBSERVATION (read) [compacted: {c}.py lines 1-60, {len(full):,} chars already seen; read again if needed]"
    assert obs[3].startswith("OBSERVATION (read):\nFILE d.py") and obs[4].startswith("OBSERVATION (read):\nFILE e.py")
    assert [k for k, _ in last].count("AIMessage") == 5              # the model's own replies stay
    assert last[0][0] == "SystemMessage" and last[1] == ("HumanMessage", "what does a.py do?")     # system + question untouched
    assert out.startswith("DONE") and "mode lean" in out


def test_compacted_calls_can_be_repeated_and_other_tools_get_stubs():
    tree = {p: t.encode() for p, t in FILES.items()}
    tools = RepoTools(tree)
    a1 = AIMessage(content=act("grep", pattern="line 1:", path="*.py"))
    o1 = tools.run("grep", {"pattern": "line 1:", "path": "*.py"})
    a2 = AIMessage(content=act("read", path="a.py"))
    o2 = tools.run("read", {"path": "a.py"})
    msgs = [SystemMessage(content="s"), HumanMessage(content="q"), a1, HumanMessage(content=f"OBSERVATION (grep):\n{o1}\n\n[8 tool call(s) left]"),
            a2, HumanMessage(content=f"OBSERVATION (read):\n{o2}\n\n[7 tool call(s) left]")]
    n, saved = compact_observations(msgs, 1, tools)
    assert n == 1 and saved > 0
    assert msgs[3].content == f"OBSERVATION (grep) [compacted: /line 1:/ in *.py, {len(o1):,} chars already seen; run it again if needed]"
    assert tools.run("grep", {"pattern": "line 1:", "path": "*.py"}) == o1          # forgotten: runs again, no "already ran this"
    assert tools.run("read", {"path": "a.py"}).startswith("You already ran")        # the newest one is still remembered
    assert compact_observations(msgs, 1, tools) == (0, 0)                            # idempotent
    assert observation_stub("ls", {"path": "x"}, "abc") == "OBSERVATION (ls) [compacted: x, 3 chars already seen; run it again if needed]"


def test_lean_history_is_trimmed_to_the_last_turns_and_normal_keeps_all():
    hist = []
    for i in range(7):
        hist += [HumanMessage(content=f"question {i}"), AIMessage(content=f"answer {i}")]
    hist.append(HumanMessage(content="current question"))
    kept, dropped = trim_history(hist, 6)
    assert [m.content for m in kept] == ["question 4", "answer 4", "question 5", "answer 5", "question 6", "answer 6", "current question"]
    assert dropped == sum(len(m.content) for m in hist[:8])
    odd, _ = trim_history(hist, 5)                                   # a slice never starts with an assistant reply
    assert odd[0].content == "question 4" or isinstance(odd[0], HumanMessage)
    assert trim_history(hist[-3:], 6)[1] == 0

    llm = Scripted("one")
    with token_mode_scope("lean"):
        run(ask(llm, list(hist)))
    shown = [t for k, t in llm.seen[0] if k != "SystemMessage"]
    assert shown == [m.content for m in kept]
    llm = Scripted("one")
    with token_mode_scope("normal"):
        run(ask(llm, list(hist)))
    assert len(llm.seen[0]) == 1 + len(hist)


def test_lean_tool_budgets_apply_to_read_and_steps(monkeypatch):
    monkeypatch.setattr(settings, "lean_ask_read_chars", 300)
    t = RepoTools({"a.py": FILES["a.py"].encode()})
    normal = t.read("a.py")
    with token_mode_scope("lean"):
        t2 = RepoTools({"a.py": FILES["a.py"].encode()})
        lean = t2.read("a.py")
    assert "TRUNCATED" not in normal and "TRUNCATED" in lean and len(lean) < 600
    monkeypatch.setattr(settings, "lean_ask_max_steps", 2)
    llm = Scripted(act("read", path="a.py"), act("read", path="b.py"), "stopped")
    with token_mode_scope("lean"):
        run(ask(llm))
    assert len(llm.seen) == 3 and "tool budget is used up" in llm.seen[-1][-1][1]


# ------------------------------------------------------------------ input cap
def test_input_cap_drops_oldest_history_first_then_old_observations(monkeypatch):
    monkeypatch.setattr(settings, "lean_keep_full_obs", 10)         # lean compaction off, so only the cap acts
    monkeypatch.setattr(settings, "lean_history_turns", 50)
    hist = []
    for i in range(4):
        hist += [HumanMessage(content=f"old question {i} " + "h" * 3000), AIMessage(content=f"old answer {i} " + "h" * 3000)]
    hist.append(HumanMessage(content="current question"))
    sys_tokens = (len("SYS") + len(PROTOCOL)) // 4

    # 1) cap leaves room for the system prompt, the question and ~3 old messages: only history goes
    monkeypatch.setattr(settings, "max_input_tokens", sys_tokens + 3 * 760)
    llm = Scripted("answer")
    with token_mode_scope("lean"):
        _, events = run(ask(llm, list(hist)))
    shown = [t for k, t in llm.seen[0]]
    assert shown[0].startswith("SYS") and shown[-1] == "current question"
    assert [t[:12] for t in shown[1:-1]] == [m.content[:12] for m in hist[-3:-1]]        # the newest earlier turn survives (a leading assistant reply is dropped too), oldest went first
    assert any(t == "Ask agent" and "input cap" in m and "dropped 6 earlier message" in m for t, m, _ in events)

    # 2) a cap below the system prompt + question: ALL history goes, then old tool output; system + question never
    monkeypatch.setattr(settings, "max_input_tokens", sys_tokens + 200)
    llm = Scripted(act("read", path="a.py"), act("read", path="b.py"), act("read", path="c.py"), "answer")
    with token_mode_scope("lean"):
        _, events = run(ask(llm, list(hist)))
    first = [t for k, t in llm.seen[0]]
    assert len(first) == 2 and first[1] == "current question"
    last = llm.seen[-1]
    obs = [t for k, t in last if t.startswith("OBSERVATION")]
    assert len(obs) == 3 and "[compacted:" in obs[0] and "[compacted:" in obs[1]
    assert obs[2].startswith("OBSERVATION (read):\nFILE c.py")      # the newest tool output is the one thing it keeps
    assert last[0][1].startswith("SYS") and last[1][1] == "current question"
    assert any("still over" in m for _, m, _ in events)


# ------------------------------------------------------------------ session affinity
def test_cloudflare_sends_a_stable_session_affinity_header_per_thread(monkeypatch):
    class Rec:
        def __init__(self, **kw):
            self.kw = kw
    cf = get_runner("cloudflare")
    with thread_scope("repoA", "th1"):
        h1 = cf.chat("m", chat_cls=Rec).kw["default_headers"]
        h1b = cf.chat("m", chat_cls=Rec).kw["default_headers"]
    with thread_scope("repoA", "th2"):
        h2 = cf.chat("m", chat_cls=Rec).kw["default_headers"]
    with thread_scope("repoB", "th1"):
        h3 = cf.chat("m", chat_cls=Rec).kw["default_headers"]
    assert h1 == h1b == {"x-session-affinity": session_key("repoA", "th1")}
    assert len({h1["x-session-affinity"], h2["x-session-affinity"], h3["x-session-affinity"]}) == 3
    assert "repoA" not in h1["x-session-affinity"] and "th1" not in h1["x-session-affinity"]
    assert "default_headers" not in cf.chat("m", chat_cls=Rec).kw          # no thread (e.g. a script): no header
    monkeypatch.setattr(settings, "cf_session_affinity", False)
    with thread_scope("repoA", "th1"):
        assert "default_headers" not in cf.chat("m", chat_cls=Rec).kw       # switched off
    with thread_scope("repoA", "th1"):
        assert "default_headers" not in get_runner("ollama").chat("m", chat_cls=Rec).kw


# ------------------------------------------------------------------ telemetry and footer
def _result(usage=None, token_usage=None, text="hi", resp_meta=None):
    msg = types.SimpleNamespace(usage_metadata=usage, response_metadata=resp_meta or {})
    gen = types.SimpleNamespace(message=msg, generation_info={}, text=text)
    return types.SimpleNamespace(generations=[[gen]], llm_output={"token_usage": token_usage} if token_usage else {})


def test_callback_records_cached_tokens_and_the_ledger_aggregates(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    sent = []
    with ledger_scope() as led:
        cb = StdoutTelemetryCallback("m1", "r1", phase="ASK_AGENT", emitter=sent.append)
        led.add_saved(40000)                                          # lean saved 10k tokens before the first call
        cb.on_chat_model_start({}, [[HumanMessage(content="x")]])
        cb.on_llm_end(_result(usage={"input_tokens": 12000, "output_tokens": 500, "input_token_details": {"cache_read": 4000}}))
        cb.on_chat_model_start({}, [[HumanMessage(content="x")]])
        cb.on_llm_end(_result(token_usage={"prompt_tokens": 6200, "completion_tokens": 400,
                                           "prompt_tokens_details": {"cached_tokens": 2100}}))
    first, second = (json.loads(s) for s in sent)
    assert (first["runner"], first["token_mode"], first["cached_input_tokens"], first["est_tokens_saved"]) == ("cloudflare", "lean", 4000, 10000)
    assert (second["input_tokens"], second["output_tokens"], second["cached_input_tokens"], second["est_tokens_saved"]) == (6200, 400, 2100, 0)
    t = led.totals()
    assert (t["calls"], t["input_tokens"], t["output_tokens"], t["cached_input_tokens"], t["est_tokens_saved"]) == (2, 18200, 900, 6100, 10000)
    assert footer_text(t) == "_Tokens: 18.2k in (6.1k cached) · 0.9k out · 2 calls · runner cloudflare · mode lean (saved ~10k)_"


def test_footer_text_variants():
    base = dict(calls=1, input_tokens=800, output_tokens=50, cached_input_tokens=0, est_tokens_saved=0, runner="ollama", token_mode="normal")
    assert footer_text(base) == "_Tokens: 0.8k in · 50 out · 1 call · runner ollama · mode normal_"
    assert footer_text({**base, "est_tokens_saved": 31400, "calls": 6}).endswith("6 calls · runner ollama · mode normal (saved ~31k)_")
    assert footer_text({**base, "calls": 0}) == ""


def test_answer_has_the_token_footer_and_one_summary_event_and_it_is_switchable(monkeypatch):
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    llm = Scripted(act("read", path="a.py"), "DONE")
    out, events = run(ask(llm))
    assert re.search(r"\n\n_Tokens: [\d.k]+ in · [\d.k]+ out · 2 calls · runner cloudflare · mode lean", out)
    summaries = [json.loads(m) for t, m, _ in events if t == "Token usage"]
    assert len(summaries) == 1 and summaries[0]["calls"] == 2 and summaries[0]["runner"] == "cloudflare"
    assert len(summaries[0]["calls_detail"]) == 2 and summaries[0]["calls_detail"][0]["phase"] == "ASK_AGENT"
    monkeypatch.setattr(settings, "show_token_footer", False)
    out, events = run(ask(Scripted("DONE")))
    assert "_Tokens:" not in out and any(t == "Token usage" for t, _, _ in events)       # the event stays; only the footer is optional


def test_footer_helper_keeps_the_old_text_without_a_ledger():
    t = RepoTools({"a.py": b"x\n"})
    assert CodeAgent._footer(t) == "\n\n_No search tools were used: this answer comes from the semantic-search leads only._"


# ------------------------------------------------------------------ runner switch
def test_runner_scope_routes_chat_but_embeddings_stay_on_the_embedding_runner(monkeypatch):
    fake = add_runner("fake")
    assert get_runner().name == "ollama"
    with runner_scope("fake"):
        assert current_runner_name() == "fake" and get_runner() is fake
        assert make_chat("m", temperature=0) == ("chat", "fake", "m")
        assert resolve_model("x") == "fake:x"
        assert run(model_calls.list_chat_models()) == ["fake-m1", "fake-m2"]
        assert get_embedding_runner().name == "ollama"                # index compatibility: never the per-request runner
        monkeypatch.setattr(settings, "embedding_runner", "cloudflare")
        assert get_embedding_runner().name == "cloudflare"
        assert get_runner("ollama").name == "ollama"                  # an explicit name still wins
    assert get_runner().name == "ollama" and resolve_model() == settings.llm_model
    with runner_scope(""):
        assert get_runner().name == "ollama"
    with runner_scope("nope"), pytest.raises(RunnerConfigError):
        get_runner()


def test_build_maps_the_model_with_the_per_request_runner(monkeypatch):
    fake = add_runner("fake")
    with runner_scope("fake"):
        assert model_calls._build("gemma4:31b-cloud", json_mode=False, num_predict=5, num_ctx=7, repo_id="r", phase="P") == \
            ("chat", "fake", "fake:gemma4:31b-cloud")
    assert fake.built[0][1]["max_tokens"] == 5
    with runner_scope("cloudflare"):
        assert resolve_model("gemma4:31b-cloud") == CF_DEFAULT          # Ollama-style name -> Cloudflare default
    assert get_runner("ollama").map_model("@cf/google/gemma-3-12b-it") == settings.llm_model    # and back, after a switch to Ollama
    assert get_runner("ollama").map_model("qwen2.5-coder:7b") == "qwen2.5-coder:7b"


def test_runners_payload_is_built_from_the_registry(monkeypatch):
    add_runner("fake", remote=True, ok=False)
    p = runners_payload()
    assert p["default"] == "ollama" and p["token_mode"] == "auto"
    by = {r["name"]: r for r in p["runners"]}
    assert by["ollama"] == {"name": "ollama", "label": "Ollama (local)", "remote": False, "configured": True, "error": None}
    assert by["cloudflare"]["remote"] is True and by["cloudflare"]["configured"] is True
    assert by["fake"]["remote"] is True and by["fake"]["configured"] is False and "not set up" in by["fake"]["error"]
    monkeypatch.setattr(settings, "cf_api_token", "")
    cf = {r["name"]: r for r in runner_status()}["cloudflare"]
    assert cf["configured"] is False and "CF_API_TOKEN" in cf["error"]


def test_models_payload_per_runner_and_validation(monkeypatch):
    add_runner("fake", ok=False)
    default = run(models_payload())
    assert default["runner"] == "ollama" and default["default"] in default["models"]
    cf = run(models_payload("Cloudflare"))
    assert cf == {"default": CF_DEFAULT, "models": [CF_DEFAULT], "runner": "cloudflare"}
    monkeypatch.setattr(settings, "llm_runner", "cloudflare")
    assert run(models_payload())["runner"] == "cloudflare"            # no ?runner= -> the configured one
    with pytest.raises(RunnerConfigError, match="Unknown runner"):
        run(models_payload("nope"))
    with pytest.raises(RunnerConfigError, match="not set up"):
        run(models_payload("fake"))


def test_chat_choice_rejects_unknown_unconfigured_runner_and_bad_mode(monkeypatch):
    add_runner("fake", ok=False)
    assert check_chat_choice("", "") == ("", "")
    assert check_chat_choice(" Cloudflare ", "LEAN") == ("cloudflare", "lean")
    with pytest.raises(RunnerConfigError, match="Unknown runner 'nope'"):
        check_chat_choice("nope", "")
    with pytest.raises(RunnerConfigError, match="not set up"):
        check_chat_choice("fake", "")
    monkeypatch.setattr(settings, "cf_account_id", "")
    with pytest.raises(RunnerConfigError, match="CF_ACCOUNT_ID"):
        check_chat_choice("cloudflare", "")
    with pytest.raises(ValueError, match="token_mode"):
        check_chat_choice("", "turbo")
    assert require_runner(None) == ""


def test_the_api_layer_wires_runner_and_token_mode_through():
    """FastAPI is stubbed in this environment, so check the wiring in the source of the thin layers."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    api = (root / "views" / "api_view.py").read_text()
    assert '"/api/v1/runners"' in api and "runner: Optional[str] = None" in api and api.count("runner=runner, token_mode=token_mode") == 3
    assert 'runner: str = Form("")' in api and 'token_mode: str = Form("")' in api
    ctl = (root / "controllers" / "engine_controller.py").read_text()
    assert "runner_scope(runner)" in ctl and "token_mode_scope(token_mode)" in ctl and "thread_scope(repo_id, thread_id)" in ctl
    js = (root / "static" / "js" / "app.js").read_text()
    assert js.count("runner: S.runner") == 1 and "fd.append('runner', S.runner)" in js and "fd.append('token_mode', S.tokenMode)" in js
    assert "LS.get('runner'" in js and "LS.set('tokenMode'" in js and "/api/v1/runners" in js
    html = (root / "templates" / "index.html").read_text()
    assert 'id="runner-select"' in html and 'id="token-mode-select"' in html
