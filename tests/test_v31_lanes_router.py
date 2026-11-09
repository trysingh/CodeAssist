"""v3.1: mixed-request lanes, unknown slash commands, guidance leaking into output, pack text, line endings."""
import asyncio
import importlib
import sys
import types
from pathlib import Path

import pytest

from services.reasoning import (ChangeType, IntakeAgent, Lane, ReasoningSupervisor, VerifierAgent, guidance_block,
                                strip_guidance_echo)
from services.reasoning import skill_pack as sp
from services.reasoning.baseline import GUIDANCE_HEADER
from services.reasoning.intake import classify_lane, detect_change_type, detect_change_types, recommend_validation_level
from services.reasoning.skill_pack import guidance_for, lane_statement, load_skill_prompt, sections_for

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "skills" / "risk-proportional-code-dev"
REQUEST = ("give a more modern look to the landing page with color schemes for red-green color blindness friendly users. "
           "Add audible effects when starting and stopping services and ensure it is responsive")


def brief(q, intent="change"):
    return IntakeAgent().analyze(q, intent)


# ---------------------------------------------------------------- bug 2: mixed requests
def test_transcript_request_is_standard_not_fast():
    b = brief(REQUEST)
    assert b.lane == Lane.STANDARD, b.lane_reasons
    assert b.change_type == ChangeType.FE_BEHAVIOURAL
    assert set(b.change_types) == {ChangeType.FE_BEHAVIOURAL, ChangeType.FE_PRESENTATIONAL}
    assert b.change_types[0] == b.change_type
    assert b.validation_level == "L1, L3"
    assert "fe_presentational" in " ".join(b.lane_reasons)


def test_pure_css_tweak_stays_fast():
    for q in ("change the button colour to blue in styles.css", "make the dashboard layout responsive in layout.scss",
              "use a colour-blind friendly palette in theme.css"):
        b = brief(q)
        assert b.lane == Lane.FAST and b.change_types == [ChangeType.FE_PRESENTATIONAL], (q, b.lane_reasons)


@pytest.mark.parametrize("q", [
    "play a sound when the service starts and update the colours in styles.css",
    "add a click handler and restyle the button in app.css",
    "fetch the status every 5 seconds and restyle the badge colour",
    "add audio feedback to the page",
    "play an audible beep on start",
    "restyle the header and call fetch on submit",
])
def test_js_behaviour_in_a_mixed_request_is_at_least_standard(q):
    b = brief(q)
    assert b.lane in (Lane.STANDARD, Lane.STRICT), b.lane_reasons
    assert ChangeType.FE_BEHAVIOURAL in b.change_types or ChangeType.FE_SECURITY in b.change_types


def test_lane_is_the_max_across_types_and_primary_drives_it():
    types = detect_change_types("restyle styles.css and write an alembic migration")
    assert [t for t, _ in types][0] == ChangeType.MIGRATION and len(types) >= 2
    assert brief("restyle styles.css and write an alembic migration").lane == Lane.STRICT
    lane, _, locked = classify_lane("x", brief("add retries to client.py").risk,
                                    [ChangeType.FE_PRESENTATIONAL, ChangeType.FE_BEHAVIOURAL])
    assert lane == Lane.STANDARD and not locked
    assert detect_change_type(REQUEST)[0] == ChangeType.FE_BEHAVIOURAL
    assert recommend_validation_level(Lane.STANDARD, [ChangeType.FE_PRESENTATIONAL, ChangeType.FE_BEHAVIOURAL]) == "L1, L3"
    assert recommend_validation_level(Lane.STRICT, [ChangeType.BACKEND, ChangeType.SCRIPT]) == "L2"


def test_single_type_requests_have_a_one_item_list():
    b = brief("add retries to client.py")
    assert b.change_types == [ChangeType.BACKEND] and b.change_type == ChangeType.BACKEND


def test_override_still_cannot_lower_strict_on_mixed_requests():
    b = brief("lane: fast restyle styles.css and use innerHTML in app.js")
    assert b.lane == Lane.STRICT


# ---------------------------------------------------------------- bug 5: change-like asks get the CORE skill text only
ASK = "do you think we can " + REQUEST[0].lower() + REQUEST[1:]


def test_change_like_ask_detection():
    assert brief(ASK, "ask").change_like
    assert brief("can you add retries to client.py?", "ask").change_like
    for q in ("what does the audio code do?", "explain how the sound effects work", "why is the page not responsive",
              "how does the dashboard start a service?", "show me the click handler"):
        assert not brief(q, "ask").change_like, q
    assert not brief("add retries to client.py", "change").change_like


def test_core_only_for_change_like_ask():
    assert sections_for("standard", "fe_behavioural", "low", "ask", change_like=True) == ["core"]
    assert sections_for("standard", "fe_behavioural", "low", "ask") == []
    assert sections_for("strict", "backend", "high", "ask", change_like=True) == ["core"]
    b = brief(ASK, "ask")
    g = guidance_for(b, enabled=True, pack_dir=PACK, max_chars=12000)
    assert g.startswith("LANE FOR THIS REQUEST: STANDARD") and "Never weaken, skip, delete or rewrite a test" in g
    assert "## Handoff protocol" not in g and "# Validation ladder" not in g and "# Test-fix loop" not in g
    assert len(g) < 6000
    small = guidance_for(b, enabled=True, pack_dir=PACK, max_chars=1500)
    assert len(small) <= 1500 + len(lane_statement(b)) + 4
    assert guidance_for(brief("what does the audio code do?", "ask"), enabled=True, pack_dir=PACK, max_chars=12000) == ""


# ---------------------------------------------------------------- bug 6: guidance must not leak
LEAK = ("LANE FOR THIS REQUEST: FAST (change type: fe_presentational). Apply the Fast-lane rules below.\n\n"
        "FEASIBLE: yes, purely front-end.\nFILES:\n- templates/dashboard.html - restyle\n")


def test_strip_guidance_echo():
    out = strip_guidance_echo(LEAK)
    assert out.startswith("FEASIBLE: yes") and "LANE FOR THIS REQUEST" not in out and "FILES:" in out
    assert strip_guidance_echo("**Plan**\n\n> LANE FOR THIS REQUEST: STRICT (x)\nbody\n") == "**Plan**\n\nbody\n"
    assert strip_guidance_echo("a\nengineering guidance (how to work on this request): x\nb") == "a\nb"
    assert strip_guidance_echo("a\nINTERNAL CONTEXT (background...):\nb\n") == "a\nb\n"
    clean = "FEASIBLE: yes\nRISKS:\n- the lane of traffic\n"
    assert strip_guidance_echo(clean) == clean and strip_guidance_echo("") == "" and strip_guidance_echo(None) is None


def test_header_says_internal_and_do_not_repeat():
    h = GUIDANCE_HEADER.lower()
    assert "internal context" in h and "never repeat" in h and "not part of the task or the answer" in h
    st = lane_statement(brief("change the colour in styles.css")).lower()
    assert "internal" in st and "do not state the lane" in st


def test_supervisor_strips_an_echoed_header_from_the_answer():
    async def execute(b, p):
        return LEAK, {}
    sup = ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {}, True, guidance_provider=lambda b, p: "G")
    out = asyncio.run(sup.run(query="change the colour in styles.css", intent="change", repo_files=[], execute=execute))
    assert "LANE FOR THIS REQUEST" not in out and out.startswith("FEASIBLE")
    sup = ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {}, True)         # no guidance: answer untouched
    out = asyncio.run(sup.run(query="change the colour in styles.css", intent="change", repo_files=[], execute=execute))
    assert out.startswith("LANE FOR THIS REQUEST")


def test_plan_path_strips_the_echo():
    assert "strip_guidance_echo(plan)" in (ROOT / "services" / "repo_change.py").read_bytes().decode("utf-8")


# ---------------------------------------------------------------- bug 7: FE behavioural guidance in the pack
def test_fe_behavioural_row_has_the_practices_and_stays_tiny():
    ladder = (PACK / "references" / "validation-ladder.md").read_text(encoding="utf-8")
    row = next(l for l in ladder.splitlines() if l.startswith("| FE behavioural"))
    for needle in ("autoplay", "user-gesture", "one AudioContext", "feedback after the outcome is known", "user-controllable",
                   "stored preference", "colour alone"):
        assert needle in row, needle
    assert len(row) < 900
    assert len((PACK / "SKILL.md").read_text(encoding="utf-8").splitlines()) <= 60


def test_standard_prompt_for_the_transcript_request_carries_the_practices_and_fast_css_does_not():
    b = brief(REQUEST)
    std = load_skill_prompt(lane=b.lane, change_type=b.change_type, risk=b.risk, pack_dir=PACK)
    assert "one AudioContext" in std and "colour alone" in std
    fast = load_skill_prompt(lane=Lane.FAST, change_type=ChangeType.FE_PRESENTATIONAL, risk="low", pack_dir=PACK)
    assert "AudioContext" not in fast and "not colour-only" in fast


# ---------------------------------------------------------------- bug 4: unknown slash commands
@pytest.fixture(scope="module")
def router():
    """services.intent_router needs config.settings (pydantic_settings) and services.decision. If the real ones import, use
    them; otherwise load the router against throw-away stubs and remove every trace afterwards."""
    try:
        return importlib.import_module("services.intent_router")
    except ImportError:
        pass
    names = ["config", "config.settings", "services.decision", "services.intent_router"]
    saved = {n: sys.modules.get(n) for n in names}
    pkg = types.ModuleType("config")
    pkg.__path__ = []
    st = types.ModuleType("config.settings")
    st.settings = types.SimpleNamespace(router_auto=False, router_model="", router_min_confidence=0.75)
    dec = types.ModuleType("services.decision")

    async def decide_noul(*a, **k):
        raise AssertionError("an unknown command must not reach the decision model")
    dec.decide_noul = decide_noul
    sys.modules.update({"config": pkg, "config.settings": st, "services.decision": dec})
    sys.modules.pop("services.intent_router", None)
    try:
        mod = importlib.import_module("services.intent_router")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return mod


def route(router, q):
    return asyncio.run(router.route_intent(q))


def test_apply_is_an_unknown_command_with_a_hint(router):
    d = route(router, "/apply " + REQUEST)
    assert d.intent == "help" and d.source == "rules" and d.confidence == 1.0
    assert d.reply.startswith("Unknown command `/apply`. Did you mean `/change` (propose a revision) or `/ask`?")
    assert "`/plan <request>`" in d.reply and router.COMMANDS_HELP in d.reply
    assert route(router, "/apply").intent == "help"                      # end of string counts too
    assert route(router, "  /Apply   now").reply.startswith("Unknown command `/Apply`")


@pytest.mark.parametrize("q", [
    "/home/x fails with a permission error", "/etc/hosts is wrong", "/usr/bin/python crashes", "/v1/api returns 500",
    "see /apply/foo", "what does /apply do?", "fix the bug in app.py",
])
def test_paths_and_ordinary_messages_are_not_hints(router, q):
    d = route(router, q)
    assert not d.reply and d.intent != "help"


def test_known_commands_are_unchanged(router):
    assert route(router, "/ask what is x").intent == "ask"
    assert route(router, "/change do it").intent == "change" and route(router, "/fix the bug").intent == "change"
    assert route(router, "/plan it").intent == "plan"
    assert route(router, "/help").intent == "help" and not route(router, "/help").reply
    assert route(router, "/commands").intent == "help"
    assert router.unknown_command("/web latest news") is None
    assert router.unknown_command("/apply x") == "apply" and router.unknown_command("/home/x") is None


def test_apply_is_not_an_alias(router):
    assert "apply" not in router._KNOWN_COMMANDS and "apply" not in router._PREFIX.pattern


def test_controller_returns_the_hint_reply():
    src = (ROOT / "controllers" / "engine_controller.py").read_bytes().decode("utf-8")
    assert src.count("decision.reply or COMMANDS_HELP") == 2 and "return COMMANDS_HELP" not in src


# ---------------------------------------------------------------- line endings of the CRLF files must survive edits
@pytest.mark.parametrize("path", ["README.md", "config/settings.py", "controllers/engine_controller.py",
                                  "services/repo_change.py", "services/code_agent.py", "services/intent_router.py"])
def test_crlf_files_stay_crlf(path):
    b = (ROOT / path).read_bytes()
    assert b.count(b"\r\n") > 0 and b.count(b"\r\n") == b.count(b"\n"), path
