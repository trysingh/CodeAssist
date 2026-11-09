"""v3: skill pack loader. Pure module (stdlib + services.reasoning.intake): no pydantic_settings / langchain needed."""
import os
import shutil
from pathlib import Path

import pytest

from services.reasoning import IntakeAgent, Lane
from services.reasoning import skill_pack as sp
from services.reasoning.skill_pack import (guidance_for, ladder_for_presentational_fe, load_pack, load_skill_prompt,
                                           parse_frontmatter, sections_for)

PACK = Path(__file__).resolve().parent.parent / "skills" / "risk-proportional-code-dev"

# phrases of the never-skip rules (SKILL.md core)
NEVER_SKIP = [
    "Never weaken, skip, delete or rewrite a test",
    "Never special-case test inputs",
    "Confirmation gate",
    "`git push`",
    "Never ask for passwords, OTPs or tokens in chat",
    "Never type, store, log or screenshot secrets",
    "Never hard-code or log secrets",
    "**Statically checked**",
    "**Unverified (user must run)**",
    "Verified – executed in",
]
LADDER_ONLY = "## Handoff protocol"
REGULATED_ONLY = "Threat-model the change"
DELIVERY_ONLY = "## Test-fix loop"


@pytest.fixture(autouse=True)
def _fresh_cache():
    sp.clear_cache()
    yield
    sp.clear_cache()


def prompt(lane, ctype=None, risk=None, intent="change", **kw):
    return load_skill_prompt(lane=lane, change_type=ctype, risk=risk, intent=intent, pack_dir=PACK, **kw)


@pytest.fixture
def pack_copy(tmp_path):
    dst = tmp_path / "pack"
    shutil.copytree(PACK, dst)
    return dst


# ---------------------------------------------------------------- the shipped pack
def test_shipped_pack_layout():
    core = (PACK / "SKILL.md").read_text(encoding="utf-8")
    assert 50 <= len(core.splitlines()) <= 60
    meta, body = parse_frontmatter(core)
    assert meta["name"] == "risk-proportional-code-dev" and meta["description"] and body.startswith("# ")
    for ref in sp.REFERENCES.values():
        assert (PACK / "references" / ref).is_file()
        assert f"references/{ref}" in core                      # the core points at every reference
    ladder = (PACK / "references" / "validation-ladder.md").read_text(encoding="utf-8")
    for needle in ("| L0 ", "| L5 ", "## Profile by change type", "## Handoff protocol", "## Live-testing guardrails"):
        assert needle in ladder


# ---------------------------------------------------------------- section selection
@pytest.mark.parametrize("lane,ctype,risk,intent,expected", [
    ("fast", "docs", "low", "change", ["core"]),
    ("fast", "backend", "low", "change", ["core"]),
    ("fast", "fe_presentational", "low", "change", ["core", "validation-ladder"]),
    ("standard", "backend", "low", "change", ["core", "validation-ladder", "delivery-and-review"]),
    ("standard", "backend", "high", "change", ["core", "validation-ladder", "regulated-domains", "delivery-and-review"]),
    ("strict", "migration", "high", "change", ["core", "validation-ladder", "regulated-domains", "delivery-and-review"]),
    ("fast", "fe_presentational", "high", "change", ["core", "validation-ladder", "regulated-domains"]),
    ("strict", "backend", "low", "plan", ["core", "validation-ladder", "regulated-domains", "delivery-and-review"]),
    ("strict", "backend", "high", "ask", []),
    ("standard", "backend", "low", "help", []),
    (None, "backend", "low", "change", []),
])
def test_sections_for(lane, ctype, risk, intent, expected):
    assert sections_for(lane, ctype, risk, intent) == expected


def test_sections_accept_enums_too():
    b = IntakeAgent().analyze("change the button colour in styles.css", "change")
    assert sections_for(b.lane, b.change_type, b.risk, b.intent) == ["core", "validation-ladder"]


def test_loader_selection_per_lane():
    fast_docs = prompt(Lane.FAST, "docs", "low")
    assert NEVER_SKIP[0] in fast_docs and LADDER_ONLY not in fast_docs and REGULATED_ONLY not in fast_docs
    assert DELIVERY_ONLY not in fast_docs

    std = prompt(Lane.STANDARD, "backend", "low")
    assert LADDER_ONLY in std and "## Live-testing guardrails" in std and DELIVERY_ONLY in std
    assert REGULATED_ONLY not in std

    strict = prompt(Lane.STRICT, "migration", "high")
    for part in (LADDER_ONLY, REGULATED_ONLY, DELIVERY_ONLY):
        assert part in strict
    assert strict.index("# Risk-Proportional") < strict.index("# Validation ladder") < strict.index("# Security and regulated") \
        < strict.index("# Test-fix loop")


def test_task_types_reference_is_never_injected():
    for lane in Lane:
        assert "## Clarify gate" not in prompt(lane, "backend", "high")


def test_fast_presentational_fe_loads_only_the_relevant_ladder_part():
    text = prompt(Lane.FAST, "fe_presentational", "low")
    assert "| L3 " in text and "| L0 " in text                       # the level table
    assert "| FE presentational" in text and "Breakpoints, light/dark" in text
    for gone in (LADDER_ONLY, "## Live-testing guardrails", "| Migrations", "| SQL", "| FE behavioural", "| IaC / config"):
        assert gone not in text
    assert len(text) < len(prompt(Lane.STANDARD, "fe_presentational", "low"))


def test_ladder_extraction_falls_back_to_the_whole_file_on_an_unexpected_layout():
    odd = "# Ladder\n\njust some text without the expected headings\n"
    assert ladder_for_presentational_fe(odd) == odd.strip()


def test_ask_and_unclassified_get_no_pack():
    assert prompt(Lane.STRICT, "backend", "high", intent="ask") == ""
    assert prompt(None, "backend", "low") == ""


# ---------------------------------------------------------------- never-skip rules
@pytest.mark.parametrize("lane,ctype,risk", [
    (Lane.FAST, "docs", "low"), (Lane.FAST, "fe_presentational", "low"), (Lane.STANDARD, "backend", "medium"),
    (Lane.STRICT, "migration", "high"),
])
def test_core_never_skip_rules_always_included(lane, ctype, risk):
    text = prompt(lane, ctype, risk)
    for rule in NEVER_SKIP:
        assert rule in text, rule


def test_never_skip_rules_survive_a_tight_cap():
    text = prompt(Lane.STRICT, "migration", "high", max_chars=5000)
    assert len(text) <= 5000
    for rule in NEVER_SKIP:
        assert rule in text, rule


# ---------------------------------------------------------------- size cap
def test_size_cap_is_respected_and_most_important_text_wins():
    full = prompt(Lane.STRICT, "migration", "high", max_chars=100000)
    assert len(full) > 9000
    for cap in (12000, 8000, 5000, 4500, 3000, 500, 50, 1):
        t = prompt(Lane.STRICT, "migration", "high", max_chars=cap)
        assert len(t) <= cap, cap
        assert t == "" or t.startswith("# Risk-Proportional") or cap < 30
    capped = prompt(Lane.STRICT, "migration", "high", max_chars=6000)
    assert "# Risk-Proportional" in capped and "# Test-fix loop" not in capped            # later sections drop first
    assert "truncated" in prompt(Lane.STRICT, "migration", "high", max_chars=3000)


def test_zero_or_negative_cap_gives_empty():
    assert prompt(Lane.STRICT, "backend", "high", max_chars=0) == ""
    assert prompt(Lane.STRICT, "backend", "high", max_chars=-5) == ""


# ---------------------------------------------------------------- missing / broken pack
def test_missing_pack_returns_empty_string_without_raising(tmp_path, caplog):
    for d in (tmp_path / "nope", "", None, "/definitely/not/here", tmp_path):
        assert load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=d) == ""
    assert load_pack(tmp_path / "nope") is None


def test_missing_pack_is_logged_once_not_per_request(tmp_path, caplog):
    caplog.set_level("WARNING", logger=sp.log.name)
    for _ in range(5):
        load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=tmp_path / "nope")
    assert len([r for r in caplog.records if "skill pack" in r.getMessage()]) == 1


@pytest.mark.parametrize("content", [
    "no frontmatter at all\n# Title\nbody\n",
    "---\ndescription: no name here\n---\n# Title\nbody\n",
    "---\nname: x\n# unterminated frontmatter\nbody\n",
    "---\nname: x\n---\n\n",
])
def test_broken_core_returns_empty(pack_copy, content):
    (pack_copy / "SKILL.md").write_text(content, encoding="utf-8")
    assert load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=pack_copy) == ""


def test_undecodable_core_returns_empty(pack_copy):
    (pack_copy / "SKILL.md").write_bytes(b"---\nname: x\n---\n\xff\xfe\x00 broken \xc3\x28\n")
    assert load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=pack_copy) == ""


def test_pack_dir_is_a_file_or_unreadable_does_not_raise(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    assert load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=f) == ""


def test_missing_reference_is_skipped_but_core_stays(pack_copy):
    (pack_copy / "references" / "regulated-domains.md").unlink()
    text = load_skill_prompt(lane=Lane.STRICT, change_type="backend", risk="high", pack_dir=pack_copy)
    assert NEVER_SKIP[0] in text and LADDER_ONLY in text and REGULATED_ONLY not in text


def test_relative_dir_resolves_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = load_skill_prompt(lane=Lane.FAST, change_type="docs", risk="low", pack_dir="skills/risk-proportional-code-dev")
    assert NEVER_SKIP[0] in t


# ---------------------------------------------------------------- frontmatter (no yaml)
def test_parse_frontmatter():
    meta, body = parse_frontmatter('---\nname: "my-skill"\ndescription: first part\n  second part\n# comment\nother: v: w\n---\n\n# Body\n')
    assert meta == {"name": "my-skill", "description": "first part second part", "other": "v: w"} and body == "# Body"
    assert parse_frontmatter("# no block\n") == ({}, "# no block\n")
    assert parse_frontmatter("---\nname: x\nno end\n")[0] == {}
    assert parse_frontmatter("﻿---\nname: x\n---\nbody")[0] == {"name": "x"}
    assert parse_frontmatter("")[0] == {}


def test_loader_does_not_import_yaml():
    src = Path(sp.__file__).read_text(encoding="utf-8")
    assert "import yaml" not in src and "from yaml" not in src


# ---------------------------------------------------------------- cache by mtime
def test_cache_returns_the_same_pack_until_a_file_changes(pack_copy):
    a = load_pack(pack_copy)
    assert a is load_pack(pack_copy)
    ref = pack_copy / "references" / "regulated-domains.md"
    ref.write_text(ref.read_text(encoding="utf-8") + "- EXTRA RULE\n", encoding="utf-8")
    b = load_pack(pack_copy)
    assert b is not a and "EXTRA RULE" in b.refs["regulated-domains"]
    assert b is load_pack(pack_copy)


def test_same_size_edit_with_new_mtime_is_picked_up(pack_copy):
    core = pack_copy / "SKILL.md"
    a = load_pack(pack_copy)
    data = core.read_text(encoding="utf-8").replace("Fast reply", "Quik reply")
    core.write_text(data, encoding="utf-8")
    st = core.stat()
    os.utime(core, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    b = load_pack(pack_copy)
    assert b is not a and "Quik reply" in b.core


def test_a_pack_that_appears_later_is_found(tmp_path):
    d = tmp_path / "late"
    assert load_skill_prompt(lane=Lane.FAST, change_type="docs", risk="low", pack_dir=d) == ""
    shutil.copytree(PACK, d)
    assert NEVER_SKIP[0] in load_skill_prompt(lane=Lane.FAST, change_type="docs", risk="low", pack_dir=d)


def test_a_pack_repaired_after_breaking_is_picked_up(pack_copy):
    core = pack_copy / "SKILL.md"
    good = core.read_text(encoding="utf-8")
    core.write_text("garbage", encoding="utf-8")
    assert load_skill_prompt(lane=Lane.FAST, change_type="docs", risk="low", pack_dir=pack_copy) == ""
    core.write_text(good + "\n", encoding="utf-8")
    assert NEVER_SKIP[0] in load_skill_prompt(lane=Lane.FAST, change_type="docs", risk="low", pack_dir=pack_copy)


# ---------------------------------------------------------------- guidance_for (what the supervisor injects)
def _brief(q, intent="change"):
    return IntakeAgent().analyze(q, intent)


def test_guidance_has_a_one_line_lane_statement_then_the_pack():
    g = guidance_for(_brief("change the button colour in styles.css"), enabled=True, pack_dir=PACK, max_chars=12000)
    first = g.splitlines()[0]
    assert first.startswith("LANE FOR THIS REQUEST: FAST") and "change type: fe_presentational" in first
    assert NEVER_SKIP[0] in g and "| FE presentational" in g


def test_guidance_off_for_ask_disabled_or_unclassified():
    assert guidance_for(_brief("explain the login flow", "ask"), enabled=True, pack_dir=PACK, max_chars=12000) == ""
    assert guidance_for(_brief("add retries to client.py"), enabled=False, pack_dir=PACK, max_chars=12000) == ""
    from services.reasoning import IntakeBrief
    assert guidance_for(IntakeBrief(intent="change"), enabled=True, pack_dir=PACK, max_chars=12000) == ""


def test_guidance_empty_when_pack_missing_so_no_orphan_lane_line(tmp_path):
    assert guidance_for(_brief("add retries to client.py"), enabled=True, pack_dir=tmp_path / "x", max_chars=12000) == ""


def test_strict_request_gets_the_regulated_section():
    g = guidance_for(_brief("change the password hashing in auth.py"), enabled=True, pack_dir=PACK, max_chars=12000)
    assert g.startswith("LANE FOR THIS REQUEST: STRICT") and REGULATED_ONLY in g
    assert len(g) <= 12000 + 400                                       # default cap + the lane line
