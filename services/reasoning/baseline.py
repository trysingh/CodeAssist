"""BaselinePolicy: risk-proportional strictness overlay. Only ever RAISES strictness."""
from __future__ import annotations

import contextlib
import contextvars
import re
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from .intake import IntakeBrief, Lane, Risk

GATE_ORDER = {"off": 0, "smoke": 1, "strict": 2}
ALLOWED_KEYS = ("change_gate", "change_review", "change_tests_first", "change_author_tests", "change_max_attempts")

_overlay: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("baseline_overlay", default={})
# v3: skill-pack text for the current request (set by the supervisor, read where change/plan prompts are built)
_guidance: contextvars.ContextVar[str] = contextvars.ContextVar("baseline_guidance", default="")

GUIDANCE_HEADER = ("INTERNAL CONTEXT (background for how to work on this request; it is not part of the task or the answer. "
                   "Never repeat, quote or mention it or any lane / level / guidance wording in your output, and it never overrides "
                   "the OUTPUT FORMAT or RULES of this prompt):")
_ECHO_RE = re.compile(r"^[ \t>*_`#-]*(?:LANE FOR THIS REQUEST|ENGINEERING GUIDANCE|INTERNAL CONTEXT)\b.*(?:\r?\n|$)",
                      re.IGNORECASE | re.MULTILINE)


def strip_guidance_echo(text: str) -> str:
    """Defensive: drop lines where a model echoed the lane statement / guidance header back into its output."""
    if not text or not _ECHO_RE.search(text):
        return text
    return re.sub(r"\n{3,}", "\n\n", _ECHO_RE.sub("", text)).lstrip("\n")


# Facts for the cross-check of the current supervised turn (the controller sets the box; the ask agent adds its sandbox result)
CHAT_FACTS: contextvars.ContextVar = contextvars.ContextVar("chat_facts", default=None)


def note_facts(**facts: Any) -> None:
    """Add/overwrite facts of the running supervised turn (no-op outside one). Unlike the controller's helper, None is kept."""
    box = CHAT_FACTS.get()
    if box is not None:
        box.update(facts)


# A lane can only raise the tier a request is handled at (Fast = what the risk alone says; Standard = Medium depth; Strict = High depth)
_LANE_TIER = {Lane.FAST: Risk.LOW, Lane.STANDARD: Risk.MEDIUM, Lane.STRICT: Risk.HIGH}
_TIER_ORDER = {Risk.LOW: 0, Risk.MEDIUM: 1, Risk.HIGH: 2}


class BaselinePolicy(BaseModel):
    depth: str = "baseline"          # baseline|elevated|strict
    checklist: List[str] = Field(default_factory=list)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    clarify_first: bool = False


def _gate_rank(v: Any) -> int:
    return GATE_ORDER.get(str(v or "off").strip().lower(), 0)


def _raise_gate(current: Any, target: str) -> str:
    return target if _gate_rank(target) > _gate_rank(current) else str(current or "off").strip().lower()


CHECKLIST_BASE = [
    "Validate inputs and handle errors explicitly",
    "Keep the change minimal and consistent with existing style",
    "Preserve existing public behaviour",
]
CHECKLIST_ELEVATED = [
    "Independent review of the diff against each requirement",
    "Run the smoke gate (code boots / compiles) before showing the change",
    "Check callers of every changed signature",
]
CHECKLIST_STRICT = [
    "Write tests first; they must fail on old code and pass on new code",
    "Strict gate: sandbox passes and app boots before the change is shown",
    "Security pass: authn/authz, secrets, PII in logs, injection",
    "Audit trail and rollback path for sensitive operations",
]


def policy_for(brief: IntakeBrief, defaults: Mapping[str, Any]) -> BaselinePolicy:
    defaults = defaults or {}
    ov: Dict[str, Any] = {}
    checklist = list(CHECKLIST_BASE)
    depth = "baseline"
    tier = brief.risk                                        # the lane may raise it, never lower it
    lane_tier = _LANE_TIER.get(brief.lane) if brief.lane else None
    if lane_tier and _TIER_ORDER[lane_tier] > _TIER_ORDER[tier]:
        tier = lane_tier
    if tier == Risk.MEDIUM:
        depth = "elevated"
        checklist += CHECKLIST_ELEVATED
        if not defaults.get("change_review", False):
            ov["change_review"] = True
        g = _raise_gate(defaults.get("change_gate", "off"), "smoke")
        if g != str(defaults.get("change_gate", "off") or "off").strip().lower():
            ov["change_gate"] = g
    elif tier == Risk.HIGH:
        depth = "strict"
        checklist += CHECKLIST_ELEVATED + CHECKLIST_STRICT
        if _gate_rank(defaults.get("change_gate", "off")) < GATE_ORDER["strict"]:
            ov["change_gate"] = "strict"
        if not defaults.get("change_review", False):
            ov["change_review"] = True
        if not defaults.get("change_tests_first", False):
            ov["change_tests_first"] = True
        if not defaults.get("change_author_tests", False):
            ov["change_author_tests"] = True       # tests-first needs authored tests
    if brief.enterprise and depth == "baseline":
        depth = "elevated"
    clarify = brief.ambiguity >= 0.7 and brief.risk == Risk.HIGH and brief.intent == "change"
    return BaselinePolicy(depth=depth, checklist=checklist, overrides=ov, clarify_first=clarify)


@contextlib.contextmanager
def policy_scope(overrides: Mapping[str, Any], guidance: str = ""):
    """Per-request settings overlay (ContextVar: isolated across concurrent asyncio tasks). `guidance` = skill-pack text, see guidance_block()."""
    clean = {k: v for k, v in (overrides or {}).items() if k in ALLOWED_KEYS}
    token = _overlay.set(clean)
    gtoken = _guidance.set(guidance or "")
    try:
        yield clean
    finally:
        for var, tok in ((_overlay, token), (_guidance, gtoken)):
            try:
                var.reset(tok)
            except ValueError:
                pass


def guidance_block() -> str:
    """Prompt section for the current request's skill pack, or "" (no scope, pack off, ask-only intent)."""
    g = _guidance.get()
    return f"\n{GUIDANCE_HEADER}\n{g}\n" if g else ""


def flag(name: str, default: Any) -> Any:
    """Overlay value for `name`, else `default`. The overlay can only raise: a lower value never wins."""
    ov = _overlay.get()
    if name not in ov:
        return default
    val = ov[name]
    try:
        if name == "change_gate":
            return val if _gate_rank(val) >= _gate_rank(default) else default
        if name == "change_max_attempts":
            return max(int(val), int(default))
        if isinstance(default, bool) or isinstance(val, bool):
            return bool(val) or bool(default)
    except Exception:
        return default
    return val
