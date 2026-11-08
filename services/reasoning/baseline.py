"""BaselinePolicy: risk-proportional strictness overlay. Only ever RAISES strictness."""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from .intake import IntakeBrief, Risk

GATE_ORDER = {"off": 0, "smoke": 1, "strict": 2}
ALLOWED_KEYS = ("change_gate", "change_review", "change_tests_first", "change_author_tests", "change_max_attempts")

_overlay: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("baseline_overlay", default={})


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
    if brief.risk == Risk.MEDIUM:
        depth = "elevated"
        checklist += CHECKLIST_ELEVATED
        if not defaults.get("change_review", False):
            ov["change_review"] = True
        g = _raise_gate(defaults.get("change_gate", "off"), "smoke")
        if g != str(defaults.get("change_gate", "off") or "off").strip().lower():
            ov["change_gate"] = g
    elif brief.risk == Risk.HIGH:
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
def policy_scope(overrides: Mapping[str, Any]):
    """Per-request settings overlay (ContextVar: isolated across concurrent asyncio tasks)."""
    clean = {k: v for k, v in (overrides or {}).items() if k in ALLOWED_KEYS}
    token = _overlay.set(clean)
    try:
        yield clean
    finally:
        try:
            _overlay.reset(token)
        except ValueError:
            pass


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
