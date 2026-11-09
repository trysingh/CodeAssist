"""Decide whether a chat message is a question ("ask") or a code change ("change").

Order of authority:
  1. explicit prefix:  /ask ...   /plan ...   /change ...   /fix ...   (always wins, no model call)
  2. no prefix:        "ask" (settings.router_auto=False, the default). Steps 3-4 only run when router_auto=True
  3. decision model:   settings.router_model via Ollama /v1/systemone (services/decision.py)
  4. keyword rules:    always available; used if no model is set, it is unreachable, or unsure

The decision model answers ONE yes/no question and returns P(true). P >= 0.5 means
"change"; confidence is how far from a coin flip it is, and must reach
settings.router_min_confidence, otherwise the rules decide.

"change" only creates a PROPOSAL (nothing is written to the repo on disk), so a
wrong guess costs one wasted generation, not a damaged file. Ambiguous -> "ask".
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from config.settings import settings
from services.decision import decide_noul

_PREFIX = re.compile(r"^\s*/(ask|change|fix|edit|implement|plan)\b[:\s]*", re.IGNORECASE)

_VERBS = (r"fix|implement|add|create|build|write|refactor|rename|remove|delete|change|update|modify|"
          r"rewrite|patch|migrate|convert|optimi[sz]e|replace|extend|integrate|support|resolve|handle|"
          r"introduce|move|split|merge|wire|hook")
_LEAD = r"(?:please\s+|pls\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:need|want)\s+(?:you\s+)?to\s+|help\s+me\s+(?:to\s+)?|let'?s\s+)?"
_STRONG_CHANGE = re.compile(rf"^\s*{_LEAD}(?:{_VERBS})\b", re.IGNORECASE)
_BUG_FIX = re.compile(r"\b(fix|resolve|solve|debug)\b.*\b(bug|error|issue|exception|traceback|crash|failing|broken)\b|"
                      r"\b(bug|error|exception|traceback)\b.*\b(fix|resolve|solve)\b", re.IGNORECASE)
_STRONG_ASK = re.compile(
    r"^\s*(?:what|why|how|where|which|who|when|explain|describe|summari[sz]e|show|list|tell|review|"
    r"is|are|does|do|did|walk)\b", re.IGNORECASE)


_HELP_PREFIX = re.compile(r"^\s*/(help|commands|\?)\s*$", re.IGNORECASE)
_HELP_ASK = re.compile(r"\bcommands?\b", re.IGNORECASE)
_HELP_HINT = re.compile(r"\bslash\b|/ask\b|/change\b", re.IGNORECASE)
_ASK_GUARD = 0.9    # read-only opener needs a model "change" at least this sure

COMMANDS_HELP = """**Commands** (start your message with one):

- `/ask <question>`: read-only answer, explanation or search
- `/change <request>`: propose a code change (nothing is written until you apply it)
- `/fix <bug>`: same as `/change`, aimed at a bug
- `/edit <request>`: same as `/change`
- `/implement <feature>`: same as `/change`, aimed at a new feature
- `/plan <request>`: show the files and steps a change would need, without building it
- `/web <question>`: answer using a live web search (works with or without a repository)
- `/help` (or `/commands`): show this list

Without a command, your message is a read-only `/ask`: the assistant searches and reads the
code (find, grep, full files) before answering, and checks any fix it suggests in the sandbox.
It uses the web only when needed (see the Web chip next to the message box)."""


_KNOWN_COMMANDS = {"ask", "change", "fix", "edit", "implement", "plan", "web", "help", "commands"}
_UNKNOWN_CMD = re.compile(r"^\s*/([A-Za-z][\w-]*)(?=\s|$)")      # "/word" then space or end; "/home/x" and "/v1/api" are paths, not commands


def unknown_command(query: str) -> Optional[str]:
    """The word of a leading "/word" that is not a known command, else None."""
    m = _UNKNOWN_CMD.match(query or "")
    return m.group(1) if m and m.group(1).lower() not in _KNOWN_COMMANDS else None


def unknown_command_reply(word: str) -> str:
    return (f"Unknown command `/{word}`. Did you mean `/change` (propose a revision) or `/ask`?\n\n" + COMMANDS_HELP)


def is_help(query: str) -> bool:
    q = query.strip()
    return bool(_HELP_PREFIX.match(q) or (_STRONG_ASK.match(q) and _HELP_ASK.search(q) and _HELP_HINT.search(q)))

_Q = "Does the user want code or files modified, a bug fixed, or a feature built?"
_TRUE = "The user wants code or files modified, a bug fixed, or a feature built."
_FALSE = "The user wants an explanation, search, review, or any read-only answer."

NoulFn = Callable[..., Awaitable[Optional[float]]]


@dataclass
class RouteDecision:
    intent: str            # "ask" | "change"
    query: str             # message with any /prefix removed
    source: str            # prefix | model | rules
    confidence: float
    reply: str = ""        # "help" only: text to show instead of the plain command list (e.g. an unknown-command hint)


def route_by_rules(query: str) -> RouteDecision:
    q = query.strip()
    if _STRONG_CHANGE.match(q) or _BUG_FIX.search(q) and not _STRONG_ASK.match(q):
        return RouteDecision("change", q, "rules", 0.85)
    if _STRONG_ASK.match(q) or q.endswith("?"):
        return RouteDecision("ask", q, "rules", 0.85)
    return RouteDecision("ask", q, "rules", 0.5)


async def route_intent(query: str, *, noul_fn: NoulFn = decide_noul,
                       model: Optional[str] = None, repo_id: str = "") -> RouteDecision:
    m = _PREFIX.match(query)
    if is_help(query):
        return RouteDecision("help", query.strip(), "rules", 1.0)
    if m:
        word = m.group(1).lower()
        intent = "ask" if word == "ask" else "plan" if word == "plan" else "change"
        return RouteDecision(intent, query[m.end():].strip(), "prefix", 1.0)
    bad = unknown_command(query)
    if bad:         # never guess: "/apply ..." must not silently become a read-only ask. No model call.
        return RouteDecision("help", query.strip(), "rules", 1.0, reply=unknown_command_reply(bad))

    q = query.strip()
    if not settings.router_auto:           # default: nothing is changed or planned unless the user typed /change, /fix, /plan ...
        return RouteDecision("ask", q, "default", 1.0)
    model = settings.router_model if model is None else model
    if model:
        p = await noul_fn(model, q, _Q, true=_TRUE, false=_FALSE, phase="ROUTER", repo_id=repo_id)
        if p is not None:
            intent, conf = ("change", p) if p >= 0.5 else ("ask", 1.0 - p)
            sys.stdout.write(f"[ROUTER] {intent} (conf={conf:.2f}) via {model}\n")
            if intent == "change" and _STRONG_ASK.match(q) and conf < _ASK_GUARD:
                return route_by_rules(q)
            if conf >= settings.router_min_confidence:
                return RouteDecision(intent, q, "model", round(conf, 4))
    return route_by_rules(q)