"""Decide whether a chat message is a question ("ask") or a code change ("change").

Order of authority:
  1. explicit prefix:  /ask ...   /change ...   /fix ...   (always wins, no model call)
  2. decision model:   settings.router_model on Ollama, JSON out, used when confident
  3. keyword rules:    always available, used if no model is configured or it fails

"change" only creates a PROPOSAL (nothing is written to the repo on disk), so a
wrong guess costs one wasted generation, not a damaged file. Ambiguous -> "ask".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config.settings import settings
from services.model_calls import LLMJson, ask_json

_PREFIX = re.compile(r"^\s*/(ask|change|fix|edit|implement)\b[:\s]*", re.IGNORECASE)

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


@dataclass
class RouteDecision:
    intent: str            # "ask" | "change"
    query: str             # message with any /prefix removed
    source: str            # prefix | model | rules
    confidence: float


def route_by_rules(query: str) -> RouteDecision:
    q = query.strip()
    if _STRONG_CHANGE.match(q) or _BUG_FIX.search(q) and not _STRONG_ASK.match(q):
        return RouteDecision("change", q, "rules", 0.85)
    if _STRONG_ASK.match(q) or q.endswith("?"):
        return RouteDecision("ask", q, "rules", 0.85)
    return RouteDecision("ask", q, "rules", 0.5)


_PROMPT = """You route messages sent to a code-repository assistant.
"change" = the user wants code or files modified, a bug fixed, or a feature built.
"ask"    = the user wants an explanation, search, review, or any read-only answer.
Reply with ONLY JSON: {{"intent": "ask" or "change", "confidence": 0.0-1.0}}

Message: {message}"""


async def route_intent(query: str, *, llm_json: LLMJson = ask_json,
                       model: Optional[str] = None, repo_id: str = "") -> RouteDecision:
    m = _PREFIX.match(query)
    if m:
        word = m.group(1).lower()
        return RouteDecision("ask" if word == "ask" else "change", query[m.end():].strip(), "prefix", 1.0)

    model = settings.router_model if model is None else model
    if model:
        data = await llm_json(model, _PROMPT.format(message=query.strip()[:2000]),
                              phase="ROUTER", repo_id=repo_id)
        intent = str((data or {}).get("intent", "")).strip().lower()
        try:
            conf = float((data or {}).get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
            
        print(f"Intent: {intent} (conf={conf:.2f}) from model {model}")
        if intent in ("ask", "change") and conf >= settings.router_min_confidence:
            return RouteDecision(intent, query.strip(), "model", conf)
    return route_by_rules(query)
