"""Client for Ollama decision models (tev1 and friends) via POST /v1/systemone.

Decision models are NOT chat models: you send `state` (the text to judge) and
named `questions` (choice / noul / score); they return probabilities, not prose.
So they get their own client, separate from services/model_calls.py (chat/JSON).

Design points
  * URL: settings.ollama_base_url_for_sys1_model may be a full ".../v1/systemone",
    a ".../v1" URL, or just the Ollama base URL; all are normalised. Empty falls
    back to settings.ollama_base_url.
  * Fail soft: any problem returns None so callers use their rule-based fallback.
  * Circuit breaker: after a failure (Ollama < 0.35, model not pulled, timeout)
    calls are skipped for settings.decision_cooldown_s instead of making every
    chat message wait on a dead endpoint.
  * Many questions per call are allowed by the API (up to 64) and share one
    prompt, so batch related questions about the same state in one `decide()`.
  * Keep `state` short: tev1 runs with roughly a 2K-token context; bodies <= 64 KiB.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import httpx

from config.settings import settings

State = Union[str, dict, list]
_MAX_BODY_BYTES = 60_000          # API limit is 64 KiB


def systemone_url(raw: Optional[str] = None) -> str:
    base = (raw or settings.ollama_base_url_for_sys1_model or settings.ollama_base_url).strip().rstrip("/")
    if base.endswith("/v1/systemone"):
        return base
    if base.endswith("/v1"):
        return base + "/systemone"
    return base + "/v1/systemone"


# ------------------------------------------------------------------ question builders
def q_noul(instructions: str, true: Optional[str] = None, false: Optional[str] = None) -> dict:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true and false:
        q["criteria"] = {"true": true, "false": false}
    return q


def q_choice(instructions: str, options: Dict[str, Optional[str]]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def q_score(instructions: str, levels: List[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


@dataclass
class Choice:
    choice: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ client
class DecisionClient:
    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._transport = transport            # tests inject httpx.MockTransport
        self._down_until = 0.0

    @staticmethod
    def _log(msg: str) -> None:
        sys.stdout.write(f"[DECISION] {msg}\n")

    def _trip(self, why: str) -> None:
        self._down_until = time.monotonic() + settings.decision_cooldown_s
        self._log(f"unavailable ({why}); using rules for {settings.decision_cooldown_s}s")

    async def decide(self, model: str, state: State, questions: Dict[str, dict], *,
                     phase: str = "DECIDE", repo_id: str = "") -> Optional[Dict[str, dict]]:
        """Returns {question_name: answer_dict} or None on any failure."""
        if not model or not questions:
            return None
        if time.monotonic() < self._down_until:
            return None
        if isinstance(state, str):
            state = state[: settings.decision_max_state_chars]
        body = {"model": model, "state": state, "questions": questions}
        if len(json.dumps(body).encode("utf-8")) > _MAX_BODY_BYTES:
            self._log(f"{phase}: request too large for the decision model, skipped")
            return None
        try:
            async with httpx.AsyncClient(
                    timeout=httpx.Timeout(settings.decision_timeout_s, connect=3.0),
                    transport=self._transport) as client:
                resp = await client.post(systemone_url(), json=body)
        except Exception as e:  # noqa: BLE001 — connection refused, timeout, DNS ...
            self._trip(f"{type(e).__name__}")
            return None
        if resp.status_code != 200:
            detail = resp.text[:160].replace("\n", " ")
            if resp.status_code in (404, 405) or resp.status_code >= 500:
                self._trip(f"HTTP {resp.status_code}: {detail}")      # old Ollama / model not pulled
            else:
                self._log(f"{phase} repo={repo_id}: HTTP {resp.status_code}: {detail}")
            return None
        try:
            answers = resp.json().get("answers")
        except Exception:  # noqa: BLE001
            return None
        return answers if isinstance(answers, dict) else None


decision_client = DecisionClient()


# ------------------------------------------------------------------ one-question helpers
async def decide_noul(model: str, state: State, instructions: str, *, true: Optional[str] = None,
                      false: Optional[str] = None, phase: str = "DECIDE", repo_id: str = "",
                      client: Optional[DecisionClient] = None) -> Optional[float]:
    """Probability (0..1) that the answer to a yes/no question is true."""
    answers = await (client or decision_client).decide(
        model, state, {"q": q_noul(instructions, true, false)}, phase=phase, repo_id=repo_id)
    try:
        return float(answers["q"]["noul"])
    except Exception:  # noqa: BLE001
        return None


async def decide_choice(model: str, state: State, instructions: str, options: Dict[str, Optional[str]], *,
                        phase: str = "DECIDE", repo_id: str = "",
                        client: Optional[DecisionClient] = None) -> Optional[Choice]:
    answers = await (client or decision_client).decide(
        model, state, {"q": q_choice(instructions, options)}, phase=phase, repo_id=repo_id)
    try:
        a = answers["q"]
        chosen = a["choice"]
        if chosen not in options:
            return None
        probs = a.get("probabilities") or {}
        return Choice(chosen, float(a.get("confidence", 0.0)),
                      probs if isinstance(probs, dict) else {})
    except Exception:  # noqa: BLE001
        return None
