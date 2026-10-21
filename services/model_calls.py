"""Small LLM helpers shared by the router and the change agents.

Every model (router, patcher, ...) is just a name in settings pointing at the
same Ollama endpoint. Callers receive these functions by injection, so tests
and future providers swap them without touching agent code.
"""
from __future__ import annotations
import asyncio
import json
import re
from typing import Awaitable, Callable, Optional
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from config.settings import settings

# (model, prompt, *, phase, repo_id, num_predict) -> text
LLMText = Callable[..., Awaitable[str]]
# (model, prompt, *, phase, repo_id) -> parsed dict or None
LLMJson = Callable[..., Awaitable[Optional[dict]]]


def _callbacks(model: str, repo_id: str, phase: str) -> list:
    try:
        from services.notifier import notifier
        from services.telemetry import StdoutTelemetryCallback
        return [StdoutTelemetryCallback(
            model_name=model, repo_id=repo_id, phase=phase,
            emitter=lambda data: notifier.notify("Telemetry Trace", data, level="INFO"))]
    except Exception:
        return []


def _build(model: str, *, json_mode: bool, num_predict: Optional[int], num_ctx: int,
           repo_id: str, phase: str):
    from langchain_ollama import ChatOllama
    kw = dict(model=model, base_url=settings.ollama_base_url_for_sys1_model, temperature=0,
              num_ctx=num_ctx, callbacks=_callbacks(model, repo_id, phase))
    if num_predict:
        kw["num_predict"] = num_predict     # never -1: cloud endpoints reject non-positive values
    if json_mode:
        kw["format"] = "json"
    return ChatOllama(**kw)


async def ask_text(model: str, prompt: str, *, phase: str = "CHANGE", repo_id: str = "",
                   num_predict: Optional[int] = None) -> str:
    llm = _build(model, json_mode=False, num_predict=num_predict or settings.change_max_output_tokens,
                 num_ctx=settings.change_num_ctx, repo_id=repo_id, phase=phase)
    resp = await llm.ainvoke(prompt)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def parse_json_loose(raw: str) -> Optional[dict]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```[\w-]*\s*|\s*```$", "", raw)
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                val = json.loads(m.group(0))
                return val if isinstance(val, dict) else None
            except Exception:
                return None
    return None


async def ask_json__OLD(model: str, prompt: str, *, phase: str = "DECIDE", repo_id: str = "") -> Optional[dict]:
    """Small, fast decision call. Any failure returns None so callers fall back to rules."""
    try:
        llm = _build(model, json_mode=True, num_predict=256, num_ctx=4096,
                     repo_id=repo_id, phase=phase)
        resp = await llm.ainvoke(prompt)
        return parse_json_loose(resp.content if isinstance(resp.content, str) else str(resp.content))
    except Exception:
        return None

# ==========================================
# ASK_JSON ADAPTER (SystemOne decision model)
# ==========================================
OLLAMA_SYSTEMONE_URL = settings.ollama_base_url_for_sys1_model
async def ask_json(model: str, prompt: str, *, phase: str = "DECIDE", repo_id: str = "") -> dict:
    """
    Adapter that executes against Ollama's /v1/systemone endpoint and returns
    the exact dict format expected by route_intent:
    {"intent": "ask" | "change", "confidence": float}
    """
    # Extract the user message if it was passed via _PROMPT.format(...)
    if "Message:" in prompt:
        raw_message = prompt.split("Message:", 1)[1].strip()
    else:
        raw_message = prompt.strip()

    payload = {
        "model": model,
        "state": raw_message[:2000],
        "questions": {
            "wants_change": {
                "type": "noul",
                "instructions": "Does the user want code or files modified, a bug fixed, or a feature built?",
                "criteria": {
                    "true": "The user wants code or files modified, a bug fixed, or a feature built.",
                    "false": "The user wants an explanation, search, review, or any read-only answer."
                }
            }
        }
    }

    def _make_request() -> dict:
        req = urllib.request.Request(
            OLLAMA_SYSTEMONE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                noul_score = res_data.get("answers", {}).get("wants_change", {}).get("noul")

                if noul_score is None:
                    return {"intent": "", "confidence": 0.0}

                score = float(noul_score)
                # noul >= 0.5 indicates 'true' (wants_change -> "change")
                # noul < 0.5 indicates 'false' (read-only -> "ask")
                if score >= 0.5:
                    intent = "change"
                    confidence = score
                else:
                    intent = "ask"
                    confidence = 1.0 - score

                return {
                    "intent": intent,
                    "confidence": round(confidence, 4)
                }
        except urllib.error.HTTPError as e:
            error_details = e.read().decode("utf-8")
            print(f"  [!] Ollama SystemOne HTTP Error ({phase} | repo={repo_id}): {error_details}")
            return {"intent": "", "confidence": 0.0}
        except Exception as e:
            print(f"  [!] Ollama SystemOne Connection Error ({phase} | repo={repo_id}): {e}")
            return {"intent": "", "confidence": 0.0}

    return await asyncio.to_thread(_make_request)