"""What the UI's runner / token-mode switches need, as plain functions (views/api_view.py only wraps them in routes)."""
from __future__ import annotations

from typing import Optional, Tuple

from config.settings import settings
from services.model_calls import list_chat_models
from services.runners import current_runner_name, require_runner, resolve_model, runner_scope, runner_status
from services.token_budget import MODES, normalize_mode


def runners_payload() -> dict:
    """GET /api/v1/runners: every registered runner (new ones appear by themselves) and whether it can be used right now;
    token_mode = the server's TOKEN_MODE (what the UI toggle shows until the user picks one)."""
    return {"default": settings.llm_runner, "runners": runner_status(), "token_mode": normalize_mode(settings.token_mode) or "auto"}


async def models_payload(runner: Optional[str] = None) -> dict:
    """GET /api/v1/models[?runner=x]: chat models of that runner (default: the configured one) and its default model.
    Unknown / unconfigured runner -> RunnerConfigError (the route answers 400)."""
    name = require_runner(runner)
    with runner_scope(name):
        names = list(await list_chat_models())
        default = resolve_model()       # Ollama: settings.llm_model; Cloudflare: an Ollama-style default maps to settings.cf_llm_model
        if default not in names:
            names.insert(0, default)
        return {"default": default, "models": names, "runner": current_runner_name()}


def check_chat_choice(runner: Optional[str], token_mode: Optional[str]) -> Tuple[str, str]:
    """Validate the per-request picks of a chat call -> (runner or '', token_mode or '').
    RunnerConfigError: unknown / not configured runner. ValueError: token_mode is not auto | lean | normal."""
    r = require_runner(runner)
    raw = (token_mode or "").strip()
    mode = normalize_mode(raw)
    if raw and not mode:
        raise ValueError(f"Unknown token_mode '{raw}'. Use one of: {', '.join(MODES)}.")
    return r, mode
