"""ChatGPT subscription runner: spawns the official OpenAI Codex CLI (`codex exec`), logged in by the user (`codex login`, Sign in with ChatGPT).

No API key. Deliberately no token handling: the app only runs the official binary and never reads ~/.codex/auth.json (personal,
single-user use only). Flags come from the documented `codex exec` options and are built in cli_core.build_codex_argv.
Settings: CODEX_BIN / _MODEL / _TIMEOUT_S / _MAX_INPUT_TOKENS. See docs/SUBSCRIPTION_RUNNERS.md.
"""
from __future__ import annotations

import re

from config.settings import settings
from services.runners.cli_core import Adapter, CLISubscriptionRunner, CodexAdapter


class CodexRunner(CLISubscriptionRunner):
    name = "chatgpt_codex"
    label = "ChatGPT (subscription via Codex CLI)"
    cli_name = "Codex"
    timeout_setting = "CODEX_TIMEOUT_S"
    login_hint = "install it (npm i -g @openai/codex), then run `codex login` (Sign in with ChatGPT), or set CODEX_BIN."
    # ChatGPT sign-in only accepts the current Codex models (gpt-5 / gpt-5-codex were removed for it). Override with CODEX_MODELS;
    # "default" (= no -m flag, the CLI's own default) always works.
    static_models = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]
    vendor_model = re.compile(r"(gpt-|o\d|codex)", re.IGNORECASE)

    @property
    def max_input_tokens(self) -> int:      # type: ignore[override]
        return max(0, int(settings.codex_max_input_tokens))

    def binary(self) -> str:
        return (settings.codex_bin or "codex").strip()

    def default_model(self) -> str:
        return settings.codex_model or ""

    def model_list_setting(self) -> str:
        return settings.codex_models or ""

    def timeout_s(self) -> float:
        return float(settings.codex_timeout_s)

    def make_adapter(self, model: str, web: bool = False) -> Adapter:
        a = CodexAdapter(self.binary(), model, web)
        mode = (settings.codex_web_mode or "live").strip().lower()
        a.web_mode = mode if mode in ("live", "cached") else "live"     # cached = OpenAI's index: fewer tokens, less injection risk, older
        return a
