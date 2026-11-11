"""Claude subscription runner: spawns the official Claude Code CLI (`claude -p`), logged in by the user (`claude`, then /login, Pro / Max).

No API key. Deliberately no token handling: the app only runs the official binary and never reads ~/.claude (Anthropic's terms forbid
third-party apps using Claude subscription OAuth directly; running the user's own Claude Code CLI is the supported path; personal,
single-user use only). Settings: CLAUDE_CODE_BIN / _MODEL / _TIMEOUT_S / _MAX_INPUT_TOKENS. See docs/SUBSCRIPTION_RUNNERS.md.
"""
from __future__ import annotations

import re

from config.settings import settings
from services.runners.cli_core import Adapter, CLISubscriptionRunner, ClaudeCodeAdapter


class ClaudeCodeRunner(CLISubscriptionRunner):
    name = "claude_code"
    label = "Claude (subscription via Claude Code CLI)"
    cli_name = "Claude Code"
    timeout_setting = "CLAUDE_CODE_TIMEOUT_S"
    login_hint = "install it (npm i -g @anthropic-ai/claude-code), run `claude` once and log in with /login, or set CLAUDE_CODE_BIN."
    static_models = ["sonnet", "opus", "haiku"]
    vendor_model = re.compile(r"(claude-|opus|sonnet|haiku)", re.IGNORECASE)

    @property
    def max_input_tokens(self) -> int:      # type: ignore[override]
        return max(0, int(settings.claude_code_max_input_tokens))

    def binary(self) -> str:
        return (settings.claude_code_bin or "claude").strip()

    def model_list_setting(self) -> str:
        return settings.claude_code_models or ""

    def default_model(self) -> str:
        return settings.claude_code_model or ""

    def timeout_s(self) -> float:
        return float(settings.claude_code_timeout_s)

    def make_adapter(self, model: str, web: bool = False) -> Adapter:
        return ClaudeCodeAdapter(self.binary(), model, web)
