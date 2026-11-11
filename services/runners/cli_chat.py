"""LangChain chat model that answers by running an official vendor CLI (claude / codex) once per call. See cli_core.py for the logic and
for why only the official binary is executed (no OAuth token or credential file is ever read by this app)."""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from services.runners.cli_core import (Adapter, CLIReply, call_cli, compose_prompt, estimate_tokens, flatten_messages, run_subprocess,
                                       strip_json_fences)


class CLIChatModel(BaseChatModel):
    adapter: Any = None             # cli_core.Adapter: argv builder + output parser of one CLI
    executor: Any = None            # (argv, stdin, timeout_s, cwd) -> ExecResult; None = run_subprocess (tests inject a fake)
    model_name: str = ""
    runner_name: str = ""
    timeout_s: float = 300.0
    json_mode: bool = False
    max_tokens: Optional[int] = None
    vision: bool = False            # text only: image parts are dropped with a note
    web: bool = False               # informational: the adapter already carries the flag that enables the CLI's web tools

    @property
    def _llm_type(self) -> str:
        return f"cli-{self.runner_name or 'subscription'}"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "runner": self.runner_name}

    def _call_cli(self, messages: List[BaseMessage]) -> CLIReply:
        adapter: Adapter = self.adapter
        system, prompt = compose_prompt(*flatten_messages(messages), json_mode=self.json_mode, max_tokens=self.max_tokens)
        reply, used_model = call_cli(adapter, self.executor or run_subprocess, system, prompt, self.timeout_s)
        if used_model != (adapter.model or "default"):     # the CLI refused the picked model and its default answered instead
            reply.raw = dict(reply.raw or {}, used_model=used_model)
        if self.json_mode:
            reply.text = strip_json_fences(reply.text)
        if not reply.usage_known:       # no usage in the CLI output: chars/4 estimate so telemetry still counts something
            reply.input_tokens, reply.output_tokens = estimate_tokens(system + prompt), estimate_tokens(reply.text)
        return reply

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        reply = self._call_cli(messages)
        text = reply.text
        for s in stop or []:
            if s and s in text:
                text = text.split(s, 1)[0]
        usage = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                 "total_tokens": reply.input_tokens + reply.output_tokens}
        if reply.cache_read_tokens:
            usage["input_token_details"] = {"cache_read": reply.cache_read_tokens}
        used = reply.actual_model or (reply.raw or {}).get("used_model") or self.model_name
        meta = {"model_name": used, "runner": self.runner_name}
        if reply.actual_model:
            meta["actual_model"] = reply.actual_model       # telemetry relabels the call with it (instead of "default")
        msg = AIMessage(content=text, usage_metadata=usage, response_metadata=meta)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return await asyncio.to_thread(self._generate, messages, stop, None, **kwargs)
