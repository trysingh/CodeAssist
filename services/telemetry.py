import time
import json
import sys
import tiktoken
from typing import Any, Dict, List, Optional, Callable
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

def _cached_of(usage) -> int:
    """Cached prompt tokens a server reported: usage_metadata.input_token_details.cache_read, or OpenAI's
    prompt_tokens_details.cached_tokens (also Cloudflare's), or a flat cached_tokens. 0 when not reported."""
    if not isinstance(usage, dict):
        return 0
    for outer, inner in (("input_token_details", "cache_read"), ("prompt_tokens_details", "cached_tokens")):
        d = usage.get(outer)
        if isinstance(d, dict) and d.get(inner):
            try:
                return int(d[inner])
            except (TypeError, ValueError):
                return 0
    try:
        return int(usage.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


class StdoutTelemetryCallback(BaseCallbackHandler):
    """
    Traces every LLM & Embedding request and outputs structured telemetry to stdout.
    Uses Rust-backed tiktoken for fallback token counting when local providers omit metadata.
    """
    def __init__(self, model_name: str, repo_id: str,phase: str = "RAG_CHAT", emitter: Optional[Callable] = None,
                 requested_model: str = ""):
        self.model_name = model_name                # the id the runner really calls (make_chat relabels it)
        self.requested_model = requested_model      # the name the call site asked for (may be an Ollama-style default); "" = same
        self.repo_id = repo_id
        self.phase = phase
        self.emitter = emitter # Function to send data to frontend
        self.start_time = 0.0
        self.prompts: List[str] = []
        self.ledger = None                      # the question's TokenLedger (services/token_budget.py), when one is open
        self.runner, self.token_mode = "", ""
        try:
            from services.token_budget import current_ledger, effective_mode
            from services.runners import current_runner_name
            self.ledger, self.runner, self.token_mode = current_ledger(), current_runner_name(), effective_mode()
        except Exception:
            pass
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def _count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return max(1, len(text) // 4)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.start_time = time.perf_counter()
        self.prompts = prompts

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any) -> None:
        self.start_time = time.perf_counter()
        flat_msgs = []
        for msg_list in messages:
            for m in msg_list:
                flat_msgs.append(getattr(m, "content", str(m)))
        self.prompts = flat_msgs

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self.start_time
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0

        # Extract native Ollama / LangChain usage metadata if present
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) or getattr(gen, "generation_info", {})
                if isinstance(usage, dict):
                    input_tokens += usage.get("input_tokens") or usage.get("prompt_eval_count") or usage.get("prompt_tokens") or 0
                    output_tokens += usage.get("output_tokens") or usage.get("eval_count") or usage.get("completion_tokens") or 0
                    cached_tokens += _cached_of(usage)
                if not cached_tokens:           # OpenAI-shaped: response_metadata.token_usage.prompt_tokens_details.cached_tokens
                    cached_tokens += _cached_of((getattr(msg, "response_metadata", None) or {}).get("token_usage"))
        # OpenAI-shaped servers report the totals once per call in llm_output["token_usage"]
        if input_tokens == 0 and output_tokens == 0:
            tu = (getattr(response, "llm_output", None) or {}).get("token_usage")
            if isinstance(tu, dict):
                input_tokens = tu.get("prompt_tokens") or 0
                output_tokens = tu.get("completion_tokens") or 0
                cached_tokens = cached_tokens or _cached_of(tu)

        # Fallback to Rust-based tiktoken calculation
        if input_tokens == 0:
            input_tokens = sum(self._count_tokens(p) for p in self.prompts)
        if output_tokens == 0:
            for gen_list in response.generations:
                for gen in gen_list:
                    output_tokens += self._count_tokens(gen.text)

        total_tokens = input_tokens + output_tokens
        cached_tokens = min(int(cached_tokens or 0), input_tokens)
        saved_tokens = 0
        if self.ledger is not None:
            try:
                saved_tokens = self.ledger.record(
                    runner=self.runner, model=self.model_name, phase=str(self.phase), token_mode=self.token_mode,
                    input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_tokens).est_tokens_saved
            except Exception:
                pass

        telemetry_data = {
            "timestamp": time.strftime("%H:%M:%S"),
            "repo_id": self.repo_id,
            "phase": self.phase,
            "model": self.model_name,
            "requested_model": self.requested_model or self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "duration_sec": int(elapsed),
            "runner": self.runner,
            "token_mode": self.token_mode,
            "cached_input_tokens": cached_tokens,
            "est_tokens_saved": saved_tokens,
        }


        # trace_line = (
        #     f"\n[TELEMETRY TRACE] Phase={self.phase} | Model={self.model_name} | "
        #     f"InputTokens={input_tokens} | OutputTokens={output_tokens} | "
        #     f"TotalTokens={total_tokens} | TimeSpent={elapsed:.3f}s\n"
        # )
        # sys.stdout.write(trace_line)
        # sys.stdout.flush()

        # Keep stdout for logs, but emit for UI
        trace_line = f"[TELEMETRY] {telemetry_data}"
        sys.stdout.write(trace_line + "\n")
        sys.stdout.flush()
        
        try:       # one LLM action event per call (soft-fail)
            from services.action_log import log_action
            log_action(phase="LLM", agent=str(self.phase), action="llm_call", status="ok",
                       detail=f"{self.model_name}: {input_tokens} in / {output_tokens} out",
                       data={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
                             "requested_model": self.requested_model or self.model_name, "runner": self.runner, "token_mode": self.token_mode, "cached_input_tokens": cached_tokens,
                             "est_tokens_saved": saved_tokens},
                       model=self.model_name, duration_ms=int(elapsed * 1000), repo_id=self.repo_id or None)
        except Exception:
            pass

        if self.emitter:
            try:
                self.emitter(json.dumps(telemetry_data))
            except Exception:
                pass
