import time
import sys
import tiktoken
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class StdoutTelemetryCallback(BaseCallbackHandler):
    """
    Traces every LLM & Embedding request and outputs structured telemetry to stdout.
    Uses Rust-backed tiktoken for fallback token counting when local providers omit metadata.
    """
    def __init__(self, model_name: str, phase: str = "RAG_CHAT"):
        self.model_name = model_name
        self.phase = phase
        self.start_time = 0.0
        self.prompts: List[str] = []
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

        # Extract native Ollama / LangChain usage metadata if present
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) or getattr(gen, "generation_info", {})
                if isinstance(usage, dict):
                    input_tokens += usage.get("input_tokens") or usage.get("prompt_eval_count") or 0
                    output_tokens += usage.get("output_tokens") or usage.get("eval_count") or 0

        # Fallback to Rust-based tiktoken calculation
        if input_tokens == 0:
            input_tokens = sum(self._count_tokens(p) for p in self.prompts)
        if output_tokens == 0:
            for gen_list in response.generations:
                for gen in gen_list:
                    output_tokens += self._count_tokens(gen.text)

        total_tokens = input_tokens + output_tokens
        trace_line = (
            f"\n[TELEMETRY TRACE] Phase={self.phase} | Model={self.model_name} | "
            f"InputTokens={input_tokens} | OutputTokens={output_tokens} | "
            f"TotalTokens={total_tokens} | TimeSpent={elapsed:.3f}s\n"
        )
        sys.stdout.write(trace_line)
        sys.stdout.flush()
