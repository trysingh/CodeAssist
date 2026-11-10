"""Token mode: spend fewer (billed) tokens on remote runners, and tell the user what was spent.

mode      settings.token_mode ("auto" | "lean" | "normal"), overridable per request (token_mode_scope, the UI toggle).
          auto => lean when the request's runner is remote (Runner.remote: Cloudflare), normal for local runners (Ollama).
          normal is exactly the behaviour of the versions before token modes: no prompt differs by a byte.
budget()  the number a call site should use: the lean value (settings.lean_<name>, never above the normal one) in lean mode,
          settings.<name> otherwise. One helper, so a budget is changed in one place.
ledger    TokenLedger = the LLM calls of ONE question (runner, model, mode, tokens in / out / cached, estimated tokens saved),
          summed for the telemetry event and the footer line under the answer.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List, Optional

from config.settings import settings
from services.runners import current_runner_name, get_runner

MODES = ("auto", "lean", "normal")
# normal setting name -> the lean counterpart is settings.lean_<name>
BUDGETS = ("ask_max_steps", "ask_obs_chars", "ask_read_chars", "ask_seed_chars", "skill_pack_max_chars", "change_context_chars")

_mode: ContextVar[str] = ContextVar("token_mode", default="")


def normalize_mode(mode: Any) -> str:
    """'auto' | 'lean' | 'normal', or '' for anything else (empty, unknown)."""
    m = str(mode or "").strip().lower()
    return m if m in MODES else ""


@contextmanager
def token_mode_scope(mode: Optional[str]):
    """The user's token-mode pick for this request. Empty / invalid = keep the current one (settings.token_mode)."""
    m = normalize_mode(mode)
    if not m:
        yield
        return
    token = _mode.set(m)
    try:
        yield
    finally:
        _mode.reset(token)


def requested_mode() -> str:
    """The mode asked for (request pick, else settings.token_mode); an invalid setting counts as 'auto'."""
    return _mode.get() or normalize_mode(settings.token_mode) or "auto"


def runner_is_remote(name: Optional[str] = None) -> bool:
    try:
        return bool(getattr(get_runner(name), "remote", False))
    except Exception:  # noqa: BLE001 - an unusable runner is reported by the call itself; stay on today's behaviour
        return False


def effective_mode() -> str:
    """'lean' or 'normal' for the current request."""
    m = requested_mode()
    if m == "auto":
        return "lean" if runner_is_remote() else "normal"
    return m


def is_lean() -> bool:
    return effective_mode() == "lean"


def budget(name: str) -> int:
    """settings.<name>, or in lean mode min(settings.lean_<name>, settings.<name>)."""
    normal = getattr(settings, name)
    if name in BUDGETS and is_lean():
        lean = getattr(settings, "lean_" + name, None)
        if lean is not None:
            return min(int(lean), int(normal))
    return normal


def keep_full_obs() -> int:
    return max(0, int(settings.lean_keep_full_obs))


def max_input_tokens() -> int:
    """Estimated-token cap for one call (0 = off). Lean: MAX_INPUT_TOKENS if set, else the runner's own (Cloudflare 60000).
    Normal: only a remote runner AND an explicit MAX_INPUT_TOKENS (a safety net, never a surprise)."""
    explicit = settings.max_input_tokens
    if is_lean():
        if explicit is not None:
            return max(0, int(explicit))
        try:
            return max(0, int(getattr(get_runner(), "max_input_tokens", 0) or 0))
        except Exception:  # noqa: BLE001
            return 0
    if explicit is not None and runner_is_remote():
        return max(0, int(explicit))
    return 0


# --------------------------------------------------------------------------- estimates
def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def estimate_tokens(msgs: Iterable[Any]) -> int:
    """Cheap estimate: characters / 4 over every message."""
    return sum(len(text_of(getattr(m, "content", m))) for m in msgs) // 4


def trim_history(history: List[Any], keep: int) -> tuple:
    """(trimmed history, removed chars). `history` ends with the current question: it is always kept, plus at most `keep` messages
    before it (the newest). A kept slice never starts with an assistant reply."""
    from langchain_core.messages import HumanMessage
    msgs = list(history)
    cut = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=len(msgs) - 1)
    prior, current = msgs[:cut], msgs[cut:]
    if len(prior) <= max(0, keep):
        return msgs, 0
    kept = prior[len(prior) - keep:] if keep > 0 else []
    while kept and not isinstance(kept[0], HumanMessage):
        kept = kept[1:]
    dropped = prior[: len(prior) - len(kept)]
    return kept + current, sum(len(text_of(m.content)) for m in dropped)


# --------------------------------------------------------------------------- ledger
@dataclass
class CallRecord:
    runner: str
    model: str
    phase: str
    token_mode: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    est_tokens_saved: int = 0


@dataclass
class TokenLedger:
    """LLM calls of one question. Savings (chars never sent) accrue in `saved_chars`; each call takes what accrued since the last one."""
    calls: List[CallRecord] = field(default_factory=list)
    saved_chars: int = 0
    _pending: int = 0
    started: float = field(default_factory=time.monotonic)

    def add_saved(self, chars: int) -> None:
        chars = max(0, int(chars or 0))
        self.saved_chars += chars
        self._pending += chars

    def record(self, **kw: Any) -> CallRecord:
        rec = CallRecord(est_tokens_saved=self._pending // 4, **kw)
        self._pending = 0
        self.calls.append(rec)
        return rec

    def totals(self) -> dict:
        last = self.calls[-1] if self.calls else None
        return {"calls": len(self.calls),
                "input_tokens": sum(c.input_tokens for c in self.calls),
                "output_tokens": sum(c.output_tokens for c in self.calls),
                "cached_input_tokens": sum(c.cached_input_tokens for c in self.calls),
                "est_tokens_saved": self.saved_chars // 4,
                "runner": last.runner if last else current_runner_name(),
                "token_mode": last.token_mode if last else effective_mode()}


_ledger: ContextVar[Optional[TokenLedger]] = ContextVar("token_ledger", default=None)


@contextmanager
def ledger_scope():
    """A fresh TokenLedger for one question; the telemetry callbacks created inside it record into it."""
    led = TokenLedger()
    token = _ledger.set(led)
    try:
        yield led
    finally:
        _ledger.reset(token)


def current_ledger() -> Optional[TokenLedger]:
    return _ledger.get()


def note_saved(chars: int) -> None:
    """Chars that were NOT sent because of lean mode (compaction, trim, cap, budgets); counted in the question's ledger."""
    led = _ledger.get()
    if led is not None:
        led.add_saved(chars)


def fmt_tokens(n: int) -> str:
    n = int(n or 0)
    return str(n) if n < 500 else f"{n / 1000:.1f}k"


def footer_text(totals: dict) -> str:
    """`_Tokens: 18.2k in (6.1k cached) · 0.9k out · 6 calls · runner cloudflare · mode lean (saved ~31k)_` ('' = no calls)."""
    if not totals.get("calls"):
        return ""
    inp = fmt_tokens(totals["input_tokens"]) + " in"
    if totals.get("cached_input_tokens"):
        inp += f" ({fmt_tokens(totals['cached_input_tokens'])} cached)"
    n = totals["calls"]
    mode = f"mode {totals['token_mode']}"
    saved = int(totals.get("est_tokens_saved") or 0)
    if saved > 0:
        mode += f" (saved ~{round(saved / 1000)}k)" if saved >= 10000 else f" (saved ~{fmt_tokens(saved)})"
    return (f"_Tokens: {inp} · {fmt_tokens(totals['output_tokens'])} out · {n} call{'s' if n != 1 else ''} · "
            f"runner {totals['runner']} · {mode}_")


def summary_event(led: TokenLedger) -> dict:
    return {**led.totals(), "calls_detail": [asdict(c) for c in led.calls]}
