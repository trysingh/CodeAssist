"""Pure logic of the subscription runners ("claude_code", "chatgpt_codex"): argv builders, output parsers, message flattening, error
classification and the shared Runner base class. NO langchain import here (cli_chat.py holds the LangChain class), so this module is
testable on its own.

DELIBERATE: the app only EXECUTES the vendors' official binaries (`claude`, `codex`), which the user logged into themselves. It never reads,
copies, stores or forwards their OAuth tokens or credential files (~/.claude, ~/.codex/auth.json). Anthropic's terms forbid third-party apps
from using a Claude subscription's OAuth directly; running the user's own Claude Code CLI is the supported path. Personal, single-user use only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.runners.base import RunnerConfigError

DEFAULT_MODEL = "default"           # what map_model returns when no model is configured: the CLI then picks its own (no model flag)
STDERR_TAIL = 500
# The Claude system prompt is written to a file in the per-call temp dir and passed with --system-prompt-file: no multi-line text in argv
# (Windows .cmd shims would mangle it, and Linux caps one argv string at 128 KiB).
SYSTEM_PROMPT_FILE = "system_prompt.txt"


# ------------------------------------------------------------------ errors
class CLIRunError(RuntimeError):
    """The CLI failed. `kind` is "auth" | "quota" | "other"; services/llm_errors maps auth -> ModelAuthError, quota -> ModelQuotaError."""

    def __init__(self, message: str, kind: str = "other", returncode: Optional[int] = None):
        super().__init__(message)
        self.cli_kind, self.returncode = kind, returncode
        self.status_code = {"auth": 401, "quota": 429}.get(kind)      # same duck-typed shape llm_errors reads on provider errors


class CLITimeoutError(TimeoutError):
    """The CLI did not finish in time (class name TimeoutError: llm_errors.is_timeout -> ModelTimeoutError)."""


_AUTH_RE = re.compile(r"not logged in|please run /login|run /login|\blogin required|invalid api key|authentication[_ ]error|unauthorized|"
                      r"please (?:log ?in|sign in)|codex login|not signed in|token (?:has )?expired|\b401\b", re.IGNORECASE)
_QUOTA_RE = re.compile(r"usage limit|rate limit|rate_limit|5-hour|weekly limit|limit reached|quota|too many requests|\b429\b|"
                       r"credits? (?:are |have )?(?:used|exhausted)|resets? (?:at|in)", re.IGNORECASE)


_UNSUPPORTED_MODEL_RE = re.compile(r"model is not supported|model (?:is )?not (?:available|found)|unknown model|invalid model|"
                                   r"does not have access to (?:the )?model|model .{0,60}? (?:is )?not supported", re.IGNORECASE)


def is_unsupported_model(text: str) -> bool:
    """The CLI refused the requested model (e.g. Codex: "The 'gpt-5' model is not supported when using Codex with a ChatGPT account")."""
    return bool(_UNSUPPORTED_MODEL_RE.search(text or ""))


def classify_failure(text: str) -> str:
    """"auth" | "quota" | "other" for the text a failing CLI printed (auth wins: a logged-out CLI says so before anything else)."""
    t = text or ""
    if _AUTH_RE.search(t):
        return "auth"
    if _QUOTA_RE.search(t):
        return "quota"
    return "other"


def tail(text: str, n: int = STDERR_TAIL) -> str:
    t = " ".join((text or "").split())
    return t[-n:]


def failure(cli: str, returncode: Optional[int], *texts: str) -> CLIRunError:
    """CLIRunError for a failed run; the message carries the tail of what the CLI printed (never the prompt)."""
    body = " | ".join(t for t in (tail(x) for x in texts) if t)[-STDERR_TAIL:]
    kind = classify_failure(body)
    head = f"{cli} failed" + (f" (exit {returncode})" if returncode is not None else "")
    return CLIRunError(f"{head}: {body}" if body else f"{head} without output", kind, returncode)


# ------------------------------------------------------------------ executor
@dataclass
class ExecResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Executor = Callable[[List[str], str, float, str], ExecResult]       # (argv, stdin_text, timeout_s, cwd) -> ExecResult


_NPM_SHIM_JS = re.compile(r'%~?dp0%?\\?([^"%\r\n]+?\.[cm]?js)\b', re.I)


def _npm_shim_target(cmd_path: str) -> Optional[str]:
    """The JS entry point an npm `.cmd` shim launches (`"%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*`), or None."""
    try:
        with open(cmd_path, encoding="utf-8", errors="replace") as fh:
            m = _NPM_SHIM_JS.search(fh.read())
    except OSError:
        return None
    if not m:
        return None
    js = os.path.normpath(os.path.join(os.path.dirname(cmd_path), m.group(1).replace("\\", os.sep)))
    return js if os.path.isfile(js) else None


def resolve_command(binary: str, *, windows: Optional[bool] = None) -> List[str]:
    """argv prefix that really starts `binary`. Windows: CreateProcess (shell=False) only finds .exe files by bare name, but npm installs
    `codex` / `claude` as `codex.cmd` / `claude.cmd` shims (shutil.which finds those via PATHEXT, so check_config passed while the start
    failed with WinError 2). A shim is run as `node <entry.js>` (no cmd.exe argument re-parsing); a shim we cannot read is run by full path."""
    win = (os.name == "nt") if windows is None else windows
    path = shutil.which(binary) or (binary if os.path.isfile(binary) else None)
    if not path:
        raise RunnerConfigError(f"Cannot start '{binary}': not found on PATH. Install the CLI or set its full path "
                                f"(CLAUDE_CODE_BIN / CODEX_BIN, docs/SUBSCRIPTION_RUNNERS.md).")
    if win and path.lower().endswith((".cmd", ".bat")):
        js = _npm_shim_target(path)
        if js:
            local_node = os.path.join(os.path.dirname(path), "node.exe")
            node = local_node if os.path.isfile(local_node) else shutil.which("node")
            if node:
                return [node, js]
    return [path]


def run_subprocess(argv: List[str], stdin: str, timeout_s: float, cwd: str) -> ExecResult:
    """Default executor: the real CLI. No shell, environment inherited (the CLI finds its own login there), prompt on stdin."""
    if argv:
        argv = resolve_command(argv[0]) + list(argv[1:])
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout_s or None, cwd=cwd, shell=False)
    except subprocess.TimeoutExpired as exc:        # subprocess.run has already killed the child
        raise CLITimeoutError(f"{argv[0] if argv else 'CLI'} did not finish within {timeout_s:g}s") from exc
    except FileNotFoundError as exc:
        raise RunnerConfigError(f"Cannot start '{argv[0] if argv else ''}': not found on PATH. Install the CLI (docs/SUBSCRIPTION_RUNNERS.md).") from exc
    return ExecResult(p.returncode, p.stdout or "", p.stderr or "")


# ------------------------------------------------------------------ message flattening (plain objects or dicts, no langchain)
_ROLES = {"system": "system", "human": "user", "user": "user", "ai": "assistant", "assistant": "assistant", "tool": "tool", "function": "tool"}
IMAGE_NOTE = "[an image was attached here and omitted: this runner is text-only]"


def _field(m: Any, *names: str) -> Any:
    for n in names:
        v = m.get(n) if isinstance(m, dict) else getattr(m, n, None)
        if v is not None:
            return v
    return None


def content_text(content: Any) -> str:
    """Text of a message content: a str, or a list of parts ({"type": "text", "text"} / image parts -> a note)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" or ("text" in part and "type" not in part):
                    out.append(str(part.get("text", "")))
                elif "image" in str(part.get("type", "")):
                    out.append(IMAGE_NOTE)
        return "\n".join(x for x in out if x)
    return str(content)


def flatten_messages(messages: List[Any]) -> Tuple[str, str]:
    """(system prompt, transcript). A lone user message is passed as is; with history the turns are labelled so the model sees the dialogue."""
    system: List[str] = []
    turns: List[Tuple[str, str]] = []
    for m in messages:
        role = _ROLES.get(str(_field(m, "type", "role") or "user").lower(), "user")
        text = content_text(_field(m, "content")).strip()
        if role == "system":
            if text:
                system.append(text)
        elif text:
            turns.append((role, text))
    if len(turns) == 1 and turns[0][0] == "user":
        return "\n\n".join(system), turns[0][1]
    labels = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL RESULT"}
    body = "\n\n".join(f"[{labels[r]}]\n{t}" for r, t in turns)
    if turns:
        body = ("Below is the conversation so far. Reply as the ASSISTANT to the last message, in the same style, "
                "without a role label.\n\n" + body)
    return "\n\n".join(system), body


JSON_INSTRUCTION = "Answer with a single JSON object only: no prose before or after it, no markdown fences."
_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_json_fences(text: str) -> str:
    m = _FENCE.match(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def compose_prompt(system: str, transcript: str, *, json_mode: bool = False, max_tokens: Optional[int] = None) -> Tuple[str, str]:
    """Adds the json / length instructions (the CLIs have no flag for either) to the system prompt."""
    extra = []
    if json_mode:
        extra.append(JSON_INSTRUCTION)
    if max_tokens:
        extra.append(f"Keep the answer under about {int(max_tokens)} tokens.")
    if extra:
        system = (system + "\n\n" if system else "") + " ".join(extra)
    return system, transcript


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ------------------------------------------------------------------ adapters: how to call one CLI and read its output
@dataclass
class CLIReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: Optional[float] = None
    usage_known: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)
    actual_model: str = ""          # the model that really answered, when the CLI (or its local session record) says so; "" = unknown


def _int(v: Any) -> int:
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 0


def _model_arg(model: str) -> List[str]:
    m = (model or "").strip()
    return [] if m in ("", DEFAULT_MODEL) else [m]


CLAUDE_WEB_TOOLS = "WebSearch,WebFetch"     # the only tools ever enabled: no Bash / Edit / Read


def build_claude_argv(binary: str, model: str = "", system_file: str = "", web: bool = False) -> List[str]:
    """`claude -p` as a plain completion engine: prompt on stdin, JSON result, all tools off, nothing persisted.
    web=True turns on exactly WebSearch + WebFetch (`--allowedTools` too, so -p mode does not stop for a permission prompt); slower and
    several turns instead of one. (`--max-turns` is not in `claude --help` of the installed version, so it is not passed.)"""
    argv = [binary, "-p", "--output-format", "json", "--tools", CLAUDE_WEB_TOOLS if web else "", "--no-session-persistence"]
    if web:
        argv += ["--allowedTools", CLAUDE_WEB_TOOLS]
    argv += (["--model"] + _model_arg(model)) if _model_arg(model) else []
    if system_file:
        argv += ["--system-prompt-file", system_file]
    return argv


def parse_claude_output(stdout: str) -> Tuple[CLIReply, bool]:
    """(reply, is_error) from `claude -p --output-format json`: {"result", "is_error", "usage", "total_cost_usd"}. Also accepts a JSON array
    of events (the last {"type": "result"} wins). Unparseable stdout is taken as the reply text."""
    raw = (stdout or "").strip()
    try:
        data: Any = json.loads(raw)
    except ValueError:
        return CLIReply(text=raw), False
    if isinstance(data, list):
        results = [d for d in data if isinstance(d, dict) and d.get("type") == "result"]
        data = results[-1] if results else {}
    if not isinstance(data, dict):
        return CLIReply(text=raw), False
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    cache_read = _int(usage.get("cache_read_input_tokens"))
    inp = _int(usage.get("input_tokens")) + cache_read + _int(usage.get("cache_creation_input_tokens"))
    cost = data.get("total_cost_usd")
    reply = CLIReply(text=str(data.get("result") or "").strip(), input_tokens=inp, output_tokens=_int(usage.get("output_tokens")),
                     cache_read_tokens=cache_read, cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                     usage_known=bool(usage), raw=data, actual_model=claude_actual_model(data))
    return reply, bool(data.get("is_error")) or str(data.get("subtype", "")).startswith("error")


def claude_actual_model(data: Dict[str, Any]) -> str:
    """The model that answered, from `modelUsage` ({"claude-sonnet-...": {"outputTokens": ...}, ...}) of `claude -p --output-format json`:
    the one with the most output tokens (a small helper model may appear too). '' when absent."""
    mu = data.get("modelUsage") if isinstance(data, dict) else None
    if not isinstance(mu, dict) or not mu:
        return ""
    def out(v: Any) -> int:
        return _int((v or {}).get("outputTokens") or (v or {}).get("output_tokens")) if isinstance(v, dict) else 0
    return str(max(mu, key=lambda k: out(mu[k])))


# ------------------------------------------------------------------ Codex: which model answered (its JSONL events do not say)
def codex_thread_id(stdout: str) -> str:
    """thread_id of `{"type":"thread.started","thread_id":...}` in `codex exec --json` output ('' if absent)."""
    for line in (stdout or "").splitlines():
        if '"thread.started"' in line:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict) and ev.get("thread_id"):
                return str(ev["thread_id"])
    return ""


def codex_home() -> str:
    return os.environ.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")


def codex_session_model(thread_id: str, home: Optional[str] = None) -> str:
    """Model recorded by Codex for this run in its own session log (~/.codex/sessions/YYYY/MM/DD/rollout-...-<thread_id>.jsonl, lines
    {"type": "turn_context", "payload": {"model": ...}}). Only those lines are parsed; never auth.json. '' when not found."""
    import glob
    if not thread_id or not re.fullmatch(r"[\w\-]{8,80}", thread_id):
        return ""
    try:
        files = glob.glob(os.path.join(home or codex_home(), "sessions", "*", "*", "*", f"*{thread_id}*.jsonl"))
    except OSError:
        return ""
    model = ""
    for path in files[:1]:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if '"turn_context"' not in line and '"session_meta"' not in line:
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    p = ev.get("payload") if isinstance(ev, dict) and isinstance(ev.get("payload"), dict) else {}
                    if p.get("model"):
                        model = str(p["model"])     # last one wins (a turn_context follows session_meta)
        except OSError:
            return ""
    return model


def codex_config_model(home: Optional[str] = None) -> str:
    """Top-level `model = "..."` of ~/.codex/config.toml (settings only, no credentials); '' when unset or unreadable."""
    path = os.path.join(home or codex_home(), "config.toml")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    try:
        import tomllib
        v = tomllib.loads(text).get("model")
        return str(v) if isinstance(v, str) else ""
    except Exception:  # noqa: BLE001 - old Python or a broken file: top-level line before the first [table]
        top = re.split(r"(?m)^\s*\[", text, maxsplit=1)[0]
        m = re.search(r'(?m)^\s*model\s*=\s*["\']([^"\']+)["\']', top)
        return m.group(1) if m else ""


CODEX_WEB_MODES = ("live", "cached")


def build_codex_argv(binary: str, model: str = "", last_message_file: str = "", web: bool = False, web_mode: str = "live") -> List[str]:
    """`codex exec` reading the prompt from stdin (`-`), read-only sandbox, JSONL events on stdout, final reply also written to a file.
    Codex's web search tool is ON by default (cached mode) even in a read-only sandbox, so it is always set explicitly:
    web=False -> `-c web_search="disabled"`, web=True -> `-c web_search="live"`.
    Kept in one small function: adjust here if a codex release renames a flag."""
    argv = [binary, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
            "-c", f'web_search="{web_mode if web_mode in CODEX_WEB_MODES else "live"}"' if web else 'web_search="disabled"']
    if last_message_file:
        argv += ["--output-last-message", last_message_file]
    argv += (["-m"] + _model_arg(model)) if _model_arg(model) else []
    return argv + ["-"]


def parse_codex_events(stdout: str) -> Tuple[str, int, int, int, bool, str]:
    """(last agent message, input, output, cached tokens, usage_known, error text) from `codex exec --json` JSONL. Unknown events and
    non-JSON lines are ignored. Understands item.completed/agent_message, turn.completed.usage, token_count events and error / turn.failed."""
    text, err = "", ""
    inp = out = cached = 0
    known = False
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        msg = ev.get("msg") if isinstance(ev.get("msg"), dict) else {}      # older wrapper: {"id", "msg": {"type", ...}}
        etype = str(ev.get("type") or msg.get("type") or "")
        item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        if item.get("type") == "web_search":       # a tool call (item.started / item.completed), never the reply
            continue
        if item.get("type") == "agent_message" and item.get("text"):
            text = str(item["text"])
        elif etype == "agent_message" and (msg.get("message") or ev.get("message")):
            text = str(msg.get("message") or ev.get("message"))
        usage = ev.get("usage") if isinstance(ev.get("usage"), dict) else None
        if usage is None and etype == "token_count":
            info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
            usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else None
        if usage:
            known = True
            inp, out = _int(usage.get("input_tokens")), _int(usage.get("output_tokens"))
            cached = _int(usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens"))
        if etype in ("error", "turn.failed"):
            e = ev.get("error")
            err = str((e.get("message") if isinstance(e, dict) else e) or ev.get("message") or msg.get("message") or err)
    return text.strip(), inp, out, cached, known, err


class Adapter:
    """How to call one CLI. `build(system, prompt, workdir) -> (argv, stdin)`; `parse(result, workdir) -> CLIReply` (raises CLIRunError)."""
    label = "CLI"

    def __init__(self, binary: str, model: str = "", web: bool = False):
        self.binary, self.model, self.web = binary, model, bool(web)

    def build(self, system: str, prompt: str, workdir: str) -> Tuple[List[str], str]:
        raise NotImplementedError

    def parse(self, res: ExecResult, workdir: str) -> CLIReply:
        raise NotImplementedError


class ClaudeCodeAdapter(Adapter):
    label = "Claude Code CLI"

    def build(self, system: str, prompt: str, workdir: str) -> Tuple[List[str], str]:
        system_file = ""
        if system:
            system_file = os.path.join(workdir, SYSTEM_PROMPT_FILE)
            with open(system_file, "w", encoding="utf-8") as fh:
                fh.write(system)
        return build_claude_argv(self.binary, self.model, system_file, self.web), prompt

    def parse(self, res: ExecResult, workdir: str) -> CLIReply:
        reply, is_error = parse_claude_output(res.stdout)
        if res.returncode != 0 or is_error:
            raise failure(self.label, res.returncode, reply.text if (is_error or res.returncode) else "", res.stderr)
        if not reply.text:
            raise failure(self.label, res.returncode, res.stderr, "empty reply")
        return reply


class CodexAdapter(Adapter):
    label = "Codex CLI"
    PREAMBLE = ("You are a plain text completion engine. Do not run commands, read files or use tools; answer the request directly "
                "from the text below.\n\n")
    PREAMBLE_WEB = ("You are a plain text completion engine. You may use web search for current information and must cite the URLs you "
                    "used, but do not run commands or read files; answer the request from the text below and what you find.\n\n")

    def build(self, system: str, prompt: str, workdir: str) -> Tuple[List[str], str]:
        body = (self.PREAMBLE_WEB if self.web else self.PREAMBLE) + (f"SYSTEM INSTRUCTIONS:\n{system}\n\n" if system else "") + prompt
        return build_codex_argv(self.binary, self.model, os.path.join(workdir, "last_message.txt"), self.web,
                                getattr(self, "web_mode", "live")), body

    def parse(self, res: ExecResult, workdir: str) -> CLIReply:
        text, inp, out, cached, known, err = parse_codex_events(res.stdout)
        try:
            with open(os.path.join(workdir, "last_message.txt"), encoding="utf-8", errors="replace") as fh:
                from_file = fh.read().strip()
        except OSError:
            from_file = ""
        text = from_file or text               # the file is the robust source; the JSONL message is the fallback
        if res.returncode != 0 or (err and not text):
            raise failure(self.label, res.returncode, err, res.stderr)
        if not text:
            raise failure(self.label, res.returncode, res.stderr, "empty reply")
        actual = self.model if self.model not in ("", DEFAULT_MODEL) else ""
        if not actual:              # no -m flag: Codex's events do not name the model, so ask its session log, then config.toml
            try:
                actual = codex_session_model(codex_thread_id(res.stdout)) or codex_config_model()
            except Exception:  # noqa: BLE001 - labelling only
                actual = ""
        return CLIReply(text=text, input_tokens=inp, output_tokens=out, cache_read_tokens=cached, usage_known=known, actual_model=actual)


def call_cli(adapter: Adapter, run: Executor, system: str, prompt: str, timeout_s: float) -> Tuple[CLIReply, str]:
    """Run the CLI once in a fresh temp dir -> (reply, model actually used). When the CLI refuses the requested model (the models a
    subscription may use change often), retry ONCE without a model flag so the CLI picks its own default; the reply then reports DEFAULT_MODEL."""
    import logging
    import tempfile

    def once(a: Adapter) -> CLIReply:
        with tempfile.TemporaryDirectory(prefix="codeassist-cli-") as workdir:     # cwd of the CLI: it never sees the project tree
            argv, stdin = a.build(system, prompt, workdir)
            return a.parse(run(argv, stdin, timeout_s, workdir), workdir)

    try:
        return once(adapter), adapter.model or DEFAULT_MODEL
    except CLIRunError as exc:
        if adapter.model in ("", DEFAULT_MODEL) or not is_unsupported_model(str(exc)):
            raise
        logging.getLogger(__name__).warning("%s refused model '%s'; retrying with the CLI's default model. Set a supported model "
                                            "(CODEX_MODEL / CLAUDE_CODE_MODEL) or pick Default in the UI. (%s)", adapter.label, adapter.model, exc)
        import copy
        fallback = copy.copy(adapter)         # same CLI, web flag and web mode; only the model flag is dropped
        fallback.model = ""
        return once(fallback), DEFAULT_MODEL


# ------------------------------------------------------------------ shared Runner base
class CLISubscriptionRunner:
    """Common Runner-protocol part of claude_code_runner / codex_runner. Subclasses set the class attributes and the settings accessors."""
    name = ""
    label = ""
    remote = True
    supports = {"vision": False, "json": True, "reasoning": False, "num_ctx": False, "embeddings": False, "web_search": True}
    timeout_setting = ""
    cli_name = ""                       # "claude" / "codex"
    login_hint = ""
    static_models: List[str] = []
    vendor_model = re.compile(r"$^")    # names of this vendor are passed through to the CLI

    def binary(self) -> str:
        raise NotImplementedError

    def default_model(self) -> str:
        raise NotImplementedError

    def make_adapter(self, model: str, web: bool = False) -> Adapter:
        raise NotImplementedError

    def model_list_setting(self) -> str:
        """Comma list from settings (CODEX_MODELS / CLAUDE_CODE_MODELS) that replaces static_models in the UI picker; '' = static_models."""
        return ""

    def timeout_s(self) -> float:
        raise NotImplementedError

    def max_retries(self) -> int:
        return 0                        # a retry would burn subscription quota; the quota / fallback logic decides instead

    def check_config(self) -> None:
        """Only looks for the binary on PATH (or the configured path): no subprocess, no network."""
        if not shutil.which(self.binary()):
            raise RunnerConfigError(f"The '{self.name}' runner needs the official {self.cli_name} CLI ('{self.binary()}' was not found): {self.login_hint}")

    def map_model(self, model: str, *, vision: bool = False) -> str:
        m = (model or "").strip()
        if m and self.vendor_model.match(m):
            return m
        return self.default_model().strip() or DEFAULT_MODEL

    def chat(self, model: str, *, temperature: float = 0, max_tokens: Optional[int] = None, json_mode: bool = False,
             reasoning: Any = None, num_ctx: Optional[int] = None, callbacks: Optional[list] = None,
             vision: bool = False, chat_cls: Any = None, web: bool = False) -> Any:
        if chat_cls is None:
            from services.runners.cli_chat import CLIChatModel as chat_cls      # lazy: needs langchain_core
        actual = self.map_model(model)
        kw: Dict[str, Any] = dict(adapter=self.make_adapter(actual, web=bool(web)), model_name=actual, runner_name=self.name,
                                  timeout_s=float(self.timeout_s() or 0), json_mode=json_mode, max_tokens=max_tokens, web=bool(web))
        if callbacks is not None:
            kw["callbacks"] = callbacks
        return chat_cls(**kw)           # temperature / reasoning / num_ctx: the CLIs expose no such flags

    def embeddings(self, model: Optional[str] = None, *, embeddings_cls: Any = None) -> Any:
        raise RunnerConfigError(f"The '{self.name}' runner has no embeddings; keep EMBEDDING_RUNNER=ollama.")

    async def list_models(self) -> List[str]:
        try:
            custom = [x.strip() for x in (self.model_list_setting() or "").split(",") if x.strip()]
            names = custom or list(self.static_models)
            d = self.default_model().strip()
            return names + ([d] if d and d not in names else [])
        except Exception:  # noqa: BLE001
            return list(self.static_models)
