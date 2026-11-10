"""Ask-mode agent: find / grep / ls / read tools over the raw store, then a sandbox check of any proposed edit.

Why: the single-shot answer sees 4 vector hits plus their neighbours, so the model fills the gaps with guesses.
Here the model searches the repo itself (exact text, whole files, line numbers) before it answers. If the answer
proposes code, the SEARCH/REPLACE blocks are applied to a COPY of your files (the quoted lines must exist exactly),
statically checked and run in the same pre-warmed container the /change pipeline uses. Failures go back to the model.
Nothing is written to your disk and no revision is saved: the answer only reports what the check found.

The model replies with one `ACTION: {"tool": ..., "args": {...}}` line per turn (works with any Ollama model, no native
tool-calling needed); a reply without an ACTION line is the final answer.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import settings
from services.action_log import action as _action, log_action as _log_action
from services import call_check, static_check
from services.model_calls import parse_json_loose
from services.patch_service import Edit, apply_edits, parse_edits, resolve_edit_paths
from services.reasoning.baseline import guidance_block, note_facts
from services.reasoning.crosscheck import facts_from_sandbox
from services.sandbox_service import is_test_file
from services.token_budget import (budget, current_ledger, effective_mode, estimate_tokens, footer_text, is_lean, keep_full_obs,
                                   ledger_scope, max_input_tokens, note_saved, summary_event, text_of, trim_history)

PROTOCOL = """

=== HOW TO WORK (read-only investigation) ===
The snippets above are only LEADS from a semantic search: they can be partial and unrelated. You see the repository
through these tools. Use exactly ONE per turn, as a single line:
ACTION: {"tool": "grep", "args": {"pattern": "def login", "path": "services/*.py"}}

Tools:
- find {"pattern": "*auth*"}: file paths matching a glob or part of a name
- grep {"pattern": "<regex>", "path": "<optional glob or folder>", "context": 1}: matching lines with line numbers
- ls {"path": "services"}: one folder level
- read {"path": "services/auth.py", "start": 1, "end": 0}: the file with line numbers (end 0 = to the end). A long file comes back
  TRUNCATED: call read again with the start line it names, until you have seen all of it.

Rules:
1. Never say something is missing, unused or "probably" so until a grep for it returned nothing. Never explain or change a file you have not read completely.
2. Bug: find where the symptom shows up, read those files completely, grep the callers and usages, then state the root cause with `path:line`.
3. Feature: grep for the closest existing implementation, read it, follow its conventions, grep what the new code must call.
4. Cite as `path:line`. Say plainly what you could not verify.
5. When you have enough evidence, answer in markdown with NO ACTION line.
6. When you propose code, write each change as an edit block in this exact layout:
   - Line 1: `FILE: <exact path from the repo>` as plain text, not inside a fence.
   - Line 2: an opening markdown fence with the language, e.g. ```python
   - `<<<<<<< SEARCH` starts the block. Below it, paste the lines that exist TODAY, copied exactly from `read` output without the `12| ` prefix, keeping indentation.
   - `=======` separates the old lines from the new lines.
   - `>>>>>>> REPLACE` ends the block. Below it, write the new lines that replace the SEARCH text.
   - The closing fence ``` comes last, after `>>>>>>> REPLACE`.
   - A new file: leave the SEARCH part empty (`<<<<<<< SEARCH` directly followed by `=======`).
   - One FILE line and one fenced block per change. Several blocks per answer are fine.
   Never write the markers without the fence. Example:

FILE: services/auth.py
```python
<<<<<<< SEARCH
    if token is None:
        return None
=======
    if not token:
        return None
>>>>>>> REPLACE
```
   Your edit blocks are applied to a copy of the repo and run in a sandbox before the user sees the answer: fix what it reports.
"""

_ACTION = re.compile(r"ACTION\s*:\s*", re.I)
# Gemma-4 style native tool call, which some runners (e.g. Cloudflare Workers AI) pass through as plain text:
#   <|tool_call>call:read{path: "services/x.py", start: 1, end: 185}<tool_call|>
# Strings may also be wrapped in the <|"|> token instead of quotes. The closing tag is optional (the reply can be cut off).
_NATIVE_CALL = re.compile(r"<\|tool_call\|?>\s*(?:call\s*:\s*)?([\w.\-]+)\s*(\{.*?\})?\s*(?:<\|?tool_call\|>|$)", re.S)
_NATIVE_MARK = re.compile(r"<\|tool_call\|?>")
_BARE_KEY = re.compile(r'([{,]\s*)([A-Za-z_][\w\-]*)\s*:')
_NUM_PREFIX = re.compile(r"^ *\d+\|(?: |$)")
_NESTED_QUANT = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*[+*{]")      # (a+)+ style: catastrophic backtracking
_TOOLS = ("find", "grep", "ls", "read")


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _match_path(path: str, pat: str) -> bool:
    pat = (pat or "").strip().lower().replace("\\", "/")
    if not pat:
        return True
    low = path.lower()
    return pat in low or fnmatch.fnmatchcase(low, pat) or fnmatch.fnmatchcase(low.rsplit("/", 1)[-1], pat)


# --------------------------------------------------------------------------- tools
class RepoTools:
    """Read-only find/grep/ls/read over ONE repo snapshot ({path: bytes}). Never touches the disk; every path is a dict key."""

    def __init__(self, tree: Dict[str, bytes]):
        self.tree = tree
        self._text: Dict[str, str] = {}
        self._split: Dict[str, List[str]] = {}
        self._seen: Dict[str, List[Tuple[int, int]]] = {}      # path -> line ranges the model has READ (grep context does not count)
        self._memo: Dict[str, str] = {}
        self.calls: Dict[str, int] = {}
        self.chars = 0                                         # tool output handed to the model so far

    # -- helpers
    def text(self, path: str) -> str:
        t = self._text.get(path)
        if t is None:
            t = self._text[path] = self.tree[path].decode("utf-8", "replace").replace("\r\n", "\n")
        return t

    def _lines(self, path: str) -> List[str]:
        lines = self._split.get(path)
        if lines is None:
            lines = self.text(path).split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            self._split[path] = lines
        return lines

    def resolve(self, path: str) -> Tuple[Optional[str], str]:
        p = (path or "").strip().strip("`'\"").replace("\\", "/")
        p = p[2:] if p.startswith("./") else p
        if not p:
            return None, "ERROR: read needs a path"
        if p in self.tree:
            return p, ""
        low = p.lower()
        hits = [k for k in self.tree if k.lower() == low] or \
               [k for k in self.tree if k.lower().endswith("/" + low)] or \
               [k for k in self.tree if k.lower().rsplit("/", 1)[-1] == low]
        if len(hits) == 1:
            return hits[0], ""
        if hits:
            return None, f"ERROR: '{p}' is ambiguous: {', '.join(sorted(hits)[:8])}. Use the full path."
        return None, f"ERROR: no file '{p}'. Use find or ls to see the real paths."

    def _mark(self, path: str, a: int, b: int) -> None:
        self._seen.setdefault(path, []).append((a, b))

    def covers(self, path: str, line: int) -> bool:
        return any(a <= line <= b for a, b in self._seen.get(path, []))

    def fully_read(self, path: str) -> bool:
        n, reach = len(self._lines(path)), 0
        for a, b in sorted(self._seen.get(path, [])):
            if a > reach + 1:
                return False
            reach = max(reach, b)
        return reach >= n

    def files_read(self) -> List[str]:
        return sorted(self._seen)

    def unread(self, path: str, line: int) -> Optional[str]:
        """Why an edit to `path` (at `line`) is not backed by reading, or None when it is."""
        if path not in self.tree:
            return None
        n = len(self.tree[path])
        if n <= settings.ask_full_read_max_chars:
            if not self.fully_read(path):
                return f"`{path}`: you edit it but have not read it completely. Read the whole file first (read, until nothing is TRUNCATED)."
        elif not self.covers(path, line):
            return f"`{path}` is large: you edit line {line} but never read that part. Read the code around it first."
        return None

    # -- tools
    def find(self, pattern: str = "", limit: int = 80) -> str:
        hits = [p for p in sorted(self.tree) if _match_path(p, pattern)]
        if not hits:
            return f"No file matches '{pattern}' ({len(self.tree)} files in the repo). Try a shorter name or ls."
        more = f"\n... {len(hits) - limit} more: use a stricter pattern" if len(hits) > limit else ""
        return f"{len(hits)} file(s):\n" + "\n".join(hits[:limit]) + more

    def ls(self, path: str = "") -> str:
        d = (path or "").strip().strip("/").replace("\\", "/")
        prefix = d + "/" if d else ""
        dirs: Dict[str, int] = {}
        files: List[Tuple[str, int]] = []
        for p in sorted(self.tree):
            if not p.startswith(prefix):
                continue
            rest = p[len(prefix):]
            if "/" in rest:
                dirs[rest.split("/", 1)[0]] = dirs.get(rest.split("/", 1)[0], 0) + 1
            else:
                files.append((rest, len(self.tree[p])))
        if not dirs and not files:
            return f"No folder '{d}'. ls with an empty path shows the top level."
        rows = [f"{k}/  ({v} files)" for k, v in sorted(dirs.items())] + [f"{k}  ({v} bytes)" for k, v in files]
        return f"/{d}\n" + "\n".join(rows)

    def grep(self, pattern: str, path: str = "", ignore_case: bool = True, context: int = 1) -> str:
        pattern = str(pattern or "")
        if not pattern:
            return "ERROR: grep needs a pattern"
        if len(pattern) > 200 or _NESTED_QUANT.search(pattern):
            return "ERROR: pattern too long or too complex. Use a simpler regex."
        flags = re.I if ignore_case else 0
        note = ""
        try:
            rx = re.compile(pattern, flags)
        except re.error as e:
            rx, note = re.compile(re.escape(pattern), flags), f"(not a valid regex: {e}; searched as plain text)\n"
        ctx = max(0, min(_int(context, 1), 3))
        limit, cap = settings.ask_grep_max, 9000
        deadline = time.monotonic() + 5.0
        searched = files_hit = total = shown = chars = 0
        out: List[str] = []
        cut = False
        for p in sorted(self.tree):
            if not _match_path(p, path):
                continue
            searched += 1
            lines = self._lines(p)
            hits = [i for i, ln in enumerate(lines) if len(ln) < 2000 and rx.search(ln)]
            if not hits:
                continue
            files_hit += 1
            total += len(hits)
            if shown < limit and chars < cap:
                take = hits[: limit - shown]
                keep = sorted({j for i in take for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1))})
                hit_set, prev = set(take), None
                for j in keep:
                    if prev is not None and j != prev + 1:
                        out.append("--")
                    row = f"{p}{':' if j in hit_set else '-'}{j + 1}{':' if j in hit_set else '-'} {lines[j][:300]}"
                    out.append(row)
                    chars += len(row) + 1
                    prev = j
                shown += len(take)
            else:
                cut = True
            if time.monotonic() > deadline:
                cut = True
                break
        if not total:
            return f"{note}No match for /{pattern}/ in {searched} file(s) searched" + (f" (path filter '{path}')" if path else "") + "."
        head = f"{note}{total} match(es) in {files_hit} file(s)"
        if cut or shown < total:
            head += f"; showing the first {shown}. Narrow it with path= or a stricter pattern"
        return head + ":\n" + "\n".join(out)

    def read(self, path: str, start: int = 1, end: int = 0) -> str:
        p, err = self.resolve(path)
        if not p:
            return err
        lines = self._lines(p)
        n = len(lines)
        a = max(1, _int(start, 1))
        b = _int(end, 0)
        b = n if b <= 0 else min(b, n)
        if a > n:
            return f"{p} has only {n} lines."
        width, used, last, rows = len(str(n)), 0, a - 1, []
        cap = budget("ask_read_chars")
        for i in range(a, b + 1):
            row = f"{i:>{width}}| {lines[i - 1][:2000]}"
            if rows and used + len(row) + 1 > cap:
                break
            rows.append(row)
            used += len(row) + 1
            last = i
        if cap < settings.ask_read_chars and last < b:      # lean: chars this read would have returned under the normal budget
            extra = 0
            for i in range(last + 1, b + 1):
                row = f"{i:>{width}}| {lines[i - 1][:2000]}"
                if used + extra + len(row) + 1 > settings.ask_read_chars:
                    break
                extra += len(row) + 1
            note_saved(extra)
        self._mark(p, a, last)
        head = f"FILE {p} ({n} lines), showing {a}-{last}"
        if last < b:
            head += f" TRUNCATED: more follows, call read again with start={last + 1}"
        elif a == 1 and last == n:
            head += " (COMPLETE FILE)"
        return head + "\n" + "\n".join(rows)

    def run(self, name: str, args: dict) -> str:
        """One tool call -> text. Never raises; repeated identical calls are answered with a pointer, not re-run."""
        name = (name or "").strip().lower()
        if name not in _TOOLS:
            return f"ERROR: unknown tool '{name}'. Tools: {', '.join(_TOOLS)}."
        args = args if isinstance(args, dict) else {}
        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in self._memo:
            return "You already ran exactly this call; use that result (it is above) or try something different."
        self.calls[name] = self.calls.get(name, 0) + 1
        try:
            if name == "find":
                out = self.find(args.get("pattern") or args.get("name") or args.get("path") or "")
            elif name == "ls":
                out = self.ls(args.get("path") or "")
            elif name == "grep":
                out = self.grep(args.get("pattern") or args.get("regex") or args.get("query") or "", args.get("path") or args.get("glob") or "",
                                args.get("ignore_case", True) not in (False, "false", 0), args.get("context", 1))
            else:
                out = self.read(args.get("path") or args.get("file") or "", args.get("start", 1), args.get("end", 0))
        except Exception as e:  # noqa: BLE001 - a tool bug must not end the chat
            out = f"ERROR: {name} failed: {type(e).__name__}: {e}"
        self._memo[key] = out
        self.chars += len(out)
        return out


# --------------------------------------------------------------------------- parsing helpers
def _native_args(raw: str) -> Optional[dict]:
    """Arguments of a native tool call: unquoted keys and <|"|> string tokens made into JSON."""
    s = (raw or "{}").replace('<|"|>', '"')
    d = parse_json_loose(s)
    if not isinstance(d, dict):
        d = parse_json_loose(_BARE_KEY.sub(r'\1"\2":', s))
    return d if isinstance(d, dict) else None


def strip_tool_calls(reply: str) -> str:
    """The reply without any tool-call text (ACTION line or native call), for when no more tool calls are allowed."""
    reply = _ACTION.split(reply or "")[0]
    return _NATIVE_MARK.split(reply)[0].strip()


def parse_action(reply: str) -> Optional[Tuple[str, dict, str]]:
    """(tool, args, error) when the reply is a tool call; None when it is a final answer."""
    reply = reply or ""
    if not _ACTION.search(reply) and _NATIVE_MARK.search(reply):
        n = _NATIVE_CALL.search(reply)
        args = _native_args(n.group(2)) if n else None
        if not n or args is None:
            return "", {}, 'could not read that tool call. Use: ACTION: {"tool": "grep", "args": {"pattern": "..."}}'
        return n.group(1).strip().lower(), args, ""
    m = _ACTION.search(reply)
    raw = reply[m.end():] if m else None
    if raw is None:
        s = reply.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[\w-]*\s*|\s*```$", "", s).strip()
        if s.startswith("{") and '"tool"' in s:
            raw = s
    if raw is None:
        return None
    d = parse_json_loose(raw)
    if not d or not isinstance(d.get("tool"), str):
        return "", {}, 'could not read that ACTION line. Use: ACTION: {"tool": "grep", "args": {"pattern": "..."}}'
    args = d.get("args") if isinstance(d.get("args"), dict) else {k: v for k, v in d.items() if k != "tool"}
    return d["tool"].strip().lower(), args, ""


def leads_from_context(ctx: str, cap: int) -> str:
    """The semantic-search context reduced to leads: named files become 'read me' lines, chunks are cut short."""
    if not ctx or not ctx.strip():
        return "(no semantic matches: search with the tools)"
    out, used = [], 0
    for block in re.split(r"(?m)^(?=--- \[)", ctx):
        if not block.strip():
            continue
        head, _, body = block.partition("\n")
        if "FULL_FILE" in head:
            row = f"{head.strip('- ').strip()}  <- named in the question; read it with the read tool"
        else:
            row = head.strip() + "\n" + body[:700].rstrip()
        if used + len(row) > cap:
            break
        out.append(row)
        used += len(row) + 2
    return "\n\n".join(out) or "(no semantic matches: search with the tools)"


# --------------------------------------------------------------------------- empty / thinking replies
class EmptyModelReply(RuntimeError):
    """The model sent nothing usable, twice. Callers fall back to the single-shot answer instead of showing a blank one."""


class ModelText(str):
    """A model reply (plain text) that also carries what the server said about it: done_reason, token counts, thinking size."""
    meta: dict

    def __new__(cls, text: str = "", meta: Optional[dict] = None):
        obj = super().__new__(cls, text)
        obj.meta = dict(meta or {})
        return obj


_THINK_SPANS = (        # (open, close): reasoning a model left inside its answer text
    (re.compile(r"<(?:think|thinking|reasoning)>", re.I), re.compile(r"</(?:think|thinking|reasoning)>", re.I)),
    (re.compile(r"<\|channel>thought"), re.compile(r"<channel\|>")),                 # gemma-4 style markers
)


def clean_reply(reply) -> str:
    """The answer part of a model reply: LEADING think blocks removed, whitespace trimmed.

    A block that was opened and never closed means the output budget ended inside the thinking: nothing was answered -> ''.
    Only a block at the very start is touched, so an answer that merely talks about a <think> tag stays as it is."""
    s = str(reply or "").strip()
    while s:
        for opn, cls in _THINK_SPANS:
            m = opn.match(s)
            if m:
                c = cls.search(s, m.end())
                if not c:
                    return ""
                s = s[c.end():].lstrip()
                break
        else:
            break
    return s.strip()


def reply_meta(resp) -> dict:
    """done_reason / token counts / thinking size of a LangChain AIMessage (whatever the server reported)."""
    rm = getattr(resp, "response_metadata", None) or {}
    meta = {k: rm[k] for k in ("model", "done_reason", "eval_count", "prompt_eval_count") if rm.get(k) is not None}
    # OpenAI-shaped servers (Cloudflare Workers AI, ...): finish_reason / token_usage / usage_metadata, same meaning
    if "done_reason" not in meta and rm.get("finish_reason") is not None:
        meta["done_reason"] = rm["finish_reason"]               # "length" = ran out of output budget, like Ollama's done_reason
    usage = rm.get("token_usage") if isinstance(rm.get("token_usage"), dict) else {}
    um = getattr(resp, "usage_metadata", None) if isinstance(getattr(resp, "usage_metadata", None), dict) else {}
    if "eval_count" not in meta:
        n = usage.get("completion_tokens", um.get("output_tokens"))
        if n is not None:
            meta["eval_count"] = n
    if "prompt_eval_count" not in meta:
        n = usage.get("prompt_tokens", um.get("input_tokens"))
        if n is not None:
            meta["prompt_eval_count"] = n
    thinking = (getattr(resp, "additional_kwargs", None) or {}).get("reasoning_content")
    if thinking:
        meta["thinking_chars"] = len(str(thinking))
    return meta


def describe_meta(meta: Optional[dict]) -> str:
    meta = meta or {}
    bits = []
    if meta.get("model"):
        bits.append(f"model={meta['model']}")
    if meta.get("done_reason"):
        bits.append(f"done_reason={meta['done_reason']}")
    if meta.get("eval_count") is not None:
        bits.append(f"{meta['eval_count']} tokens generated")
    if meta.get("prompt_eval_count") is not None:
        bits.append(f"{meta['prompt_eval_count']} prompt tokens")
    if meta.get("thinking_chars"):
        bits.append(f"{meta['thinking_chars']} chars of thinking")
    return ", ".join(bits) or "no details from the server"


_NUDGE = ("Your last reply was empty. Reply now with exactly ONE ACTION line or, if you already have enough evidence, "
          "with the final answer in markdown. Do not spend the reply on deliberation.")


def _strip_line_numbers(e: Edit) -> Edit:
    """Models sometimes copy the `12| ` prefix of read output into SEARCH/REPLACE: remove it when EVERY non-empty line has it."""
    def clean(text: str) -> str:
        rows = text.split("\n")
        body = [r for r in rows if r.strip()]
        if body and all(_NUM_PREFIX.match(r) for r in body):
            return "\n".join(_NUM_PREFIX.sub("", r) for r in rows)
        return text
    return Edit(e.path, clean(e.search), clean(e.replace), e.index)


# --------------------------------------------------------------------------- lean mode: old tool output -> one line
_HISTORY_IDS: ContextVar[Optional[Set[int]]] = ContextVar("ask_history_ids", default=None)    # id()s of earlier chat messages (droppable)
_OBS_HEAD = re.compile(r"OBSERVATION \(([^)]*)\):\n")
_OBS_TAIL = re.compile(r"\n\n\[\d+ tool call\(s\) left\]\s*$")
_READ_HEAD = re.compile(r"FILE (\S+) \(\d+ lines\), showing (\d+)-(\d+)")


def _is_observation(m) -> bool:
    return isinstance(m, HumanMessage) and isinstance(m.content, str) and m.content.startswith("OBSERVATION (")


def observation_stub(name: str, args: dict, obs: str) -> str:
    """`OBSERVATION (read) [compacted: services/x.py lines 1-185, 6,420 chars already seen; read again if needed]`"""
    args = args if isinstance(args, dict) else {}
    again = "read again if needed" if name == "read" else "run it again if needed"
    r = _READ_HEAD.match(obs) if name == "read" else None
    if r:
        target = f"{r.group(1)} lines {r.group(2)}-{r.group(3)}"
    elif name == "grep":
        target = f"/{args.get('pattern') or args.get('regex') or args.get('query') or ''}/" + (f" in {args.get('path') or args.get('glob')}" if args.get("path") or args.get("glob") else "")
    elif name == "find":
        target = str(args.get("pattern") or args.get("name") or args.get("path") or "")
    else:
        target = str(args.get("path") or args.get("file") or "")
    target = target.strip()[:120] or "(no target)"
    return f"OBSERVATION ({name}) [compacted: {target}, {len(obs):,} chars already seen; {again}]"


def compact_observations(msgs: list, keep: int, tools: Optional["RepoTools"] = None) -> Tuple[int, int]:
    """Replace every OBSERVATION message except the newest `keep` by a one-line stub (in place). The system message, the user's
    question, the model's own replies and the newest observations stay. -> (messages compacted, chars saved).
    A compacted call is forgotten by `tools`, so asking for it again really runs it (instead of 'you already ran this')."""
    idx = [i for i, m in enumerate(msgs) if _is_observation(m)]
    old = idx[: len(idx) - keep] if keep > 0 else idx
    n = saved = 0
    for i in old:
        content = msgs[i].content
        head = _OBS_HEAD.match(content)
        if not head or "[compacted:" in content[:head.end() + 40]:
            continue
        obs = _OBS_TAIL.sub("", content[head.end():])
        name, args = head.group(1).strip(), {}
        prev = msgs[i - 1] if i > 0 else None
        act = parse_action(str(prev.content)) if isinstance(prev, AIMessage) else None
        if act is not None and not act[2]:
            name, args = act[0] or name, act[1]
        stub = observation_stub(name, args, obs)
        if len(stub) >= len(content):
            continue
        msgs[i] = HumanMessage(content=stub)
        n += 1
        saved += len(content) - len(stub)
        if tools is not None and act is not None and not act[2]:
            tools._memo.pop(act[0] + json.dumps(args if isinstance(args, dict) else {}, sort_keys=True, default=str), None)
    return n, saved


# --------------------------------------------------------------------------- agent
@dataclass
class Check:
    problems: str = ""      # what the model must fix (empty = nothing)
    report: str = ""        # markdown appended to the answer


class CodeAgent:
    def __init__(self, store, sandbox, llm: Optional[Callable] = None, emit: Optional[Callable] = None):
        self.store, self.sandbox = store, sandbox
        self.llm = llm or self._ollama           # async (messages, repo_id=...) -> str ; injectable for tests
        self._custom_llm = llm is not None       # an injected model does not take the extra `reasoning` argument of _ollama
        self._emit_fn = emit

    def _emit(self, msg: str, level: str = "INFO", title: str = "Ask agent") -> None:
        try:
            if self._emit_fn:
                self._emit_fn(title, msg, level)
            else:
                from services.notifier import notifier
                notifier.notify(title, msg, level=level)
        except Exception:  # noqa: BLE001
            pass

    async def _ollama(self, messages, repo_id: str = "", reasoning=None) -> str:
        """One model turn. `reasoning` None = settings.ask_reasoning; False on the retry after a reply that ran out of budget while thinking."""
        from services.model_calls import _build, reasoning_arg, reasoning_model
        from services.llm_errors import with_model_fallback
        think = reasoning if reasoning is not None else reasoning_arg(settings.ask_reasoning)

        async def _call(m: str):
            llm = _build(m, json_mode=False, num_predict=settings.change_max_output_tokens,
                         num_ctx=settings.change_num_ctx, repo_id=repo_id, phase="ASK_AGENT", reasoning=think)
            return await llm.ainvoke(messages)
        resp = await with_model_fallback(_call, reasoning_model(), phase="ASK_AGENT")    # 402 -> local fallback model, or ModelQuotaError
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return ModelText(text, reply_meta(resp))

    async def _timed_llm(self, msgs: list, **kw):
        """self.llm(...) wrapped in an LLM action event (soft: logging never changes the call).
        The default model (_ollama -> model_calls._build) already logs one LLM event per call through the telemetry
        callback, so only an injected model gets the agent-level event (no duplicates)."""
        if not self._custom_llm:
            return await self.llm(msgs, **kw)
        try:
            from services.model_calls import reasoning_model
            model = reasoning_model()
        except Exception:  # noqa: BLE001
            model = None
        with _action("LLM", "AskAgent", "llm_turn", model=model, data={"messages": len(msgs)}) as h:
            resp = await self.llm(msgs, **kw)
            try:
                h.note(chars=len(str(resp)))
            except Exception:  # noqa: BLE001
                pass
            led = current_ledger()
            if led is not None:     # an injected model reports no usage and has no telemetry callback: record an estimate
                from services.runners import current_runner_name
                led.record(runner=current_runner_name(), model=str(model or ""), phase="ASK_AGENT", token_mode=effective_mode(),
                           input_tokens=estimate_tokens(msgs), output_tokens=len(str(resp)) // 4)
            return resp

    async def _reply(self, repo_id: str, msgs: list) -> str:
        """One model turn that is never empty. An empty reply (a thinking model that used its whole output budget thinking, or an
        empty completion) is asked for once more, with thinking off when running out of budget was the cause. Empty twice -> EmptyModelReply."""
        first = await self._timed_llm(msgs, repo_id=repo_id)
        text = clean_reply(first)
        if text:
            return text
        meta1 = getattr(first, "meta", None) or {}
        no_think = meta1.get("done_reason") == "length" and not self._custom_llm
        self._emit(f"The model sent an EMPTY reply ({describe_meta(meta1)}); asking once more" + (" with thinking off" if no_think else ""), "WARNING")
        retry = list(msgs) + [AIMessage(content="(empty reply)"), HumanMessage(content=_NUDGE)]
        second = await self._timed_llm(retry, repo_id=repo_id, **({"reasoning": False} if no_think else {}))
        text = clean_reply(second)
        if text:
            return text
        meta2 = getattr(second, "meta", None) or {}
        raise EmptyModelReply(f"empty reply twice. 1st: {describe_meta(meta1)}. 2nd: {describe_meta(meta2)}")

    # ------------------------------------------------------------------ public
    async def answer(self, repo_id: str, system: str, history: list) -> str:
        """`system` = the normal ask prompt (with leads), `history` = the chat messages. Returns the markdown answer."""
        snapshot = await self.store.snapshot_bytes(repo_id, 0)      # rev 0 = your disk as the engine last saw it (what Q&A reads too)
        if not snapshot:
            raise ValueError("the raw store has no files for this repo (re-ingest it)")
        tools = RepoTools(snapshot)
        with ledger_scope() as led:
            history = list(history)
            if is_lean():       # lean: the last lean_history_turns earlier messages + the current question
                history, dropped = trim_history(history, settings.lean_history_turns)
                led.add_saved(dropped)
            msgs: list = [SystemMessage(content=system + guidance_block() + PROTOCOL)] + history   # guidance: core skill text for an edit-like ask, "" otherwise
            last_q = max((i for i, m in enumerate(history) if isinstance(m, HumanMessage)), default=len(history) - 1)
            hist_token = _HISTORY_IDS.set({id(m) for m in history[:last_q]})      # what the input cap may drop (oldest first)
            try:
                final = await self._loop(repo_id, msgs, tools, budget("ask_max_steps"))
                report = ""
                if settings.ask_validate:
                    final, report = await self._validate(repo_id, msgs, tools, snapshot, final)
            finally:
                _HISTORY_IDS.reset(hist_token)
            self._emit_usage(led)
            return final.rstrip() + report + self._footer(tools, led)

    def _emit_usage(self, led) -> None:
        """One telemetry event per question: the sum of its LLM calls (the Telemetry tab shows it as a summary line)."""
        if led.calls:
            self._emit(json.dumps(summary_event(led), default=str), "INFO", title="Token usage")

    # ------------------------------------------------------------------ loop
    async def _loop(self, repo_id: str, msgs: list, tools: RepoTools, steps: int) -> str:
        """Tool calls until the model answers. `msgs` grows in place."""
        for left in range(steps, 0, -1):
            if tools.chars > budget("ask_obs_chars"):
                break
            self._prepare(msgs, tools)
            reply = await self._reply(repo_id, msgs)
            act = parse_action(reply)
            if act is None:
                return reply.strip()
            name, args, err = act
            obs = err or tools.run(name, args)
            self._emit(f"{name or 'bad action'} {json.dumps(args, default=str)[:120]} -> {len(obs)} chars")
            try:
                target = args.get("pattern") or args.get("path") or args.get("name") or "" if isinstance(args, dict) else ""
                _log_action("RETRIEVE", "AskAgent", f"{name or 'bad action'} {str(target)[:80]}".strip(),
                            "fail" if err else "ok", f"{len(obs)} chars, {obs.count(chr(10)) + 1 if obs else 0} line(s)",
                            {"tool": name, "args": args, "chars": len(obs), "lines": obs.count("\n") + 1 if obs else 0,
                             "tool_calls_left": left - 1})
            except Exception:  # noqa: BLE001
                pass
            msgs += [AIMessage(content=reply), HumanMessage(content=f"OBSERVATION ({name or 'error'}):\n{obs}\n\n[{left - 1} tool call(s) left]")]
        msgs.append(HumanMessage(content="The tool budget is used up. Answer now from what you have read. State what you could NOT verify. "
                                         "Do not write an ACTION line."))
        self._prepare(msgs, tools)
        reply = await self._reply(repo_id, msgs)
        if parse_action(reply) is not None:
            reply = strip_tool_calls(reply) or ("I ran out of tool calls before I could answer. Ask again with a narrower question "
                                                        "or name the file.")
        return reply.strip()

    # ------------------------------------------------------------------ token savings (lean mode / input cap)
    def _prepare(self, msgs: list, tools: RepoTools) -> None:
        """Before each model call. Lean: old tool output becomes a one-line stub. Then the input cap (if any) trims until the estimate fits.
        Normal mode without MAX_INPUT_TOKENS touches nothing, so the prompt is exactly what it always was."""
        try:
            if is_lean():
                n, saved = compact_observations(msgs, keep_full_obs(), tools)
                if n:
                    note_saved(saved)
                    self._emit(f"lean: {n} older tool output(s) compacted (~{saved // 4} tokens not resent)")
            cap = max_input_tokens()
            if cap > 0:
                self._enforce_cap(msgs, tools, cap)
        except Exception as e:  # noqa: BLE001 - saving tokens must never break the answer
            self._emit(f"token trimming skipped: {type(e).__name__}: {e}", "WARNING")

    def _enforce_cap(self, msgs: list, tools: RepoTools, cap: int) -> None:
        """Input cap (estimated tokens). Order: oldest earlier chat messages first, then more tool output (all but the newest).
        The system prompt and the latest user message are never dropped."""
        est = estimate_tokens(msgs)
        if est <= cap:
            return
        hist = _HISTORY_IDS.get() or set()
        dropped_msgs = removed = 0
        while est > cap:
            idx = next((i for i, m in enumerate(msgs) if i > 0 and id(m) in hist), None)
            if idx is None:
                break
            removed += len(text_of(msgs[idx].content))
            hist.discard(id(msgs[idx]))
            del msgs[idx]
            dropped_msgs += 1
            est = estimate_tokens(msgs)
        while len(msgs) > 1 and id(msgs[1]) in hist and isinstance(msgs[1], AIMessage):     # never start the chat with an assistant reply
            removed += len(text_of(msgs[1].content))
            hist.discard(id(msgs[1]))
            del msgs[1]
            dropped_msgs += 1
        est = estimate_tokens(msgs)
        compacted = 0
        if est > cap:
            n, saved = compact_observations(msgs, 1, tools)
            compacted, removed = n, removed + saved
            est = estimate_tokens(msgs)
        note_saved(removed)
        self._emit(f"input cap {cap} tokens: dropped {dropped_msgs} earlier message(s), compacted {compacted} tool output(s); "
                   f"now ~{est} tokens" + ("" if est <= cap else " (still over: system prompt, question and newest tool output are kept)"),
                   "INFO" if est <= cap else "WARNING")

    # ------------------------------------------------------------------ validation
    async def _validate(self, repo_id: str, msgs: list, tools: RepoTools, snapshot: Dict[str, bytes], final: str) -> Tuple[str, str]:
        rounds = max(0, settings.ask_validate_rounds)
        report = ""
        for rnd in range(rounds + 1):
            with _action("SANDBOX", "AskAgent", f"validate edits (round {rnd + 1})") as _h:
                chk = await self._check(repo_id, final, tools, snapshot)
                if chk is None:
                    _h.note("no edit blocks: nothing to validate", status="skip")
                else:
                    _h.note("problems found" if chk.problems else "edits check out", ok=not chk.problems)
            if chk is None:                                  # no edit blocks: nothing to validate
                return final, ""
            report = chk.report
            if not chk.problems or rnd == rounds:
                return final, report
            self._emit(f"Proposed edits failed the check (round {rnd + 1}/{rounds}): {chk.problems[:240]}", "WARNING")
            msgs += [AIMessage(content=final),
                     HumanMessage(content="Your edit blocks were applied to a copy of the repository and checked. Problems:\n"
                                          f"{chk.problems}\n\nFix them (use the tools if you need to read more) and reply with the complete, "
                                          "corrected answer including every edit block.")]
            try:
                fixed = await self._loop(repo_id, msgs, tools, max(2, budget("ask_max_steps") // 2))
            except EmptyModelReply as e:
                self._emit(f"No reply while correcting the failed check: {e}", "WARNING")
                return final, report + "\n\n_The model stopped replying while correcting the code above, so it stays unverified._"
            final = fixed
        return final, report

    @staticmethod
    def _failed(problems: List[str]) -> Check:
        body = "\n".join(f"- {p}" for p in problems)[:4500]
        return Check(problems=body, report="\n\n---\n**The proposed edit(s) did NOT pass the check. Treat the code above as unverified:**\n" + body)

    async def _check(self, repo_id: str, answer: str, tools: RepoTools, snapshot: Dict[str, bytes]) -> Optional[Check]:
        if "<<<<<<< SEARCH" in answer and "```" not in answer:
            return self._failed(
                ["the edit blocks are not inside a markdown code fence. Wrap each FILE line's block in ```python ... ``` as described in rule 6."] 
            )
        edits, perr = parse_edits(answer)
        if not edits and not (perr and "<<<<<<< SEARCH" in answer):
            return None
        if not edits:
            return self._failed(perr)
        known = sorted(snapshot)
        edits, _notes, rerr = resolve_edit_paths([_strip_line_numbers(e) for e in edits], known)
        # facts for the delivery notes (this check can run again after a repair: each run starts from "nothing verified")
        note_facts(**{**facts_from_sandbox(None), "edits_proposed": True, "static_checked": False})
        if perr or rerr:
            return self._failed(list(perr) + list(rerr))
        paths = sorted({e.path for e in edits})
        res = apply_edits(edits, {p: snapshot.get(p) for p in paths})
        if not res.ok:
            return self._failed(res.errors)
        unread: List[str] = []
        for o in res.outcomes:
            msg = tools.unread(o.path, o.line) if (o.ok and o.mode != "new-file") else None
            if msg and msg not in unread:
                unread.append(msg)
        if unread:
            return self._failed(unread)
        working = dict(res.new_files)

        static = static_check.new_failures(static_check.check_files({p: snapshot[p] for p in working if p in snapshot}, known),
                                           static_check.check_files(working, known + list(working)))
        if static:
            return self._failed(["static check: " + r.label() for r in static[:5]])
        try:        # fail-open, like /change: a checker bug must never block an answer
            py = {p: b for p, b in snapshot.items() if p.endswith(".py")}
            calls, _owners = call_check.new_problems(py, {**py, **{p: b for p, b in working.items() if p.endswith(".py")}})
        except Exception as e:  # noqa: BLE001
            self._emit(f"call check skipped: {type(e).__name__}", "WARNING")
            calls = []
        if calls:
            return self._failed(["call check: " + c for c in calls[:5]])
        note_facts(static_checked=True)

        files = ", ".join(f"`{p}`" for p in sorted(working))
        head = f"\n\n---\n**Sandbox check of the proposed edit(s)** ({files}; applied to a copy, nothing was written to your repo)\n"
        applied = "- The edits apply exactly to your current files; syntax, template and call checks pass.\n"
        gen = [p for p in working if p not in snapshot and is_test_file(p)]
        try:
            v = await asyncio.wait_for(self.sandbox.verify(repo_id, 0, working, new_tests=gen), settings.ask_validate_timeout_s)
        except asyncio.TimeoutError:
            return Check(report=head + applied + f"- **Not run in the sandbox:** it did not finish within {settings.ask_validate_timeout_s}s.\n")
        except Exception as e:  # noqa: BLE001
            return Check(report=head + applied + f"- **Not run in the sandbox:** {type(e).__name__}: {e}\n")
        t = v.to_dict()
        note_facts(**facts_from_sandbox(t))
        if v.ran and not v.ok:
            from services.repo_change import RepoChangeService
            tops = {p.split("/")[0].removesuffix(".py") for p in list(snapshot) + list(working)}
            real = [c for c in v.new_failures if not RepoChangeService._is_environment(c, tops)]
            if real:
                return self._failed([f"test {c.id} fails: {c.message}\n" + "\n".join("    " + l for l in c.detail.splitlines()[-14:])
                                     for c in real[:3]])
            t.update(ok=False, inconclusive=True, note="The failures come from the sandbox environment (missing package or network), not from the edit.")
            note_facts(**facts_from_sandbox(t))
        from services.repo_change import ChangeResult
        line = ChangeResult(ok=True, repo_id=repo_id, request="", tests=t)._tests_line()
        tail = "- To get this as a saved, downloadable revision, send the request again with `/change`.\n"
        return Check(report=head + applied + f"- {line}\n" + tail)

    # ------------------------------------------------------------------ footer
    @staticmethod
    def _footer(tools: RepoTools, ledger=None) -> str:
        """Which tools / files the answer rests on, plus (ledger given, settings.show_token_footer) the token line."""
        if not tools.calls:
            base = "\n\n_No search tools were used: this answer comes from the semantic-search leads only._"
        else:
            used = ", ".join(f"{n} x{c}" for n, c in sorted(tools.calls.items()))
            read = tools.files_read()
            shown = ", ".join(read[:6]) + (f" (+{len(read) - 6} more)" if len(read) > 6 else "")
            base = f"\n\n_Looked at with: {used}" + (f". Files read: {shown}_" if read else "_")
        line = footer_text(ledger.totals()) if (ledger is not None and settings.show_token_footer) else ""
        return base + (f"\n\n{line}" if line else "")
