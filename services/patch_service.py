"""SEARCH/REPLACE edit blocks: parse, resolve paths, apply deterministically, diff.

The model never rewrites whole files. It emits small blocks:

    FILE: services/foo.py
    <<<<<<< SEARCH
    exact existing lines
    =======
    replacement lines
    >>>>>>> REPLACE

Apply rules (why a bad block fails loudly instead of corrupting a file):
  * the SEARCH text must match whole lines and match EXACTLY ONE place;
  * matching is tried strictest-first: exact -> trailing-space-insensitive ->
    indentation-insensitive (replacement is re-indented to fit);
  * a file with any failed block is not applied at all (no half-edited files), unless apply_edits(partial=True): then
    the blocks that DID match are kept and the failed ones are reported by index (ApplyResult.partial), so a retry only
    has to re-send what failed instead of redoing (or silently losing) the whole file;
  * original line endings (CRLF/LF) are preserved.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from services.repo_store import is_secret_path, safe_rel_path

_SEARCH = re.compile(r"^<{5,9} SEARCH\s*$")
_DIVIDE = re.compile(r"^={5,9}\s*$")
_REPLACE = re.compile(r"^>{5,9} REPLACE\s*$")
_FILE = re.compile(r"^\s*[*#`\s]*(?:FILE|PATH)\s*:\s*(.+?)[*`\s]*$", re.IGNORECASE)
_BARE_PATH = re.compile(r"^[`*\s]*([\w./\\-]+\.\w+)[`*\s]*$")
_FENCE = re.compile(r"^\s*```")


@dataclass
class Edit:
    path: str
    search: str
    replace: str
    index: int


@dataclass
class EditOutcome:
    path: str
    index: int
    ok: bool
    mode: str = ""          # exact | rstrip | indent | new-file
    line: int = 0           # 1-based line where it applied
    error: str = ""


@dataclass
class ApplyResult:
    new_files: Dict[str, bytes] = field(default_factory=dict)
    outcomes: List[EditOutcome] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    partial: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)     # path -> {"applied": [idx], "failed": [idx]}

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- parse
def parse_edits(text: str) -> Tuple[List[Edit], List[str]]:
    lines = text.replace("\r\n", "\n").split("\n")
    edits: List[Edit] = []
    errors: List[str] = []
    state, path, candidate = "out", None, None
    s_buf: List[str] = []
    r_buf: List[str] = []
    for ln in lines:
        if state == "out":
            if _SEARCH.match(ln):
                if not path and candidate:
                    path = candidate
                if not path:
                    errors.append("SEARCH block with no 'FILE: <path>' line before it")
                state, s_buf, r_buf = "search", [], []
                continue
            m = _FILE.match(ln)
            if m:
                path, candidate = m.group(1).strip().strip("`'\""), None
            elif ln.strip() and not _FENCE.match(ln):
                b = _BARE_PATH.match(ln)
                candidate = b.group(1) if b else None
                if candidate:
                    path = None  # a bare path line right before a marker names the file
        elif state == "search":
            if _DIVIDE.match(ln):
                state = "replace"
            else:
                s_buf.append(ln)
        else:
            if _REPLACE.match(ln):
                # symmetric blank padding is noise, not content
                while s_buf and r_buf and not s_buf[-1].strip() and not r_buf[-1].strip():
                    s_buf.pop(); r_buf.pop()
                while s_buf and r_buf and not s_buf[0].strip() and not r_buf[0].strip():
                    s_buf.pop(0); r_buf.pop(0)
                if path:
                    edits.append(Edit(path, "\n".join(s_buf), "\n".join(r_buf), len(edits) + 1))
                state, candidate = "out", None
            else:
                r_buf.append(ln)
    if state != "out":
        errors.append("unterminated edit block (missing '>>>>>>> REPLACE')")
    return edits, errors


# --------------------------------------------------------------------------- paths
def resolve_edit_paths(edits: List[Edit], known_paths: Sequence[str]) -> Tuple[List[Edit], List[str], List[str]]:
    """Map model-written paths onto real repo paths. Returns (edits, notes, errors)."""
    known = list(known_paths)
    known_set = set(known)
    lower = {}
    for k in known:
        lower.setdefault(k.lower(), []).append(k)
    by_base: Dict[str, List[str]] = {}
    for k in known:
        by_base.setdefault(k.rsplit("/", 1)[-1].lower(), []).append(k)

    out, notes, errors = [], [], []
    seen_notes = set()
    for e in edits:
        try:
            p = safe_rel_path(e.path)
        except ValueError:
            errors.append(f"edit {e.index}: unsafe or invalid path '{e.path}'")
            continue
        if p not in known_set:
            ci = lower.get(p.lower(), [])
            bn = by_base.get(p.rsplit("/", 1)[-1].lower(), [])
            if len(ci) == 1:
                fixed = ci[0]
            elif not e.search.strip():
                fixed = p                              # new file at the stated path
            elif len(bn) == 1:
                fixed = bn[0]
            else:
                close = difflib.get_close_matches(p, known, n=3, cutoff=0.5)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                errors.append(f"edit {e.index}: file '{e.path}' does not exist in the repo.{hint}")
                continue
            if fixed != p and (p, fixed) not in seen_notes:
                notes.append(f"path '{e.path}' resolved to '{fixed}'")
                seen_notes.add((p, fixed))
            p = fixed
        if not e.search.strip() and is_secret_path(p):
            errors.append(f"edit {e.index}: refusing to create secret-like file '{p}'")
            continue
        out.append(Edit(p, e.search, e.replace, e.index))
    return out, notes, errors


# --------------------------------------------------------------------------- apply
def _norm_exact(s: str) -> str:
    return s


def _norm_rstrip(s: str) -> str:
    return s.rstrip()


def _norm_strip(s: str) -> str:
    return s.strip()


_MODES: List[Tuple[str, Callable[[str], str]]] = [
    ("exact", _norm_exact), ("rstrip", _norm_rstrip), ("indent", _norm_strip)]


def _find_all(file_norm: List[str], needle_norm: List[str]) -> List[int]:
    n = len(needle_norm)
    if n == 0 or n > len(file_norm):
        return []
    first = needle_norm[0]
    return [i for i in range(len(file_norm) - n + 1)
            if file_norm[i] == first and file_norm[i:i + n] == needle_norm]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(replace_lines: List[str], file_line: str, search_line: str) -> Optional[List[str]]:
    fi, si = _indent_of(file_line), _indent_of(search_line)
    if fi == si:
        return replace_lines
    if fi.endswith(si):                       # model dedented the block: add the prefix back
        prefix = fi[: len(fi) - len(si)]
        return [(prefix + l) if l.strip() else l for l in replace_lines]
    if si.endswith(fi):                       # model over-indented: remove the excess
        excess = si[: len(si) - len(fi)]
        return [l[len(excess):] if l.startswith(excess) else l for l in replace_lines]
    return None


def _closest_hint(lines: List[str], needle: List[str]) -> str:
    """Point at the most similar region so a retry can fix its SEARCH text."""
    if not needle or len(lines) > 8000:
        return ""
    probe = next((l.strip() for l in needle if l.strip()), "")
    if not probe:
        return ""
    cands = [i for i, l in enumerate(lines)
             if difflib.SequenceMatcher(None, probe, l.strip()).quick_ratio() > 0.7][:25]
    best, best_r = None, 0.0
    n = len(needle)
    target = "\n".join(needle)
    for i in cands:
        r = difflib.SequenceMatcher(None, target, "\n".join(lines[i:i + n])).ratio()
        if r > best_r:
            best, best_r = i, r
    if best is None or best_r < 0.4:
        return ""
    snippet = "\n".join(lines[best:best + min(n, 8)])
    return f" Closest text is at line {best + 1}:\n{snippet}"


def _detect_newline(text: str) -> Tuple[str, bool]:
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    return ("\r\n" if crlf > lf_only else "\n"), (crlf > 0 and lf_only > 0)


def apply_edits(edits: List[Edit], contents: Dict[str, Optional[bytes]], partial: bool = False) -> ApplyResult:
    """Apply resolved edits. `contents[path]` is the current bytes, or None for a new file.
    partial=False (default): a file with any failed block is left untouched. partial=True: the blocks that matched are
    applied, the failed ones stay in `errors` and are listed per file in `partial` (a new file is never partial)."""
    res = ApplyResult()
    order: List[str] = []
    grouped: Dict[str, List[Edit]] = {}
    for e in edits:
        if e.path not in grouped:
            order.append(e.path)
        grouped.setdefault(e.path, []).append(e)

    for path in order:
        original = contents.get(path)
        failed = False
        if original is None:
            text, nl = "", "\n"
        else:
            try:
                text = original.decode("utf-8")
            except UnicodeDecodeError:
                res.errors.append(f"{path}: not UTF-8 text, cannot edit")
                continue
            nl, mixed = _detect_newline(text)
            if mixed:
                res.warnings.append(f"{path}: mixed line endings were normalised to {'CRLF' if nl == chr(13)+chr(10) else 'LF'}")
            text = text.replace("\r\n", "\n")
        lines = text.split("\n")
        creating = original is None
        file_outcomes: List[EditOutcome] = []

        for e in grouped[path]:
            s_lines = e.search.replace("\r\n", "\n").split("\n") if e.search.strip() else []
            r_lines = e.replace.replace("\r\n", "\n").split("\n") if e.replace != "" else []

            if not s_lines:                                   # new-file edit
                if creating and not lines == [""]:
                    msg = f"{path} edit {e.index}: second empty-SEARCH block for a new file"
                elif not creating:
                    msg = (f"{path} edit {e.index}: SEARCH is empty but the file already exists. "
                           f"Use a SEARCH block that quotes the lines to change.")
                else:
                    lines = r_lines + [""] if r_lines else [""]
                    file_outcomes.append(EditOutcome(path, e.index, True, "new-file", 1))
                    continue
                file_outcomes.append(EditOutcome(path, e.index, False, error=msg))
                res.errors.append(msg)
                failed = True
                continue

            if creating:
                msg = f"{path} edit {e.index}: file does not exist; to create it leave SEARCH empty"
                file_outcomes.append(EditOutcome(path, e.index, False, error=msg))
                res.errors.append(msg)
                failed = True
                continue

            hit = None
            for mode, norm in _MODES:
                f_norm = [norm(l) for l in lines]
                n_norm = [norm(l) for l in s_lines]
                found = _find_all(f_norm, n_norm)
                if len(found) > 1:
                    msg = (f"{path} edit {e.index}: SEARCH matches {len(found)} places "
                           f"(lines {', '.join(str(i + 1) for i in found[:5])}). "
                           f"Include more surrounding lines so it is unique.")
                    file_outcomes.append(EditOutcome(path, e.index, False, mode, error=msg))
                    res.errors.append(msg)
                    failed = True
                    hit = False
                    break
                if len(found) == 1:
                    hit = (found[0], mode)
                    break
            if hit is False:
                continue
            if hit is None:
                msg = (f"{path} edit {e.index}: SEARCH text not found. "
                       f"It must be copied verbatim from the file.{_closest_hint(lines, s_lines)}")
                file_outcomes.append(EditOutcome(path, e.index, False, error=msg))
                res.errors.append(msg)
                failed = True
                continue

            start, mode = hit
            new_r = r_lines
            if mode == "indent":
                ref = next(i for i, l in enumerate(s_lines) if l.strip())
                adj = _reindent(r_lines, lines[start + ref], s_lines[ref])
                if adj is None:
                    msg = f"{path} edit {e.index}: SEARCH only matched ignoring indentation and it could not be re-aligned safely"
                    file_outcomes.append(EditOutcome(path, e.index, False, mode, error=msg))
                    res.errors.append(msg)
                    failed = True
                    continue
                new_r = adj
            lines[start:start + len(s_lines)] = new_r
            file_outcomes.append(EditOutcome(path, e.index, True, mode, start + 1))

        res.outcomes.extend(file_outcomes)
        if failed:
            good = [o.index for o in file_outcomes if o.ok]
            if not (partial and good and not creating):
                continue                                      # never half-apply a file (default)
            res.partial[path] = {"applied": good, "failed": [o.index for o in file_outcomes if not o.ok]}
        new_text = "\n".join(lines)
        if nl == "\r\n":
            new_text = new_text.replace("\n", "\r\n")
        new_bytes = new_text.encode("utf-8")
        if original is not None and new_bytes == original:
            res.warnings.append(f"{path}: edits produced no change")
            continue
        res.new_files[path] = new_bytes
    return res


# --------------------------------------------------------------------------- diff
# A diff that `git apply` accepts must be byte-faithful:
#   * lines are split on "\n" ONLY (str.splitlines also splits on \x0b \x0c \x1c-\x1e \x85 \u2028 \u2029 and would
#     corrupt any file that contains one) and every line keeps its own ending, so CRLF files diff as CRLF;
#   * a last line without a newline is followed by "\\ No newline at end of file";
#   * the text ends with a newline, and chunks of several files are joined WITHOUT inserting blank lines;
#   * every file starts with `diff --git`, new/deleted files carry their mode line, odd names are C-quoted.
_NO_EOL = "\\ No newline at end of file\n"


def _split_keepends(data: bytes) -> Optional[List[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    out: List[str] = []
    start = 0
    while True:
        i = text.find("\n", start)
        if i < 0:
            if start < len(text):
                out.append(text[start:])
            return out
        out.append(text[start:i + 1])
        start = i + 1


def _git_name(prefix: str, path: str) -> str:
    """`a/<path>` the way git writes it: unquoted unless it holds quotes, backslashes, control or non-ASCII chars."""
    full = prefix + path
    if all(0x20 <= ord(c) < 0x7F and c not in '"\\' for c in full):
        return full
    esc = []
    for ch in full:
        o = ord(ch)
        if ch == '"':
            esc.append('\\"')
        elif ch == "\\":
            esc.append("\\\\")
        elif ch == "\t":
            esc.append("\\t")
        elif ch == "\n":
            esc.append("\\n")
        elif 0x20 <= o < 0x7F:
            esc.append(ch)
        else:
            esc.extend(f"\\{b:03o}" for b in ch.encode("utf-8"))
    return '"' + "".join(esc) + '"'


def make_diff(path: str, old: Optional[bytes], new: Optional[bytes]) -> str:
    """Unified diff of one file (old/new = bytes, None = absent) that `git apply` accepts. Always ends in a newline.
    Returns "" when nothing differs. Files that are not UTF-8 text cannot be expressed as a text hunk: a one-line
    comment names them (git ignores text between file diffs) instead of corrupting the patch."""
    if old == new:
        return ""
    a = [] if old is None else _split_keepends(old)
    b = [] if new is None else _split_keepends(new)
    if a is None or b is None:
        return f"# binary or non-UTF-8 file not included in this patch: {path}\n"

    na, nb = _git_name("a/", path), _git_name("b/", path)
    out: List[str] = [f"diff --git {na} {nb}\n"]
    if old is None:
        out.append("new file mode 100644\n")
    elif new is None:
        out.append("deleted file mode 100644\n")
    if not a and not b:                      # empty file created or deleted: git needs no hunks
        return "".join(out)

    tab = "\t" if " " in path else ""          # git marks names that hold a space with a trailing tab
    out.append(f"--- {'/dev/null' if old is None else na}{tab}\n")
    out.append(f"+++ {'/dev/null' if new is None else nb}{tab}\n")
    body = list(difflib.unified_diff(a, b, "", "", n=3, lineterm="\n"))[2:]       # drop difflib's own two headers
    for ln in body:
        out.append(ln)
        if ln[:1] in " +-" and not ln.endswith("\n"):       # last line of a file that has no trailing newline
            out.append("\n" + _NO_EOL)
    text = "".join(out)
    return text if text.endswith("\n") else text + "\n"


def join_diffs(chunks: Sequence[str]) -> str:
    """Concatenate per-file diffs into one patch (each chunk already ends in a newline; no separators added)."""
    return "".join(c if c.endswith("\n") else c + "\n" for c in chunks if c)


def diff_stats(diff: str) -> Tuple[int, int]:
    """(added, removed) counted inside hunks only, so a changed line that starts with '--' or '++' is not mistaken for a header."""
    added = removed = 0
    in_hunk = False
    for l in diff.split("\n"):
        if l.startswith("diff --git "):
            in_hunk = False
        elif l.startswith("@@"):
            in_hunk = True
        elif in_hunk:
            if l.startswith("+"):
                added += 1
            elif l.startswith("-"):
                removed += 1
    return added, removed
