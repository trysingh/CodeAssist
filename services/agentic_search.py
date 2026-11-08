"""Agentic (grep/glob/read/symbol) search over an in-memory copy of a repo. Stdlib only.

No embeddings: the corpus is a {path: text} mapping (built from the raw snapshot store or a directory), and
`AgenticSearch.retrieve` ranks files/blocks with path, symbol-definition and term-frequency signals, then reports a
confidence plus human-readable reasons so the caller can decide whether a (slow) vector index is worth building.
"""
from __future__ import annotations

import math
import os
import re
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # soft import: Agent A's action log may not exist (yet) or may be unimportable in minimal test envs
    from services.action_log import log_action as _log_action
except Exception:  # noqa: BLE001
    def _log_action(*_a, **_k):  # type: ignore[misc]
        return None

DEFAULT_EXCLUDES = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", ".idea", ".mypy_cache",
})


# --------------------------------------------------------------------------- data classes
@dataclass
class Hit:
    path: str
    line: int
    text: str
    score: float = 1.0


@dataclass
class Block:
    path: str
    start: int
    end: int
    text: str
    score: float
    why: str = ""


@dataclass
class SearchResult:
    blocks: List[Block] = field(default_factory=list)
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    files_considered: int = 0
    elapsed_ms: int = 0

    def to_context(self, max_chars: int) -> str:
        """'--- path:start-end ---\\n<text>' blocks, best first, cut to max_chars."""
        out: List[str] = []
        used = 0
        for b in self.blocks:
            piece = f"--- {b.path}:{b.start}-{b.end} ---\n{b.text}"
            if used + len(piece) > max_chars:
                room = max_chars - used
                if room > 80 and not out:
                    out.append(piece[:room])
                elif room > 200:
                    out.append(piece[:room])
                break
            out.append(piece)
            used += len(piece) + 2
        return "\n\n".join(out)


# --------------------------------------------------------------------------- corpus
class _Doc:
    __slots__ = ("text", "_lines", "_lower", "_nl")

    def __init__(self, text: str):
        self.text = text
        self._lines: Optional[List[str]] = None
        self._lower: Optional[str] = None
        self._nl: Optional[List[int]] = None

    @property
    def lines(self) -> List[str]:
        if self._lines is None:
            self._lines = [ln[:-1] if ln.endswith("\r") else ln for ln in self.text.split("\n")]
        return self._lines

    @property
    def lower(self) -> str:
        if self._lower is None:
            self._lower = self.text.lower()
        return self._lower

    @property
    def newlines(self) -> List[int]:
        """Offsets of '\\n' in the lowercase text (same newline positions as the original for line mapping)."""
        if self._nl is None:
            low = self.lower
            nl: List[int] = []
            i = low.find("\n")
            while i != -1:
                nl.append(i)
                i = low.find("\n", i + 1)
            self._nl = nl
        return self._nl

    def line_of(self, offset: int) -> int:
        return bisect_right(self.newlines, offset - 1) + 1 if offset > 0 else 1


def _norm_path(p: str) -> str:
    p = str(p).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


class RepoCorpus:
    """In-memory {path: text}. Binary (NUL-containing) files are skipped; text is decoded lazily-cheap (utf-8, replace)."""

    def __init__(self):
        self._docs: Dict[str, _Doc] = {}
        self._sorted: Optional[List[str]] = None

    @classmethod
    def from_mapping(cls, files: Mapping[str, Union[str, bytes]]) -> "RepoCorpus":
        c = cls()
        for path, content in files.items():
            if isinstance(content, (bytes, bytearray, memoryview)):
                raw = bytes(content)
                if b"\x00" in raw:
                    continue
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(content)
                if "\x00" in text:
                    continue
            c._docs[_norm_path(path)] = _Doc(text)
        return c

    @classmethod
    def from_dir(cls, root, include_globs: Optional[Sequence[str]] = None,
                 exclude_dirs: Iterable[str] = DEFAULT_EXCLUDES, max_file_bytes: int = 1_000_000) -> "RepoCorpus":
        root = str(root)
        excl = set(exclude_dirs)
        matcher = _compile_globs(include_globs) if include_globs else None
        files: Dict[str, bytes] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in excl)
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = _norm_path(os.path.relpath(full, root))
                if matcher and not matcher(rel):
                    continue
                try:
                    if os.path.getsize(full) > max_file_bytes:
                        continue
                    with open(full, "rb") as fh:
                        files[rel] = fh.read()
                except OSError:
                    continue
        return cls.from_mapping(files)

    def paths(self) -> List[str]:
        if self._sorted is None:
            self._sorted = sorted(self._docs)
        return list(self._sorted)

    def doc(self, path: str) -> Optional[_Doc]:
        return self._docs.get(_norm_path(path))

    def get(self, path: str) -> Optional[str]:
        d = self.doc(path)
        return d.text if d else None

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, path: str) -> bool:
        return _norm_path(path) in self._docs


# --------------------------------------------------------------------------- glob
def _expand_braces(pattern: str, limit: int = 256) -> List[str]:
    """'a/*.{py,js}' -> ['a/*.py', 'a/*.js'] (nested braces supported)."""
    start = pattern.find("{")
    if start == -1:
        return [pattern]
    depth, i, parts, last = 0, start, [], start + 1
    end = -1
    while i < len(pattern):
        ch = pattern[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parts.append(pattern[last:i])
                end = i
                break
        elif ch == "," and depth == 1:
            parts.append(pattern[last:i])
            last = i + 1
        i += 1
    if end == -1:           # unbalanced: treat literally
        return [pattern]
    pre, post = pattern[:start], pattern[end + 1:]
    out: List[str] = []
    for part in parts:
        for rest in _expand_braces(part + post, limit):
            out.append(pre + rest)
            if len(out) >= limit:
                return out
    return out


def _glob_to_regex(pattern: str) -> str:
    """Translate one brace-free glob to a regex body (forward-slash paths). ** crosses dirs, * and ? do not."""
    i, n, out = 0, len(pattern), []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**", i):
                j = i + 2
                while j < n and pattern[j] == "*":
                    j += 1
                if j < n and pattern[j] == "/":
                    out.append("(?:.*/)?")          # '**/' = zero or more directories
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 2)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _compile_globs(patterns: Union[str, Sequence[str]]) -> Callable[[str], bool]:
    pats = [patterns] if isinstance(patterns, str) else list(patterns)
    full: List[str] = []
    base: List[str] = []
    for p in pats:
        p = str(p).replace("\\", "/").strip()
        while p.startswith("./"):
            p = p[2:]
        p = p.lstrip("/")
        for e in _expand_braces(p):
            rx = _glob_to_regex(e)
            full.append(rx)
            if "/" not in e:
                base.append(rx)         # '*.py' also matches by basename at any depth
    full_re = re.compile(r"(?:%s)\Z" % "|".join(full)) if full else None
    base_re = re.compile(r"(?:%s)\Z" % "|".join(base)) if base else None

    def match(path: str) -> bool:
        if full_re and full_re.match(path):
            return True
        if base_re and base_re.match(path.rsplit("/", 1)[-1]):
            return True
        return False
    return match


# --------------------------------------------------------------------------- symbols
_W = r"[A-Za-z_$][\w$]*"
_MODS = r"(?:(?:public|private|protected|internal|static|final|abstract|override|virtual|open|sealed|async|suspend|readonly|partial|extern|synchronized|default|export)\s+)"

_PY = [re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+({_W})")]
_JS = [
    re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:function\*?|class|interface|enum|namespace)\s+({_W})"),
    re.compile(rf"^\s*(?:export\s+)?type\s+({_W})\s*(?:<[^=]*>)?\s*="),
    re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?(?:const|let|var)\s+({_W})\s*(?::[^=]+)?="),
    re.compile(rf"^\s*(?:(?:public|private|protected|static|async|readonly|get|set)\s+)*({_W})\s*\([^;{{]*\)\s*(?::\s*[^{{;]+)?\{{\s*$"),
]
_JVM = [
    re.compile(rf"^\s*{_MODS}*(?:data\s+|record\s+|annotation\s+)?(?:class|interface|enum|record|object|trait)\s+({_W})"),
    re.compile(rf"^\s*{_MODS}*(?:suspend\s+)?fun\s+(?:<[^>]+>\s*)?(?:[\w.<>?]+\.)?({_W})\s*\("),
    re.compile(rf"^\s*{_MODS}+[\w<>\[\],.?\s]+?\s+({_W})\s*\([^;]*$"),
    re.compile(rf"^\s*{_MODS}+[\w<>\[\],.?]+\s+({_W})\s*\([^;]*\)\s*(?:throws [\w, .]+)?\{{?\s*$"),
]
_GO = [
    re.compile(rf"^\s*func\s+(?:\([^)]*\)\s*)?({_W})"),
    re.compile(rf"^\s*type\s+({_W})\s+(?:struct|interface|func|\w)"),
    re.compile(rf"^\s*(?:const|var)\s+({_W})\b"),
]
_RS = [re.compile(rf"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:fn|struct|enum|trait|mod|type|const|static)\s+({_W})")]
_RB = [re.compile(rf"^\s*(?:def\s+(?:self\.)?|class\s+|module\s+)({_W}[?!=]?)")]
_PHP = [re.compile(rf"^\s*{_MODS}*(?:function|class|interface|trait|enum)\s+&?({_W})")]
_CS = _JVM + [re.compile(rf"^\s*{_MODS}*(?:struct|namespace)\s+({_W})")]
_C = [re.compile(rf"^\s*(?:static\s+|inline\s+|extern\s+)*[\w:<>*&\s]+?\s+\**({_W})\s*\([^;]*\)\s*\{{?\s*$"),
      re.compile(rf"^\s*(?:typedef\s+)?(?:struct|class|enum|union|namespace)\s+({_W})")]

_BY_EXT: Dict[str, List[re.Pattern]] = {}
for _exts, _pats in (
    (("py", "pyi"), _PY), (("js", "jsx", "ts", "tsx", "mjs", "cjs", "vue", "svelte"), _JS),
    (("java", "kt", "kts", "scala", "groovy"), _JVM), (("cs",), _CS), (("go",), _GO), (("rs",), _RS),
    (("rb",), _RB), (("php",), _PHP), (("c", "h", "cpp", "hpp", "cc", "cxx"), _C),
):
    for _e in _exts:
        _BY_EXT[_e] = _pats
_GENERIC = _PY + _JS[:3] + _JVM[:2] + _GO[:2] + _RS + _RB


def _symbol_name(path: str, line: str) -> Optional[str]:
    if not line or len(line) > 400:
        return None
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    for rx in _BY_EXT.get(ext, _GENERIC):
        m = rx.match(line)
        if m:
            name = m.group(1)
            if name in _NOT_SYMBOLS:
                continue
            return name
    return None


_NOT_SYMBOLS = frozenset({"if", "for", "while", "switch", "catch", "return", "else", "elif", "with", "new", "function", "constructor"})


# --------------------------------------------------------------------------- query understanding
_STOP = frozenset("""
a an the and or but if then else for from into onto with without within about above after again against all also am any are
as at be because been before being below between both by can cannot could did do does doing done down during each few
get got had has have having he her here hers him his how i in is it its itself just let like me more most my no nor not
now of off on once only other our out over own same she should so some such than that their them then there these they
this those through to too under until up us very was we were what when where which while who whom why will would you your
please show tell give find look search locate explain describe understand mean means need needs want wants make makes
code file files function functions class classes method methods module modules repo repository project app application
work works working used use uses using implement implemented implementation implements add adds added change changes
changed fix fixes fixed update updates updated create creates created new old thing things something somehow etc
""".split())
_SEMANTIC_RE = re.compile(r"\b(similar to|similar|like|concept|conceptually|semantic|semantically|meaning|everywhere|"
                          r"all places that|anything related|related to)\b", re.I)
_EXTS = ("py", "pyi", "js", "jsx", "ts", "tsx", "mjs", "java", "kt", "go", "rs", "cs", "rb", "php", "c", "h", "cpp", "hpp",
         "json", "yaml", "yml", "toml", "md", "txt", "sql", "html", "css", "scss", "sh", "cfg", "ini", "xml", "gradle", "vue")
_EXT_ALT = "|".join(sorted(_EXTS, key=len, reverse=True))
_QUOTED_RE = re.compile(r"`([^`\n]{2,80})`|\"([^\"\n]{2,80})\"|(?<!\w)'([^'\n]{2,80})'(?!\w)")
_PATH_SLASH_RE = re.compile(r"(?<![\w/.])((?:[\w.\-]+/)+[\w.\-]+)")
_PATH_FILE_RE = re.compile(rf"(?<![\w/.])([\w\-]+(?:\.[\w\-]+)*\.(?:{_EXT_ALT}))\b")
_EXT_HINT_RE = re.compile(rf"(?<![\w])\.({_EXT_ALT})\b")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\(\))?")
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_KIND_WEIGHT = {"quoted": 1.5, "path": 1.5, "ident": 1.2, "part": 0.5, "kw": 0.8}
_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ers", "er", "ed", "es", "s")


@dataclass
class _Term:
    text: str
    kind: str       # quoted|path|ident|part|kw

    @property
    def norm(self) -> str:
        return re.sub(r"[^a-z0-9]", "", self.text)


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _is_ident(tok: str) -> bool:
    if tok.endswith("()"):
        return True
    if "_" in tok.strip("_"):
        return True
    return bool(re.search(r"[a-z][A-Z]", tok) or re.match(r"^[A-Z][a-z0-9]+[A-Z]", tok))


def extract_terms(query: str) -> Tuple[List[_Term], List[str]]:
    """(terms, ext_hints). Deterministic order of first appearance, de-duplicated."""
    terms: List[_Term] = []
    seen: set = set()

    def add(text: str, kind: str):
        text = text.lower()
        if len(text) < 2 or text in seen:
            return
        seen.add(text)
        terms.append(_Term(text, kind))

    rest = query
    for m in _QUOTED_RE.finditer(query):
        q = next(g for g in m.groups() if g)
        add(q.strip(), "quoted")
    rest = _QUOTED_RE.sub(" ", rest)
    for rx in (_PATH_SLASH_RE, _PATH_FILE_RE):
        for m in rx.finditer(rest):
            add(m.group(1).strip("."), "path")
    exts = [m.group(1).lower() for m in _EXT_HINT_RE.finditer(rest)]
    for m in _WORD_RE.finditer(rest):
        tok = m.group(0)
        if _is_ident(tok):
            tok = tok.removesuffix("()")
            add(tok, "ident")
            for part in _CAMEL_SPLIT_RE.findall(tok.replace("_", " ")):
                p = part.lower()
                if len(p) >= 3 and p not in _STOP and not p.isdigit():
                    add(p, "part")
            continue
        low = tok.lower()
        if len(low) < 3 or low in _STOP or low in _EXTS:
            continue
        add(_stem(low), "kw")
    return terms[:24], sorted(set(exts))


# --------------------------------------------------------------------------- search
ActionCb = Callable[[str, str, dict], None]
_MAX_MATCHES_PER_FILE = 100
_MAX_SCAN_FILES = 500           # when terms are everywhere, scan only the best-named files (bounded latency)
_CONF_SCORE_THRESHOLD = 3.0


class AgenticSearch:
    def __init__(self, corpus: RepoCorpus, on_action: Optional[ActionCb] = None):
        self.corpus = corpus
        self.on_action = on_action
        self._glob_cache: Dict[str, Callable[[str], bool]] = {}

    # ---- instrumentation
    def _emit(self, action: str, detail: str, data: Optional[dict] = None, ms: Optional[int] = None) -> None:
        data = data or {}
        if self.on_action:
            try:
                self.on_action(action, detail, data)
            except Exception:  # noqa: BLE001 - instrumentation never breaks a search
                pass
        try:
            _log_action("RETRIEVE", "AgenticSearch", action, status="info", detail=detail, data=data, duration_ms=ms)
        except Exception:  # noqa: BLE001
            pass

    def _matcher(self, pattern: str) -> Callable[[str], bool]:
        m = self._glob_cache.get(pattern)
        if m is None:
            m = self._glob_cache[pattern] = _compile_globs(pattern)
        return m

    # ---- primitives
    def glob(self, pattern: str, limit: int = 200) -> List[str]:
        t0 = time.perf_counter()
        match = self._matcher(pattern)
        out = [p for p in self.corpus.paths() if match(p)][: max(0, limit)]
        self._emit("glob", f"glob {pattern} -> {len(out)} file(s)", {"pattern": pattern, "count": len(out)},
                   int((time.perf_counter() - t0) * 1000))
        return out

    def grep(self, pattern: str, path_glob: Optional[str] = None, regex: bool = True, ignore_case: bool = True,
             limit: int = 200) -> List[Hit]:
        t0 = time.perf_counter()
        flags = re.IGNORECASE if ignore_case else 0
        literal = not regex
        fallback = False
        if regex:
            try:
                rx = re.compile(pattern, flags | re.MULTILINE)
            except re.error:
                literal, fallback = True, True
        if literal:
            rx = re.compile(re.escape(pattern), flags | re.MULTILINE)
        match = self._matcher(path_glob) if path_glob else None
        hits: List[Hit] = []
        lit_lower = pattern.lower() if literal and ignore_case else None
        for path in self.corpus.paths():
            if len(hits) >= limit:
                break
            if match and not match(path):
                continue
            doc = self.corpus.doc(path)
            if lit_lower is not None and lit_lower not in doc.lower:
                continue
            if rx.search(doc.text) is None:
                continue
            for i, line in enumerate(doc.lines, 1):
                if rx.search(line):
                    hits.append(Hit(path, i, line.strip()[:300], 1.0))
                    if len(hits) >= limit:
                        break
        self._emit("grep", f"grep {pattern!r}" + (f" in {path_glob}" if path_glob else "") + f" -> {len(hits)} hit(s)"
                   + (" (invalid regex, used literal)" if fallback else ""),
                   {"pattern": pattern, "path_glob": path_glob, "count": len(hits), "literal_fallback": fallback},
                   int((time.perf_counter() - t0) * 1000))
        return hits

    def read(self, path: str, start: int = 1, end: Optional[int] = None) -> str:
        """Lines start..end (1-based, inclusive, clamped) as '  12| code'. Unknown file -> 'error: file not found: <path>'."""
        doc = self.corpus.doc(path)
        if doc is None:
            self._emit("read", f"read {path} (not found)", {"path": path, "found": False})
            return f"error: file not found: {path}"
        lines = doc.lines
        n = len(lines)
        if n and lines[-1] == "":
            n -= 1                                  # trailing newline is not a line
        s = max(1, int(start or 1))
        e = n if end is None else min(n, int(end))
        out = [f"{i:>4}| {lines[i - 1]}" for i in range(s, e + 1)] if s <= e else []
        self._emit("read", f"read {path}:{s}-{max(s, e)}", {"path": path, "start": s, "end": e, "lines": len(out)})
        return "\n".join(out)

    def ls(self, dir: str = "") -> List[str]:
        d = _norm_path(dir).rstrip("/")
        prefix = d + "/" if d else ""
        names: set = set()
        for p in self.corpus.paths():
            if not p.startswith(prefix):
                continue
            head, sep, _ = p[len(prefix):].partition("/")
            names.add(head + "/" if sep else head)
        out = sorted(names)
        self._emit("ls", f"ls {dir or '.'} -> {len(out)} entr(ies)", {"dir": dir, "count": len(out)})
        return out

    def symbols(self, name: Optional[str] = None) -> List[Hit]:
        t0 = time.perf_counter()
        want = name.lower() if name else None
        wn = re.sub(r"[^a-z0-9]", "", want) if want else None
        hits: List[Hit] = []
        for path in self.corpus.paths():
            doc = self.corpus.doc(path)
            if want and want not in doc.lower and (not wn or wn not in doc.lower.replace("_", "")):
                continue
            for i, line in enumerate(doc.lines, 1):
                sym = _symbol_name(path, line)
                if not sym:
                    continue
                if want:
                    sl = sym.lower()
                    sn = re.sub(r"[^a-z0-9]", "", sl)
                    if sl == want or sn == wn:
                        score = 3.0
                    elif want in sl or (wn and wn in sn):
                        score = 1.5
                    else:
                        continue
                else:
                    score = 1.0
                hits.append(Hit(path, i, line.strip()[:300], score))
        if want:
            hits.sort(key=lambda h: (-h.score, h.path, h.line))
        self._emit("symbols", f"symbols {name or '*'} -> {len(hits)}", {"name": name, "count": len(hits)},
                   int((time.perf_counter() - t0) * 1000))
        return hits

    # ---- ranking core
    def _rank(self, query: str, window: int, k_blocks: int) -> Tuple[List[Block], Dict[str, float], List[_Term], dict]:
        """Returns (blocks, file_scores, terms, info)."""
        terms, exts = extract_terms(query)
        paths = self.corpus.paths()
        n_files = len(paths)
        info = {"has_ident": any(t.kind in ("ident", "path", "quoted") for t in terms), "content_files": 0}
        if not terms or not n_files:
            return [], {}, terms, info

        # document frequency (file-level substring test on cached lowercase text)
        df: Dict[str, int] = {}
        present: Dict[str, List[str]] = {}          # path -> terms present in it
        for p in paths:
            low = self.corpus.doc(p).lower
            here = [t.text for t in terms if t.text in low]
            if here:
                present[p] = here
                for t in here:
                    df[t] = df.get(t, 0) + 1
        active = [t for t in terms if df.get(t.text)]
        if n_files >= 20 and len(active) > 1:       # drop ubiquitous keywords: no signal, lots of cost
            keep = [t for t in active if df[t.text] / n_files <= 0.5]
            if keep:
                active = keep
        idf = {t.text: 1.0 + math.log(n_files / df[t.text]) for t in active}
        weight = {t.text: _KIND_WEIGHT[t.kind] for t in terms}
        kind = {t.text: t.kind for t in terms}
        ident_norms = {t.norm: t for t in terms if t.kind in ("ident", "quoted", "kw") and len(t.norm) >= 3}
        ident_parts = [t for t in terms if t.kind in ("ident", "quoted") and len(t.norm) >= 4]

        rx = re.compile("|".join(re.escape(t.text) for t in sorted(active, key=lambda t: -len(t.text)))) if active else None

        file_scores: Dict[str, float] = {}
        path_scores: Dict[str, float] = {}
        blocks: List[Block] = []

        for p in paths:
            ps = self._path_score(p, terms, exts)
            if ps:
                path_scores[p] = ps

        act = {t.text for t in active}
        cands = [p for p, here in present.items() if rx is not None and act.intersection(here)]
        if len(cands) > _MAX_SCAN_FILES:
            cands.sort(key=lambda p: (-path_scores.get(p, 0.0), p))
            cands = cands[:_MAX_SCAN_FILES]
        for p in cands:
            doc = self.corpus.doc(p)
            low, lines = doc.lower, doc.lines
            line_terms: Dict[int, set] = {}
            cnt = 0
            for m in rx.finditer(low):
                ln = doc.line_of(m.start())
                line_terms.setdefault(ln, set()).add(m.group())
                cnt += 1
                if cnt >= _MAX_MATCHES_PER_FILE:
                    break
            if not line_terms:
                continue
            scored: List[Tuple[int, float, str]] = []
            for ln, ts in line_terms.items():
                s = sum(weight[t] * idf[t] for t in ts)
                tag = ",".join(sorted(ts))
                line = lines[ln - 1] if ln - 1 < len(lines) else ""
                sym = _symbol_name(p, line)
                if sym:
                    sn = re.sub(r"[^a-z0-9]", "", sym.lower())
                    if sn in ident_norms:
                        s += 3.0
                        tag = f"definition of {sym}"
                    elif any(it.norm in sn for it in ident_parts):
                        s += 1.5
                        tag = f"definition {sym} (partial name match)"
                scored.append((ln, s, tag))
            info["content_files"] += 1
            scored.sort(key=lambda x: (-x[1], x[0]))
            top = scored[:8]
            file_scores[p] = sum(s for _, s, _ in top[:5]) + path_scores.get(p, 0.0)
            blocks.extend(self._blocks_for(p, doc, sorted(top), window, path_scores.get(p, 0.0)))

        for p, ps in path_scores.items():
            if p not in file_scores and ps >= 3.0:          # filename/path matched but no content term: show its head
                doc = self.corpus.doc(p)
                n = max(1, len(doc.lines))
                end = min(n, window)
                blocks.append(Block(p, 1, end, "\n".join(doc.lines[:end]), ps, "path match"))
                file_scores[p] = ps

        blocks.sort(key=lambda b: (-b.score, b.path, b.start))
        picked: List[Block] = []
        per_file: Dict[str, int] = {}
        for b in blocks:
            if per_file.get(b.path, 0) >= 2:
                continue
            per_file[b.path] = per_file.get(b.path, 0) + 1
            picked.append(b)
            if len(picked) >= k_blocks:
                break
        return picked, file_scores, terms, info

    @staticmethod
    def _path_score(path: str, terms: List[_Term], exts: List[str]) -> float:
        low = path.lower()
        base = low.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0] if "." in base else base
        stem_n = re.sub(r"[^a-z0-9]", "", stem)
        score = 0.0
        for t in terms:
            if t.kind == "path":
                if low == t.text or low.endswith("/" + t.text):
                    score += 6.0
                elif t.text in low:
                    score += 3.0
                continue
            tn = t.norm
            if len(tn) < 3:
                continue
            if tn == stem_n:
                score += 4.0 if t.kind != "part" else 1.0
            elif tn in stem_n:
                score += {"ident": 2.0, "quoted": 2.0, "kw": 1.0, "part": 0.5}.get(t.kind, 0.5)
            elif t.kind in ("ident", "quoted") and t.text in low:
                score += 1.0
        if score and exts and base.rsplit(".", 1)[-1] in exts:
            score += 0.5
        return score

    @staticmethod
    def _blocks_for(path: str, doc: _Doc, hits: List[Tuple[int, float, str]], window: int,
                    path_score: float) -> List[Block]:
        n = max(1, len(doc.lines))
        if doc.lines and doc.lines[-1] == "":
            n = max(1, n - 1)
        before = max(1, window // 3)
        after = max(1, window - before)
        groups: List[List[Tuple[int, float, str]]] = []
        cur: List[Tuple[int, float, str]] = []
        cur_start = cur_end = 0
        for ln, s, tag in hits:                                 # hits sorted by line
            st, en = max(1, ln - before), min(n, ln + after)
            if cur and st <= cur_end + 1 and (en - cur_start + 1) <= 2 * window:
                cur.append((ln, s, tag))
                cur_end = max(cur_end, en)
            else:
                if cur:
                    groups.append(cur)
                cur, cur_start, cur_end = [(ln, s, tag)], st, en
        if cur:
            groups.append(cur)
        out: List[Block] = []
        for g in groups:
            st = max(1, g[0][0] - before)
            en = min(n, max(h[0] for h in g) + after)
            score = sum(h[1] for h in g) + path_score
            best = max(g, key=lambda h: h[1])
            why = best[2] if best[2].startswith("definition") else "terms: " + best[2]
            if path_score:
                why += "; filename match"
            out.append(Block(path, st, en, "\n".join(doc.lines[st - 1:en]), round(score, 4), why))
        return out

    def locate_files(self, query: str, k: int = 8) -> List[str]:
        t0 = time.perf_counter()
        blocks, fscores, _terms, _info = self._rank(query, 30, 10_000)
        ranked = sorted(fscores.items(), key=lambda kv: (-kv[1], kv[0]))
        out = [p for p, _ in ranked[: max(0, k)]]
        self._emit("locate_files", f"locate_files -> {len(out)} file(s)", {"files": out[:k]},
                   int((time.perf_counter() - t0) * 1000))
        return out

    def retrieve(self, query: str, k_blocks: int = 6, window: int = 30) -> SearchResult:
        t0 = time.perf_counter()
        blocks, fscores, terms, info = self._rank(query, window, k_blocks)
        n_files = len(self.corpus)
        reasons: List[str] = []
        conf = 1.0
        if not blocks:
            conf = 0.0
            reasons.append("No file path, symbol or text in the repo matched the query terms.")
        else:
            if not info["has_ident"]:
                conf -= 0.35
                reasons.append("The question is conceptual (no identifier, file name or quoted string to search for).")
            if blocks[0].score < _CONF_SCORE_THRESHOLD:
                conf -= 0.3
                reasons.append(f"Best match is weak (score {blocks[0].score:.1f} < {_CONF_SCORE_THRESHOLD:.1f}).")
            if len(fscores) > 25:
                vals = sorted(fscores.values(), reverse=True)
                median = vals[len(vals) // 2]
                if median > 0 and vals[0] / median < 1.5:
                    conf -= 0.2
                    reasons.append(f"Matches are spread across {len(fscores)} files with no clear winner.")
            if n_files > 5000:
                conf -= 0.1
                reasons.append(f"Large repository ({n_files} files); keyword search is less reliable.")
            if _SEMANTIC_RE.search(query):
                conf -= 0.25       # 0.25 (not 0.2): conceptual + similarity wording must land STRICTLY below the 0.45 default threshold
                reasons.append("The query asks for similarity/meaning, which keyword search cannot judge.")
            if conf >= 0.999 and not reasons:
                reasons.append("Query names specific identifiers/files and strong matches were found.")
        conf = max(0.0, min(1.0, conf))
        ms = int((time.perf_counter() - t0) * 1000)
        res = SearchResult(blocks=blocks, confidence=round(conf, 3), reasons=reasons,
                           terms=[t.text for t in terms], files_considered=len(fscores), elapsed_ms=ms)
        self._emit("retrieve", f"retrieve -> {len(blocks)} block(s), confidence {res.confidence:.2f}",
                   {"terms": res.terms[:12], "blocks": [f"{b.path}:{b.start}-{b.end}" for b in blocks],
                    "confidence": res.confidence, "reasons": reasons[:4]}, ms)
        return res


def needs_vectors(result: SearchResult, threshold: float = 0.45) -> Tuple[bool, str]:
    """(True, reason) when the agentic result is too weak to trust and a vector index would likely help."""
    if result.confidence < threshold:
        why = " ".join(result.reasons) or "low confidence"
        return True, f"Agentic search confidence {result.confidence:.2f} is below {threshold:.2f}: {why}"
    return False, ""
