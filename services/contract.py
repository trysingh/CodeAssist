"""Interface contract for a change. Pure functions: no model calls, no I/O.

The designer model lists every NEW or CHANGED symbol (file, typed signature, behaviour, dependencies). Here that reply is
validated against the real repo, turned into stubs (`raise NotImplementedError("contract: <name>")`) held in memory, and its
dependencies are resolved. Mistakes between files show up before any body exists.
"""
from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from services import symbols as sym

STUB_RE = re.compile(r'raise NotImplementedError\("contract: ([\w.]+)"\)')


@dataclass
class CSym:
    file: str
    name: str                                   # "func" | "Cls" | "Cls.method"
    sig: str                                    # normalised header
    kind: str                                   # function | method | class
    behavior: str
    deps: List[str] = field(default_factory=list)       # "path::name"
    imports: List[str] = field(default_factory=list)    # import lines the signature's types need
    change: str = "new"                         # new | signature (stubbed) | behavior (body changes, signature does not)


@dataclass
class Shared:
    """State that several symbols use. A task cannot write module-level code, so it is always a `self.` attribute."""
    attr: str                                   # "Class.attribute"
    file: str
    type: str
    purpose: str
    created_in: str                             # "Class.method" (normally Class.__init__): the contract symbol that assigns it
    used_by: List[str] = field(default_factory=list)    # "path::name" contract symbols that read it
    default: str = ""                           # the Python expression the creator assigns, e.g. "{}" (optional)
    lazy: bool = False                          # existing tests build the class with __new__: __init__ does not always run


def _is_test(p: str) -> bool:
    b = p.rsplit("/", 1)[-1]
    return p.startswith("tests/") or "/tests/" in p or b.startswith("test_") or b.endswith("_test.py")


def _safe(p: str) -> bool:
    return bool(p) and posixpath.normpath(p) == p and not p.startswith("/") and ".." not in p.split("/") and p.endswith(".py")


def parse(data, sources: Dict[str, bytes], known: Sequence[str], max_symbols: int = 15) -> Tuple[List[CSym], List[str]]:
    """The designer's JSON -> (symbols, problems). Any problem means: ask the designer again."""
    if not isinstance(data, dict) or not isinstance(data.get("symbols"), list):
        return [], ['the reply must be JSON: {"symbols": [{"file", "name", "signature", "behavior", "depends_on", "imports"}]}']
    table = {p: {s.qualname: s for s in sym.symbols(src)} for p, src in sources.items() if p.endswith(".py")}
    known_set = set(known)
    dirs = {p.rsplit("/", 1)[0] if "/" in p else "" for p in known}
    items, problems, out, seen = data["symbols"], [], [], set()
    if len(items) > max_symbols:
        problems.append(f"{len(items)} symbols is too many (max {max_symbols}): list only what the request needs")
        items = items[:max_symbols]
    for i, it in enumerate(items, 1):
        tag = f"symbol {i}"
        if not isinstance(it, dict):
            problems.append(f"{tag}: must be an object")
            continue
        f, name, sig, beh = (str(it.get(k) or "").strip() for k in ("file", "name", "signature", "behavior"))
        f = f.replace("\\", "/")
        if not _safe(f):
            problems.append(f"{tag}: bad file path '{f}' (a repo-relative .py path)")
            continue
        if _is_test(f):
            problems.append(f"{tag}: {f} is a test file; tests are written separately")
            continue
        if f not in known_set:
            same = [k for k in known if k.rsplit("/", 1)[-1] == f.rsplit("/", 1)[-1]]
            if same:
                problems.append(f"{tag}: '{f}' is not in the repo; did you mean {same[0]}?")
                continue
            if (f.rsplit("/", 1)[0] if "/" in f else "") not in dirs:
                problems.append(f"{tag}: new file {f} must live in an existing directory ({', '.join(sorted(dirs)[:6])})")
                continue
        ns = sym.normalize_sig(sig)
        if ns is None:
            problems.append(f"{tag} ({f}::{name}): signature must be ONE `def`, `async def` or `class` header, e.g. "
                            "`def f(a: int) -> str`, without decorators, colon or body")
            continue
        head, nsig, hname = ns
        if not name or name.split(".")[-1] != hname:
            problems.append(f"{tag}: name '{name}' does not match the signature's name '{hname}' (methods are Class.method)")
            continue
        if (f, name) in seen:
            problems.append(f"{tag}: {f}::{name} is listed twice")
            continue
        seen.add((f, name))
        if not beh:
            problems.append(f"{tag} ({f}::{name}): behavior is empty (one sentence)")
            continue
        kind = "class" if head == "class" else ("method" if "." in name else "function")
        existing = table.get(f, {}).get(name)
        if existing is not None and kind == "class":
            continue                                    # the class is already there
        change = "new" if existing is None else ("behavior" if existing.sig == nsig else "signature")
        deps = [str(d).strip() for d in (it.get("depends_on") or []) if isinstance(d, str)]
        imports, bad = [], False
        for line in (it.get("imports") or []):
            try:
                node = ast.parse(str(line)).body[0]
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError
                imports.append(ast.unparse(node))
            except (SyntaxError, ValueError, IndexError):
                problems.append(f"{tag} ({f}::{name}): imports must be plain import lines, got {str(line)[:60]!r}")
                bad = True
        if not bad:
            out.append(CSym(f, name, nsig, kind, beh, deps, imports, change))
    new_classes = {(s.file, s.name) for s in out if s.kind == "class"}
    for s in out:
        if s.kind == "method":
            parent = s.name.rsplit(".", 1)[0]
            pc = table.get(s.file, {}).get(parent)
            if not ((pc is not None and pc.kind == "class") or (s.file, parent) in new_classes):
                problems.append(f"{s.file}::{s.name}: class {parent} does not exist in {s.file}; list the class too or fix the name")
    return out, problems


def check_deps(syms: Sequence[CSym], tree: Dict[str, bytes]) -> List[str]:
    """Every `path::name` dependency must resolve in the repo with the stubs applied (so new symbols resolve too)."""
    cache: Dict[str, object] = {}

    def names(p: str):
        if p not in cache:
            cache[p] = {s.qualname for s in sym.symbols(tree[p])} if p in tree else None
        return cache[p]

    probs: List[str] = []
    for s in syms:
        for d in s.deps:
            f, sep, q = d.partition("::")
            f, q = f.strip().replace("\\", "/"), q.strip()
            if not sep or not f.endswith(".py"):
                continue                                # external (stdlib / third party): not ours to check
            ns = names(f)
            if ns is None:
                probs.append(f"{s.file}::{s.name} depends on {d}: file {f} does not exist")
            elif q not in ns:
                probs.append(f"{s.file}::{s.name} depends on {d}: {f} has no '{q}' (it defines: {', '.join(sorted(ns)[:12])})")
    return probs


def _doc(behavior: str) -> str:
    return " ".join(behavior.replace("\\", "/").replace('"""', "'''").split()).rstrip('"')[:200]


def _stub(s: CSym, indent: str) -> List[str]:
    body = indent + "    "
    lines = [f"{indent}{s.sig}:", f'{body}"""{_doc(s.behavior)}"""']
    if s.kind != "class":
        lines.append(f'{body}raise NotImplementedError("contract: {s.name}")')
    return lines


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def apply_stubs(syms: Sequence[CSym], sources: Dict[str, bytes]) -> Dict[str, bytes]:
    """New and signature-changed symbols as stubs. Returns only the files that changed (new files included).
    A symbol whose signature is unchanged keeps its body: nothing to stub, nothing to break."""
    by_file: Dict[str, List[CSym]] = {}
    for s in syms:
        if s.change != "behavior":
            by_file.setdefault(s.file, []).append(s)
    out: Dict[str, bytes] = {}
    for f, items in by_file.items():
        raw = sources.get(f, b"").decode("utf-8", "replace")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        was_empty = not lines
        table = {x.qualname: x for x in sym.symbols(text)}
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = ast.parse("")
        repl: List[Tuple[int, int, List[str]]] = []     # (start index, end index exclusive, new lines)
        ins: Dict[int, List[str]] = {}                  # index -> lines inserted before lines[index]
        new_classes = {s.name for s in items if s.kind == "class"}
        tail: List[str] = []
        for s in items:
            if s.change == "signature":
                old = table[s.name]
                repl.append((old.line - 1, old.end, _stub(s, _indent(lines[old.line - 1]))))
            elif s.kind == "method":
                parent = s.name.rsplit(".", 1)[0]
                if parent in new_classes:
                    continue                            # written together with its class
                p = table[parent]
                ins.setdefault(p.end, []).extend([""] + _stub(s, _indent(lines[p.line - 1]) + "    "))
            elif s.kind == "class":
                block = _stub(s, "")
                for m in items:
                    if m.kind == "method" and m.name.rsplit(".", 1)[0] == s.name:
                        block += [""] + _stub(m, "    ")
                tail += ["", ""] + block
            else:
                tail += ["", ""] + _stub(s, "")
        if tail:
            main = next((n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)), None)
            if main:                                    # new code goes above `if __name__ == ...`, replacing the blank run there
                at = pos = main.lineno - 1
                while pos > 0 and not lines[pos - 1].strip():
                    pos -= 1
                repl.append((pos, at, tail + ["", ""]))
            else:
                ins.setdefault(len(lines), []).extend(tail)
        have = {ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
        want: List[str] = []
        for s in items:
            for imp in s.imports:
                if imp not in have and imp not in want:
                    want.append(imp)
        if want:
            last = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
            if not last and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant) \
                    and isinstance(tree.body[0].value.value, str):
                last = tree.body[0].end_lineno          # after the module docstring
            ins[last] = want + ins.get(last, [])
        ops = [(a, 0, "r", (a, b, new)) for a, b, new in repl] + [(i, 1, "i", (i, i, new)) for i, new in ins.items()]
        for _k, _t, kind, (a, b, new) in sorted(ops, key=lambda o: (-o[0], o[1])):
            lines[a:b] = new
        while was_empty and lines and lines[0] == "":
            lines.pop(0)
        res = "\n".join(lines) + "\n"
        out[f] = (res.replace("\n", "\r\n") if crlf else res).encode("utf-8")
    return out


def render(syms: Sequence[CSym], limit: int = 6000) -> str:
    """The contract as the implementer sees it."""
    rows, used = [], 0
    for s in syms:
        row = f"- {s.file}::{s.name} [{s.change}]  {s.sig}  # {_doc(s.behavior)}"
        if s.deps:
            row += "  (uses: " + ", ".join(s.deps[:6]) + ")"
        if used + len(row) > limit:
            break
        rows.append(row)
        used += len(row)
    return "\n".join(rows)


_ATTR = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")


def parse_shared(data, syms: Sequence[CSym], sources: Dict[str, bytes]) -> Tuple[List[Shared], List[str]]:
    """The designer's `shared` list -> (valid items, problems). Invalid items are left out of the result."""
    items = data.get("shared") if isinstance(data, dict) else None
    if not items:
        return [], []
    if not isinstance(items, list):
        return [], ['"shared" must be a list of {"attr", "file", "type", "purpose", "created_in", "used_by"}']
    have = {(s.file, s.name) for s in syms}
    new_classes = {(s.file, s.name) for s in syms if s.kind == "class"}
    out, problems, seen = [], [], set()
    for i, it in enumerate(items[:8], 1):
        tag = f"shared {i}"
        if not isinstance(it, dict):
            problems.append(f"{tag}: must be an object")
            continue
        attr, f, typ, why, made = (str(it.get(k) or "").strip() for k in ("attr", "file", "type", "purpose", "created_in"))
        f = f.replace("\\", "/")
        if not _ATTR.match(attr):
            problems.append(f"{tag}: attr '{attr}' must be `Class.attribute`: a task cannot write module-level code, so shared state lives on `self`")
            continue
        cls = attr.split(".")[0]
        if not _safe(f) or _is_test(f):
            problems.append(f"{tag} ({attr}): bad file path '{f}'")
            continue
        exists = any(x.qualname == cls and x.kind == "class" for x in sym.symbols(sources.get(f, b"")))
        if not (exists or (f, cls) in new_classes):
            problems.append(f"{tag} ({attr}): class {cls} is not in {f}; put the state on an existing class (or list a new class in symbols). "
                            "Do not create a module just for it")
            continue
        if (f, made) not in have or not made.startswith(cls + "."):
            problems.append(f"{tag} ({attr}): created_in '{made}' must be a `{cls}.<method>` that is also listed in symbols for {f} "
                            f"(normally {cls}.__init__)")
            continue
        if attr in seen:
            continue
        seen.add(attr)
        users = []
        for u in it.get("used_by") or []:
            uf, _, un = str(u).replace("\\", "/").partition("::")
            if (uf.strip(), un.strip()) in have and f"{uf.strip()}::{un.strip()}" not in users:
                users.append(f"{uf.strip()}::{un.strip()}")       # a name that is not a contract symbol is dropped, not an error
        dflt = str(it.get("default") or "").strip()
        try:
            ast.parse(dflt, mode="eval") if dflt else None
        except SyntaxError:
            dflt = ""                                         # a default that is not an expression is ignored, not an error
        out.append(Shared(attr, f, typ or "object", why, made, users, dflt))
    return out, problems


def link_shared(syms: Sequence[CSym], shared: Sequence[Shared]) -> None:
    """Every user depends on the symbol that creates the state: it is implemented first, and if it fails the users are skipped."""
    by_key = {f"{s.file}::{s.name}": s for s in syms}
    for x in shared:
        maker = f"{x.file}::{x.created_in}"
        for u in x.used_by:
            s = by_key.get(u)
            if s is not None and u != maker and maker not in s.deps:
                s.deps.append(maker)


def render_shared(shared: Sequence[Shared], me: Optional[Tuple[str, str]] = None, limit: int = 2500) -> str:
    """The shared state as an implementer sees it. `me` = (file, name) of the symbol being written."""
    rows, used = [], 0
    for x in shared:
        cls, _, attr = x.attr.partition(".")
        who = "YOU assign it in this function" if me == (x.file, x.created_in) else f"assigned in {x.created_in}"
        row = f"- self.{attr}: {x.type}   (on {cls} in {x.file}; {who}) {x.purpose}".rstrip()
        if x.lazy:
            read = f"self.__dict__.setdefault({attr!r}, {x.default})" if x.default else f"getattr(self, {attr!r}, None) (and cope with None)"
            row += (f"\n  ! Existing tests build {cls} with {cls}.__new__, so __init__ may not have run. Outside {x.created_in}, never use "
                    f"self.{attr} directly: read it as {read}")
        if used + len(row) > limit:
            break
        rows.append(row)
        used += len(row)
    if not rows:
        return ""
    return ("SHARED STATE (use exactly these names; invent no other state, no module-level globals, no new module for it):\n"
            + "\n".join(rows))


_NEW_BYPASS = (re.compile(r"\b(\w+)\.__new__\(\s*\1\s*\)"), re.compile(r"\bobject\.__new__\(\s*(\w+)\s*\)"))


def init_bypassed(sources: Dict[str, bytes]) -> set:
    """Classes the repo's tests build with `Cls.__new__(Cls)`: their __init__ does not run there."""
    out = set()
    for p, src in sources.items():
        if p.endswith(".py") and _is_test(p):
            text = src.decode("utf-8", "replace")
            for rx in _NEW_BYPASS:
                out |= set(rx.findall(text))
    return out


def mark_lazy(shared: Sequence[Shared], bypassed: set) -> None:
    for x in shared:
        x.lazy = x.attr.split(".")[0] in bypassed


def shared_problems(t, new: bytes, shared: Sequence[Shared] = ()) -> List[str]:
    """What a task owes the shared state. The creator (`t.creates`) must assign the attribute. When existing tests build the
    class without __init__ (`x.lazy`), every other method of that class must read it defensively."""
    creates = getattr(t, "creates", None) or []
    cls, _, meth = t.name.rpartition(".")
    lazy = [x.attr.partition(".")[2] for x in shared if x.lazy and x.file == t.file and cls and x.attr.split(".")[0] == cls.split(".")[-1]]
    if not (creates or lazy):
        return []
    try:
        tree = ast.parse(new.decode("utf-8", "replace"))
    except SyntaxError:
        return []
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef) and c.name == cls.split(".")[-1]:
            for f in c.body:
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == meth:
                    uses = [n for n in ast.walk(f) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"]
                    stored = {n.attr for n in uses if isinstance(n.ctx, ast.Store)}
                    loaded = {n.attr for n in uses if isinstance(n.ctx, ast.Load)}
                    out = [f"`self.{a}` is shared state (see SHARED STATE): this function must assign it" for a in creates if a not in stored]
                    out += [f"`self.{a}` is read directly, but existing tests build this class without __init__ (see SHARED STATE): "
                            f"read it with the form given there" for a in lazy if a in loaded and a not in stored]
                    return out
    return []
