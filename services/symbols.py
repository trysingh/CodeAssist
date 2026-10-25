"""Python symbol table: typed signatures and exact line ranges, from the AST.

Prompts show signatures instead of bodies (small context); stubs and function-level edits use the line ranges.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class Symbol:
    qualname: str       # "func" | "Cls" | "Cls.method"
    kind: str           # function | method | class
    sig: str            # "async def f(a: int) -> str" | "class C(Base)"
    doc: str            # first docstring line, may be empty
    start: int          # 1-based, first decorator line included
    end: int            # 1-based, inclusive
    line: int = 0       # 1-based line of the def / class keyword
    params: str = ""    # compact: "a, b=, *args" (self/cls dropped, '=' marks a default)

    @property
    def depth(self) -> int:
        return self.qualname.count(".")


def _doc(n) -> str:
    d = ast.get_docstring(n) or ""
    return d.strip().splitlines()[0][:100] if d.strip() else ""


def _sig(n) -> str:
    if isinstance(n, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in n.bases)
        return f"class {n.name}" + (f"({bases})" if bases else "")
    ret = f" -> {ast.unparse(n.returns)}" if n.returns else ""
    return f"{'async ' if isinstance(n, ast.AsyncFunctionDef) else ''}def {n.name}({ast.unparse(n.args)}){ret}"


def header(n) -> str:
    """Normalised `def` / `class` header of an AST node."""
    return _sig(n)


def _params(a: ast.arguments) -> str:
    pos = [x.arg for x in a.posonlyargs + a.args]
    n_def = len(a.defaults)
    out = [p + ("=" if i >= len(pos) - n_def else "") for i, p in enumerate(pos)]
    if out and out[0].rstrip("=") in ("self", "cls"):
        out = out[1:]
    if a.vararg:
        out.append("*" + a.vararg.arg)
    out += [k.arg + ("=" if d is not None else "") for k, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        out.append("**" + a.kwarg.arg)
    return ", ".join(out)


def normalize_sig(text: str) -> Optional[Tuple[str, str, str]]:
    """'def f(a:int)->str' -> ('def', 'def f(a: int) -> str', 'f'); None when it is not one plain header."""
    t = (text or "").strip().rstrip(":").strip()
    try:
        tree = ast.parse(t + ":\n    pass")
    except (SyntaxError, ValueError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    n = tree.body[0]
    if n.decorator_list:
        return None
    return ("class" if isinstance(n, ast.ClassDef) else "def"), _sig(n), n.name


def symbols(src: Union[bytes, str]) -> List[Symbol]:
    """Top-level functions, classes and their methods. Unparseable source gives []."""
    text = src.decode("utf-8", "replace") if isinstance(src, bytes) else src
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: List[Symbol] = []

    def walk(body, prefix: str, in_class: bool) -> None:
        for n in body:
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = min([d.lineno for d in n.decorator_list] + [n.lineno])
            if isinstance(n, ast.ClassDef):
                out.append(Symbol(prefix + n.name, "class", _sig(n), _doc(n), start, n.end_lineno, n.lineno))
                walk(n.body, prefix + n.name + ".", True)
            else:
                out.append(Symbol(prefix + n.name, "method" if in_class else "function", _sig(n), _doc(n),
                                  start, n.end_lineno, n.lineno, _params(n.args)))

    walk(tree.body, "", False)
    return out


def render(syms: List[Symbol]) -> str:
    """One line per symbol, indented by nesting: the shape of a file without its bodies."""
    return "\n".join("    " * s.depth + s.sig + (f"  # {s.doc}" if s.doc else "") for s in syms)


def compact(syms: List[Symbol], limit: int = 40) -> str:
    """'f(a, b=), Cls, Cls.m(x)': names with their parameters, for the repo-wide code map."""
    return ", ".join(s.qualname + (f"({s.params})" if s.kind != "class" else "") for s in syms[:limit])
