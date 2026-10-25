"""Function-level edits: replace ONE function or method by its AST range.

The model writes the whole new definition; nothing has to match the old text, so the SEARCH-mismatch failures cannot happen.
Pure functions: no model calls, no I/O.
"""
from __future__ import annotations

import ast
import re
import textwrap
from typing import Optional, Tuple

from services import symbols as sym

_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.S)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def unfence(code: str) -> str:
    m = _FENCE.search(code)
    return m.group(1) if m else code


def import_index(tree: ast.Module) -> int:
    """List index where new imports go: after the last top-level import, else after the module docstring, else 0."""
    last = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    if not last and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        last = tree.body[0].end_lineno
    return last


def _text(src) -> str:
    return (src.decode("utf-8", "replace") if isinstance(src, bytes) else src).replace("\r\n", "\n")


def symbol_at(src, line: int) -> Optional[sym.Symbol]:
    """The innermost function or method that contains `line` (None for module- and class-level lines)."""
    best = None
    for s in sym.symbols(_text(src)):
        if s.kind != "class" and s.start <= line <= s.end and (best is None or s.end - s.start < best.end - best.start):
            best = s
    return best


def slice_of(src, qualname: str, limit: int = 6000) -> Optional[str]:
    """Source of one symbol (decorators included), dedented, for a prompt."""
    text = _text(src)
    lines = text.split("\n")
    for s in sym.symbols(text):
        if s.qualname == qualname:
            out = textwrap.dedent("\n".join(lines[s.start - 1:s.end]))
            return out if len(out) <= limit else out[:limit] + "\n    # ... (truncated)"
    return None


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def replace_symbol(src: bytes, qualname: str, code: str, keep_sig: str = "") -> Tuple[Optional[bytes], str]:
    """Replace `qualname` in `src` by the definition in `code`.
    `code` may hold import lines, the definition, and small helper definitions (inserted right after it).
    -> (new source, '') or (None, what is wrong with the reply)."""
    raw = src.decode("utf-8", "replace")
    crlf = "\r\n" in raw
    lines = raw.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    table = {s.qualname: s for s in sym.symbols("\n".join(lines))}
    old = table.get(qualname)
    if old is None or old.kind == "class":
        return None, f"{qualname} does not exist in the file"
    body = textwrap.dedent(unfence(code)).strip("\n")
    if not body.strip():
        return None, "the reply is empty: output the complete function definition"
    try:
        new_tree = ast.parse(body)
    except SyntaxError as e:
        return None, f"your definition does not parse (line {e.lineno}: {e.msg})"
    name = qualname.rsplit(".", 1)[-1]
    imports, target, extras, stray = [], None, [], False
    for n in new_tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(n))
        elif isinstance(n, _FUNCS) and n.name == name and target is None:
            target = n
        elif isinstance(n, _FUNCS + (ast.ClassDef,)):
            extras.append(n)
        elif not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)):
            stray = True
    if target is None:
        found = [n.name for n in extras] or "nothing"
        return None, f"the reply must define `{name}` (found: {found if isinstance(found, str) else ', '.join(found)})"
    if stray:
        return None, "only import lines and function definitions are allowed in the reply: no module-level statements"
    if keep_sig and sym.header(target) != keep_sig:
        return None, f"the signature must stay exactly `{keep_sig}` but you wrote `{sym.header(target)}`"
    parent = qualname.rsplit(".", 1)[0] + "." if "." in qualname else ""
    clash = [e.name for e in extras if parent + e.name in table]
    if clash:
        return None, f"{', '.join(clash)} already exist in the file: do not redefine them"

    body_lines = body.split("\n")

    def block(n) -> list:
        first = min([d.lineno for d in n.decorator_list] + [n.lineno])
        return body_lines[first - 1:n.end_lineno]

    ind = _indent(lines[old.line - 1])
    gap = [""] if old.kind == "method" else ["", ""]
    new_lines = [ind + l if l.strip() else l for l in block(target)]
    for e in extras:
        new_lines += gap + [ind + l if l.strip() else l for l in block(e)]
    start = old.start if target.decorator_list else old.line      # no decorators in the reply: keep the existing ones
    lines[start - 1:old.end] = new_lines

    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as e:
        return None, f"the file would not parse after the replacement (line {e.lineno}: {e.msg})"
    have = {ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    want = [i for i in dict.fromkeys(imports) if i not in have]
    if want:
        idx = import_index(tree)
        lines[idx:idx] = want + ([""] if idx == 0 else [])
    out = "\n".join(lines) + "\n"
    try:
        ast.parse(out)
    except SyntaxError as e:
        return None, f"the file would not parse after adding the imports (line {e.lineno}: {e.msg})"
    return (out.replace("\n", "\r\n") if crlf else out).encode("utf-8"), ""
