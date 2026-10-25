"""A rewritten function must keep calling what it called before, unless the task says otherwise.
Typical slip: `return await self._route_chat(...)` becomes a direct call to the model, and intent routing is gone.
Pure function: no model calls, no I/O."""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""                                   # a call on a call, a subscript, ...: no stable name


def _calls(fn) -> Set[str]:
    return {d for c in ast.walk(fn) if isinstance(c, ast.Call) for d in [_dotted(c.func)] if d}


def _find(tree: ast.AST, qualname: str):
    """-> (function node, {method name: node} of its class or {})"""
    cls_name, _, name = qualname.rpartition(".")
    for n in ast.walk(tree):
        if cls_name and isinstance(n, ast.ClassDef) and n.name == cls_name.split(".")[-1]:
            methods = {m.name: m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if name in methods:
                return methods[name], methods
        if not cls_name and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n, {}
    return None, {}


def _module_names(tree: ast.Module) -> Set[str]:
    out: Set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Assign):
            out |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return out


def dropped_calls(old_src: bytes, new_src: bytes, qualname: str, behavior: str = "") -> List[str]:
    """Calls of the old function (to `self.<x>` or to a name of its module) that the new one, with the new helpers it reaches, no
    longer makes. A call the task's `behavior` text names is allowed to go."""
    try:
        old_tree, new_tree = ast.parse(old_src.decode("utf-8", "replace")), ast.parse(new_src.decode("utf-8", "replace"))
    except (SyntaxError, ValueError):
        return []
    old_fn, old_methods = _find(old_tree, qualname)
    new_fn, new_methods = _find(new_tree, qualname)
    if old_fn is None or new_fn is None:
        return []                                # a new symbol, or one that cannot be compared
    mods = _module_names(old_tree)
    seen, todo, now = {qualname.rpartition(".")[2]}, [new_fn], set()
    while todo:                                  # the function and the helpers that did not exist before
        fn = todo.pop()
        calls = _calls(fn)
        now |= calls
        for c in calls:
            h = c[5:] if c.startswith("self.") else ""
            if h and "." not in h and h in new_methods and h not in old_methods and h not in seen:
                seen.add(h)
                todo.append(new_methods[h])
    said = behavior.lower()
    gone = sorted(c for c in _calls(old_fn) - now
                  if (c.startswith("self.") or c.split(".")[0] in mods) and c.rpartition(".")[2].lower() not in said)
    return [f"the old code called `{c}(...)`; the new code (with its helpers) never does. Keep that call (wrap it, do not replace it) "
            "unless the behaviour says to remove it" for c in gone[:5]]
