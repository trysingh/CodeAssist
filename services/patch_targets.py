"""Static checks of a generated test before it runs. A generated test may only patch what exists: `patch("pkg.mod.Name")`, `patch.object(mod, "name")`, `monkeypatch.setattr(...)`.
A wrong target fails every test at setup, which looks 'red' but says nothing about the change.
Pure function: no model calls, no I/O."""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

_BLOCKS = ("body", "orelse", "finalbody")


class _Mod:
    def __init__(self, src: bytes):
        self.names: Set[str] = set()
        self.classes: Dict[str, Tuple[Set[str], bool]] = {}     # class -> (class-level members, has bases)
        self.open = False                                       # `import *` or module `__getattr__`: names are not knowable
        self._walk(ast.parse(src.decode("utf-8", "replace")).body)

    def _walk(self, body) -> None:
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.names.add(n.name)
                self.open |= n.name == "__getattr__"
            elif isinstance(n, ast.ClassDef):
                self.names.add(n.name)
                members: Set[str] = set()
                for b in n.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        members.add(b.name)
                    elif isinstance(b, ast.Assign):
                        members |= {t.id for t in b.targets if isinstance(t, ast.Name)}
                    elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                        members.add(b.target.id)
                self.classes[n.name] = (members, bool(n.bases))
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    self.names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            elif isinstance(n, ast.Import):
                self.names |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name == "*":
                        self.open = True
                    self.names.add(a.asname or a.name)
            else:
                for f in _BLOCKS:
                    self._walk(getattr(n, f, None) or [])
                for h in getattr(n, "handlers", None) or []:
                    self._walk(h.body)


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _modules(sources: Dict[str, bytes]) -> Dict[str, _Mod]:
    out: Dict[str, _Mod] = {}
    for p, src in sources.items():
        if p.endswith(".py"):
            name = p[:-3].replace("/", ".")
            try:
                out[name[: -len(".__init__")] if name.endswith(".__init__") else name] = _Mod(src)
            except (SyntaxError, ValueError):
                pass
    return out


def _problem(target: str, mods: Dict[str, _Mod]) -> str:
    parts = target.split(".")
    if len(parts) < 2 or target in mods:
        return ""
    for i in range(len(parts) - 1, 0, -1):               # the longest prefix that is a module of this repo
        m = ".".join(parts[:i])
        if m not in mods:
            continue
        mod, name = mods[m], parts[i]
        if mod.open:
            return ""
        if name not in mod.names and f"{m}.{name}" not in mods:
            shown = ", ".join(sorted(mod.names)[:10])
            return (f"patch target `{target}` does not exist: module {m} has no `{name}`" + (f" (it defines: {shown})" if shown else "")
                    + ". Patch a name that exists there, or the name the code under test really looks up")
        if i + 1 < len(parts) and name in mod.classes:
            members, based = mod.classes[name]
            if not based and parts[i + 1] not in members:
                return f"patch target `{target}` does not exist: class {name} has no `{parts[i + 1]}`"
        return ""
    if parts[0] in {k.split(".")[0] for k in mods}:
        return f"patch target `{target}` does not exist: there is no module {'.'.join(parts[:-1])} in this repo"
    return ""                                            # stdlib / third party: not ours to judge


def missing_patch_targets(test_src, sources: Dict[str, bytes]) -> List[str]:
    """Patch targets in the test that do not exist in `sources` (repo files with the contract stubs applied)."""
    text = test_src.decode("utf-8", "replace") if isinstance(test_src, bytes) else test_src
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    mods = _modules(sources)
    alias: Dict[str, str] = {}                           # local name -> dotted path it was imported from
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                alias[a.asname or a.name] = f"{n.module}.{a.name}"
    out: List[str] = []
    for c in ast.walk(tree):
        if not isinstance(c, ast.Call) or not c.args:
            continue
        kws = {k.arg: k.value for k in c.keywords if k.arg}
        if any(isinstance(kws.get(k), ast.Constant) and kws[k].value in (True, False) and (k == "create") == bool(kws[k].value)
               for k in ("create", "raising")):
            continue                                     # create=True / raising=False: a missing name is allowed
        fn, first = _dotted(c.func), c.args[0]
        target = ""
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and "." in first.value:
            if fn in ("patch", "mock.patch", "unittest.mock.patch") or fn.endswith((".setattr", ".delattr")):
                target = first.value
        elif isinstance(first, ast.Name) and first.id in alias and len(c.args) > 1:
            second = c.args[1]
            if (fn.endswith("patch.object") or fn.endswith((".setattr", ".delattr"))) \
                    and isinstance(second, ast.Constant) and isinstance(second.value, str):
                target = f"{alias[first.id]}.{second.value}"
        msg = _problem(target, mods) if target else ""
        if msg and msg not in out:
            out.append(msg)
    return out[:5]


def coroutine_misuse(test_src) -> List[str]:
    """`run(go)` where `go` is a function: asyncio.run needs a coroutine, so it must be `run(go())`. Every such test fails the same way."""
    text = test_src.decode("utf-8", "replace") if isinstance(test_src, bytes) else test_src
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    runners = {"asyncio.run"}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "asyncio":
            runners |= {a.asname or a.name for a in n.names if a.name == "run"}
        elif isinstance(n, ast.Assign) and _dotted(n.value) == "asyncio.run":
            runners |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.args.args:        # def run(coro): return asyncio.run(coro)
            first = n.args.args[0].arg
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and _dotted(c.func) == "asyncio.run" \
                        and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == first:
                    runners.add(n.name)
    out: List[str] = []
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and c.args and isinstance(c.args[0], ast.Name) and c.args[0].id in funcs:
            fn = _dotted(c.func)
            if fn in runners:
                msg = f"`{fn}({c.args[0].id})` passes the function itself: it needs a coroutine, so write `{fn}({c.args[0].id}())`"
                if msg not in out:
                    out.append(msg)
    return out[:5]
