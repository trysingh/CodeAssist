"""Cheap deterministic gates for /change tasks. Pure functions: no model calls, no I/O."""
import ast
from typing import Dict, List


def unresolved_local_imports(src: bytes, known: set) -> List[str]:
    """Imports of repo-local modules that do not exist (not in the repo, not in the plan)."""
    mods = set()
    for p in known:
        if p.endswith(".py"):
            m = p[:-3].replace("/", ".")
            mods.add(m[:-9] if m.endswith(".__init__") else m)
    tops = {m.split(".")[0] for m in mods}
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names = [n.module]
        elif isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        else:
            continue
        for m in names:
            if m.split(".")[0] in tops and m not in mods:
                out.append(f"`import {m}`: no such module in this repo or plan. Do not invent modules; put the code in an existing file")
    return out


def _self_attrs(src: bytes, cls: str):
    for c in ast.walk(ast.parse(src)):
        if isinstance(c, ast.ClassDef) and c.name == cls:
            st, ld = set(), set()
            for n in ast.walk(c):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                    (st if isinstance(n.ctx, ast.Store) else ld).add(n.attr)
            return st, ld
    return set(), set()


def lost_attributes(old: bytes, new: bytes, qualname: str) -> List[str]:
    """self.X was assigned before, is still read, and nothing assigns it any more."""
    if "." not in qualname:
        return []
    cls = qualname.rsplit(".", 2)[-2]
    st_o, _ = _self_attrs(old, cls)
    st_n, ld_n = _self_attrs(new, cls)
    return [f"self.{a} used to be set and is still read, but nothing sets it now: keep it" for a in sorted(st_o - st_n) if a in ld_n]
