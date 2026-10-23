"""Cross-file call check for /change.

Flags calls to methods that no repo class defines, e.g. `self.db.delete_repo_chunks()` when
DatabaseManager has no such method. A patch that never saw the owning file invents these, and
syntax/lint checks cannot see them. Deliberately conservative: it only judges receivers it can
resolve to a repo class, and skips classes with bases or __getattr__ (methods may come from elsewhere).
"""
import ast
from typing import Dict, List, Set, Tuple

Index = Tuple[Dict[str, dict], Dict[str, Dict[str, Set[str]]]]   # classes, per-class attribute -> class names


def _parse(data: bytes):
    try:
        return ast.parse(data.decode("utf-8", "replace"))
    except (SyntaxError, ValueError):
        return None


def _ann_name(a) -> str:
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value.strip("'\" ").split(".")[-1]
    if isinstance(a, ast.Name):
        return a.id
    if isinstance(a, ast.Attribute):
        return a.attr
    return ""


def _callee(value) -> str:
    """Class name in `Cls(...)` / `mod.Cls(...)`, else ''."""
    if isinstance(value, ast.Call):
        f = value.func
        return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
    return ""


def build_index(sources: Dict[str, bytes]) -> Index:
    trees = {}
    for p, d in sources.items():
        if p.endswith(".py"):
            t = _parse(d)
            if t is not None:
                trees[p] = t
    classes: Dict[str, dict] = {}
    for path, tree in trees.items():
        for c in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            info = classes.setdefault(c.name, {"names": set(), "open": False, "path": path})
            if c.bases or c.keywords:
                info["open"] = True
            for n in c.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info["names"].add(n.name)
                elif isinstance(n, ast.Assign):
                    info["names"] |= {t.id for t in n.targets if isinstance(t, ast.Name)}
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    info["names"].add(n.target.id)
            for n in ast.walk(c):        # attributes set anywhere on self count as names
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) \
                        and isinstance(n.value, ast.Name) and n.value.id == "self":
                    info["names"].add(n.attr)
            if "__getattr__" in info["names"]:
                info["open"] = True

    attrs: Dict[str, Dict[str, Set[str]]] = {}
    for tree in trees.values():
        for c in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            held = attrs.setdefault(c.name, {})
            for init in (n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == "__init__"):
                params = {a.arg: _ann_name(a.annotation) for a in init.args.args if a.annotation is not None}
                for asg in (n for n in ast.walk(init) if isinstance(n, ast.Assign)):
                    pairs = []
                    for tgt in asg.targets:
                        if isinstance(tgt, ast.Tuple) and isinstance(asg.value, ast.Tuple) \
                                and len(tgt.elts) == len(asg.value.elts):
                            pairs += list(zip(tgt.elts, asg.value.elts))
                        else:
                            pairs.append((tgt, asg.value))
                    for tgt, val in pairs:
                        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"):
                            continue
                        cls = _callee(val) or (params.get(val.id, "") if isinstance(val, ast.Name) else "")
                        if cls in classes:
                            held.setdefault(tgt.attr, set()).add(cls)
    return classes, attrs


def unresolved(src: bytes, index: Index) -> Dict[str, Tuple[int, str, str]]:
    """{receiver.method: (line, message, owner_path)} for calls that resolve to a repo class lacking the method."""
    classes, attrs = index
    tree = _parse(src)
    out: Dict[str, Tuple[int, str, str]] = {}
    if tree is None:
        return out
    owner: Dict[int, str] = {}
    for c in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for n in ast.walk(c):
            owner[id(n)] = c.name
    roots = {t.id: _callee(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and _callee(n.value) in classes}

    def judge(recv: str, method: str, cls_names: Set[str], line: int) -> None:
        known = [classes[c] for c in cls_names if c in classes]
        if not known or any(k["open"] for k in known):
            return
        if all(method not in k["names"] for k in known):
            c0 = sorted(cls_names)[0]
            out.setdefault(f"{recv}.{method}", (
                line, f"`{recv}.{method}()` - class {c0} defines no '{method}'. "
                      f"Add it to {classes[c0]['path']} (edit that file too) or call a method that exists.",
                classes[c0]["path"]))

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
        m, v = call.func.attr, call.func.value
        if isinstance(v, ast.Name) and v.id == "self" and id(call) in owner:
            judge("self", m, {owner[id(call)]}, call.lineno)
        elif isinstance(v, ast.Name) and v.id in roots:
            judge(v.id, m, {roots[v.id]}, call.lineno)
        elif isinstance(v, ast.Attribute):
            base, x = v.value, v.attr
            if isinstance(base, ast.Name) and base.id == "self" and id(call) in owner:
                held = attrs.get(owner[id(call)], {}).get(x)
            elif isinstance(base, ast.Name) and base.id in roots:
                held = attrs.get(roots[base.id], {}).get(x)
            else:
                held = None
            if held:
                judge(f"{base.id}.{x}", m, held, call.lineno)
    return out


def new_problems(base: Dict[str, bytes], after: Dict[str, bytes]) -> Tuple[List[str], List[str]]:
    """Problems the change introduced (absent before) and the files that should own the missing methods.
    Checks every file: removing a method can break callers the patch never touched."""
    before_idx, after_idx = build_index(base), build_index(after)
    probs: List[str] = []
    owners: List[str] = []
    for p, src in after.items():
        if not p.endswith(".py"):
            continue
        old = unresolved(base.get(p, b""), before_idx)
        for key, (line, msg, owner) in unresolved(src, after_idx).items():
            if key not in old:
                probs.append(f"{p}:{line}: {msg}")
                if owner not in owners:
                    owners.append(owner)
    return probs, owners
