"""Cross-file call check for /change.

Flags calls to methods that no repo class defines, e.g. `self.db.delete_repo_chunks()` when
DatabaseManager has no such method. A patch that never saw the owning file invents these, and
syntax/lint checks cannot see them. Deliberately conservative: it only judges receivers it can
resolve to a repo class, and skips classes with bases or __getattr__ (methods may come from elsewhere).

Arity: a call that resolves to a repo method, function or constructor is also checked against its
signature (too many / missing arguments, unknown keyword). Skipped when the call uses *args / **kwargs
or the callee has decorators other than staticmethod / classmethod.
"""
import ast
from typing import Dict, List, Set, Tuple

# classes, per-class attribute -> class names, per-file top-level functions, dotted module -> path
Index = Tuple[Dict[str, dict], Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, tuple]], Dict[str, str]]


def _deco_kind(n) -> str:
    if not n.decorator_list:
        return "plain"
    if len(n.decorator_list) == 1 and isinstance(n.decorator_list[0], ast.Name):
        return {"staticmethod": "static", "classmethod": "class"}.get(n.decorator_list[0].id, "other")
    return "other"


def _arity(a: ast.arguments, call: ast.Call, skip_first: bool) -> str:
    """'' when the call fits the signature (or cannot be judged), else why it does not."""
    if any(isinstance(x, ast.Starred) for x in call.args) or any(k.arg is None for k in call.keywords):
        return ""
    full = [x.arg for x in a.posonlyargs + a.args]
    required = max(0, len(full) - len(a.defaults))
    if skip_first and full:
        full, required = full[1:], max(0, required - 1)
    n_pos, kws = len(call.args), [k.arg for k in call.keywords]
    if n_pos > len(full) and not a.vararg:
        return f"takes at most {len(full)} positional argument(s) but {n_pos} given"
    by_kw = [x.arg for x in a.args]                         # positional-or-keyword names (posonly ones cannot be passed by name)
    if skip_first and not a.posonlyargs and by_kw:
        by_kw = by_kw[1:]
    named = set(by_kw) | {k.arg for k in a.kwonlyargs}
    for k in kws:
        if k not in named and not a.kwarg:
            return f"unexpected keyword argument '{k}'"
    missing = [x for i, x in enumerate(full[:required]) if i >= n_pos and x not in kws]
    missing += [k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is None and k.arg not in kws]
    return f"missing required argument(s): {', '.join(missing)}" if missing else ""


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
            dup = c.name in classes
            info = classes.setdefault(c.name, {"names": set(), "open": False, "path": path, "sigs": {}, "decorated": False, "dup": False})
            info["dup"] = info["dup"] or dup                # same class name in two files: do not judge arguments
            info["decorated"] = info["decorated"] or bool(c.decorator_list)
            if c.bases or c.keywords:
                info["open"] = True
            for n in c.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info["sigs"][n.name] = (n.args, "other" if n.name in info["names"] else _deco_kind(n))
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
    funcs = {p: {n.name: (n.args, _deco_kind(n)) for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
             for p, t in trees.items()}
    mods = {(p[:-3][: -len("/__init__")] if p.endswith("/__init__.py") else p[:-3]).replace("/", "."): p for p in trees}
    return classes, attrs, funcs, mods


def unresolved(src: bytes, index: Index) -> Dict[str, Tuple[int, str, str]]:
    """{receiver.method: (line, message, owner_path)} for calls that resolve to a repo class lacking the method."""
    classes, attrs, funcs, mods = index
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

    def add(key: str, call: ast.Call, msg: str, owner: str) -> None:
        shape = f"{len(call.args)}:{','.join(sorted(str(k.arg) for k in call.keywords))}"
        out.setdefault(f"{key}#arity:{shape}", (call.lineno, msg, owner))

    def judge(recv: str, method: str, cls_names: Set[str], line: int, call: ast.Call = None) -> None:
        known = [classes[c] for c in cls_names if c in classes]
        if not known or any(k["open"] for k in known):
            return
        if all(method not in k["names"] for k in known):
            c0 = sorted(cls_names)[0]
            out.setdefault(f"{recv}.{method}", (
                line, f"`{recv}.{method}()` - class {c0} defines no '{method}'. "
                      f"Add it to {classes[c0]['path']} (edit that file too) or call a method that exists.",
                classes[c0]["path"]))
        elif len(known) == 1 and not known[0]["dup"] and method in known[0]["sigs"] and call is not None:
            a, kind = known[0]["sigs"][method]
            err = _arity(a, call, skip_first=kind != "static") if kind != "other" else ""
            if err:
                c0 = sorted(cls_names)[0]
                add(f"{recv}.{method}", call, f"`{recv}.{method}()` does not fit {c0}.{method}({ast.unparse(a)}) in "
                    f"{known[0]['path']}: {err}. Fix the call or the signature.", known[0]["path"])

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
        m, v = call.func.attr, call.func.value
        if isinstance(v, ast.Name) and v.id == "self" and id(call) in owner:
            judge("self", m, {owner[id(call)]}, call.lineno, call)
        elif isinstance(v, ast.Name) and v.id in roots:
            judge(v.id, m, {roots[v.id]}, call.lineno, call)
        elif isinstance(v, ast.Attribute):
            base, x = v.value, v.attr
            if isinstance(base, ast.Name) and base.id == "self" and id(call) in owner:
                held = attrs.get(owner[id(call)], {}).get(x)
            elif isinstance(base, ast.Name) and base.id in roots:
                held = attrs.get(roots[base.id], {}).get(x)
            else:
                held = None
            if held:
                judge(f"{base.id}.{x}", m, held, call.lineno, call)

    # plain calls: f(...) and Cls(...) that resolve to a repo function / class
    local = {n.name: (n.args, _deco_kind(n), "") for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    local_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    imported: Dict[str, tuple] = {}
    imported_cls: Dict[str, str] = {}
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module in mods:
            fp = mods[n.module]
            for a in n.names:
                if a.name in funcs.get(fp, {}):
                    imported[a.asname or a.name] = (*funcs[fp][a.name], fp)
                elif a.name in classes and classes[a.name]["path"] == fp:
                    imported_cls[a.asname or a.name] = a.name
    shadow = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | \
             {a.arg for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for a in
              n.args.posonlyargs + n.args.args + n.args.kwonlyargs}
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)):
        nm = call.func.id
        if nm in shadow:
            continue
        if nm in local or nm in imported:
            a, kind, where = local[nm] if nm in local else imported[nm]
            err = _arity(a, call, skip_first=False) if kind == "plain" else ""
            if err:
                add(nm, call, f"`{nm}()` does not fit {nm}({ast.unparse(a)})" + (f" in {where}" if where else "")
                    + f": {err}. Fix the call or the signature.", where)
            continue
        cname = nm if nm in local_classes else imported_cls.get(nm, "")
        info = classes.get(cname)
        if not info or info["open"] or info["decorated"] or info["dup"] or "__new__" in info["names"]:
            continue
        init = info["sigs"].get("__init__")
        if init is not None:
            err = _arity(init[0], call, skip_first=True) if init[1] == "plain" else ""
        else:
            err = "the class takes no arguments" if (call.args or call.keywords) and not any(isinstance(x, ast.Starred) for x in call.args) \
                  and not any(k.arg is None for k in call.keywords) else ""
        if err:
            add(nm, call, f"`{nm}(...)` does not fit {cname}.__init__ in {info['path']}: {err}. Fix the call or the signature.", info["path"])
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
                for f in ((owner, p) if "#arity" in key else (owner,)):     # an argument mismatch needs the caller AND the callee
                    if f and f not in owners:
                        owners.append(f)
    return probs, owners
