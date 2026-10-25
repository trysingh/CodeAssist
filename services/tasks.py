"""Task plan for a change: one task per contract symbol (dependency order), then one per caller the new signatures break.
Pure functions: no model calls, no I/O."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from services import splice
from services.contract import CSym, Shared

_RIPPLE = re.compile(r"^(?P<path>[^:\s]+):(?P<line>\d+): (?P<msg>.*)$", re.S)


@dataclass
class Task:
    id: str
    kind: str                   # implement | fix_caller
    file: str
    name: str                   # qualified: "func" | "Cls.method"
    sig: str                    # the header the result must keep
    behavior: str               # what it must do / what to fix
    deps: List[str] = field(default_factory=list)       # "path::name" it may call
    change: str = "new"         # new | signature | behavior | caller
    creates: List[str] = field(default_factory=list)    # shared-state attributes this task must assign


def dep_key(d: str) -> Tuple[str, str]:
    """'path::name' -> (path, name); ('', '') for anything else (external dependency)."""
    f, sep, q = d.partition("::")
    return (f.strip().replace("\\", "/"), q.strip()) if sep else ("", "")


def _dep_keys(t: Task) -> List[Tuple[str, str]]:
    return [k for k in map(dep_key, t.deps) if k != ("", "")]


def plan_tasks(syms: Sequence[CSym], ripple: Sequence[str], sources: Dict[str, bytes],
               shared: Sequence[Shared] = ()) -> Tuple[List[Task], List[str]]:
    """-> (ordered tasks, leftovers no task can cover; the single-patch stage handles those)."""
    impl: List[Task] = []
    leftovers: List[str] = []
    for s in syms:
        if s.kind == "class":
            has_methods = any(m.kind == "method" and m.file == s.file and m.name.rsplit(".", 1)[0] == s.name for m in syms)
            if s.change == "new" and not has_methods:
                leftovers.append(f"{s.file}::{s.name}: new class with no method in the contract (its attributes need hand-written code)")
            continue
        made = [x.attr.partition(".")[2] for x in shared if (x.file, x.created_in) == (s.file, s.name)]
        impl.append(Task("", "implement", s.file, s.name, s.sig, s.behavior, list(s.deps), s.change, made))
    keys = {(t.file, t.name): t for t in impl}

    # dependency order (callee before caller); a cycle keeps the designer's order
    ordered: List[Task] = []
    left = list(impl)
    while left:
        pending = {(x.file, x.name) for x in left}
        ready = [t for t in left if not any(k in pending and k != (t.file, t.name) for k in _dep_keys(t))]
        pick = (ready or left)[0]
        ordered.append(pick)
        left.remove(pick)

    callers: Dict[Tuple[str, str], Task] = {}
    for line in ripple:
        m = _RIPPLE.match(line)
        hit = splice.symbol_at(sources.get(m["path"], b""), int(m["line"])) if m else None
        if hit is None:
            leftovers.append(line)
            continue
        key = (m["path"], hit.qualname)
        if key in keys:                         # that function is being rewritten by its own task anyway
            continue
        if key in callers:
            if m["msg"] not in callers[key].behavior:
                callers[key].behavior = (callers[key].behavior + "; " + m["msg"])[:900]
            continue
        callers[key] = Task("", "fix_caller", m["path"], hit.qualname, hit.sig, m["msg"][:900], [], "caller")
    ordered += list(callers.values())
    for i, t in enumerate(ordered, 1):
        t.id = f"t{i}"
    return ordered, leftovers
