"""Turn the repo's requirements layout into the ONE text the sandbox image is built from.

The image build gets a single `requirements.txt` and nothing else from the repo, so:
  * every root requirements file that exists is merged (requirements-test.txt used to hide requirements.txt);
  * `-r other.txt` lines are inlined (they used to point at the copied file itself, never at the real list);
  * lines the build context cannot satisfy are dropped: `-c constraints.txt`, `-e .`, local paths.

A single plain file without any of those lines is returned UNCHANGED, so existing image tags (and their
cached builds) stay valid. Pure functions: no I/O, no settings.
"""
from __future__ import annotations

import posixpath
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

MAX_FILES = 20            # includes followed per layout; a guard against runaway / circular trees

_INCLUDE = re.compile(r"^(?:-r\s*|--requirement[\s=]+)(\S+)")
_CONSTRAINT = re.compile(r"^(?:-c\s*|--constraint[\s=]+)(\S+)")
_EDITABLE = re.compile(r"^(?:-e\b|--editable\b)")
_LOCAL = re.compile(r"^(?:\.{1,2}[/\\]|/|file:|[A-Za-z]:[/\\])")
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _lines(text: str) -> List[str]:
    """Logical lines: continuations joined (hash-pinned files), comments and blanks gone."""
    out = []
    for raw in (text or "").replace("\\\r\n", " ").replace("\\\n", " ").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _classify(line: str) -> str:
    if _INCLUDE.match(line):
        return "include"
    if _CONSTRAINT.match(line):
        return "constraint"
    if _EDITABLE.match(line) or _LOCAL.match(line):
        return "local"
    return "req"


def _resolve(current: str, target: str) -> Optional[str]:
    """Repo-relative posix path of `target` as pip resolves it (relative to the including file); None = unusable."""
    if "://" in target or target.startswith(("/", "\\")):
        return None
    path = posixpath.normpath(posixpath.join(posixpath.dirname(current), target.replace("\\", "/")))
    return None if path.startswith("..") or path == "." else path


def include_targets(name: str, text: str) -> List[str]:
    """Repo paths this file pulls in with -r / --requirement (for callers that load files lazily)."""
    out = []
    for line in _lines(text):
        m = _INCLUDE.match(line)
        if m:
            t = _resolve(name, m.group(1))
            if t:
                out.append(t)
    return out


def needs_rewrite(text: str) -> bool:
    return any(_classify(line) != "req" for line in _lines(text))


def _key(line: str) -> str:
    """Identity of a requirement for first-wins de-duplication: the normalised package name.
    Options, URL-only lines and lines with an environment marker (`; python_version < ...`) are kept as they are."""
    if line.startswith("-") or ";" in line or ("://" in line and " @ " not in line):
        return line
    m = _NAME.match(line)
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else line


def _expand(name: str, files: Dict[str, str], stack: Tuple[str, ...], notes: List[str]) -> Iterable[str]:
    if name in stack:
        notes.append(f"circular include of {name} ignored")
        return
    for line in _lines(files.get(name, "")):
        kind = _classify(line)
        if kind == "req":
            yield line
        elif kind == "include":
            target = _resolve(name, _INCLUDE.match(line).group(1))
            if target and target in files:
                yield from _expand(target, files, stack + (name,), notes)
            else:
                notes.append(f"{name}: include of {_INCLUDE.match(line).group(1)} dropped (file not in the repo)")
        elif kind == "constraint":
            notes.append(f"{name}: '{line}' dropped (constraint files are not available to the image build)")
        else:
            notes.append(f"{name}: '{line}' dropped (local path / editable install is not available to the image build)")


def collect(get: Callable[[str], Optional[str]], roots: Sequence[str]) -> Dict[str, str]:
    """Load the root files that exist plus everything they include (bounded). `get(path)` -> text or None."""
    files: Dict[str, str] = {}
    todo = list(roots)
    while todo and len(files) < MAX_FILES:
        name = todo.pop(0)
        if name in files:
            continue
        text = get(name)
        if text is None:
            continue
        files[name] = text
        todo.extend(include_targets(name, text))
    return files


def merge_requirements(files: Dict[str, str], roots: Sequence[str]) -> str:
    """The text to install. `files` = collect(...) result; `roots` = the settings priority order."""
    present = [r for r in roots if r in files]
    if not present:
        return ""
    if len(present) == 1 and not needs_rewrite(files[present[0]]):
        return files[present[0]]                      # untouched: same image tag as before
    notes: List[str] = []
    seen, out = set(), []
    for root in present:
        for line in _expand(root, files, (), notes):
            k = _key(line)
            if k not in seen:
                seen.add(k)
                out.append(line)
    head = [f"# merged by the sandbox from: {', '.join(present)}"] + [f"# {n}" for n in notes]
    return "\n".join(head + out) + "\n"


def requirements_from(get: Callable[[str], Optional[str]], roots: Sequence[str]) -> str:
    return merge_requirements(collect(get, roots), roots)
