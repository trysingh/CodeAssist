"""Fast in-process checks on edited files (verification level L0).

No sandbox needed: parse-level problems are caught before any test run.
Missing optional libraries (pyflakes, yaml, jinja2, tomllib, tree-sitter) just mean that
check is reported as skipped, never as passed.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from pyflakes.checker import Checker as _PyflakesChecker
except Exception:  # pragma: no cover
    _PyflakesChecker = None
try:
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None
try:
    import jinja2 as _jinja2
except Exception:  # pragma: no cover
    _jinja2 = None
try:
    import tomllib as _tomllib
except Exception:  # pragma: no cover
    _tomllib = None
try:
    from jinja2 import nodes as _jnodes
except Exception:  # pragma: no cover
    _jnodes = None
try:
    from tree_sitter_language_pack import get_parser as _ts_parser     # already needed by the chunker
except Exception:  # pragma: no cover
    _ts_parser = None

# pyflakes message classes that mean "this will crash", not style
_FATAL_PYFLAKES = {"UndefinedName", "UndefinedLocal", "UndefinedExport"}
_MAX_PER_FILE = 5


@dataclass
class CheckResult:
    path: str
    check: str
    ok: bool
    message: str = ""
    line: Optional[int] = None
    skipped: bool = False

    def label(self) -> str:
        loc = f":{self.line}" if self.line else ""
        return f"{self.path}{loc} [{self.check}] {self.message}"


def _py(path: str, text: str) -> List[CheckResult]:
    try:
        tree = ast.parse(text, filename=path)
        compile(tree, path, "exec", dont_inherit=True)
    except SyntaxError as e:
        return [CheckResult(path, "syntax", False, f"{e.__class__.__name__}: {e.msg}", e.lineno)]
    except (ValueError, RecursionError) as e:
        return [CheckResult(path, "syntax", False, str(e))]
    out = [CheckResult(path, "syntax", True)]
    if _PyflakesChecker is None:
        out.append(CheckResult(path, "undefined-names", True, "pyflakes not installed", skipped=True))
        return out
    bad = [m for m in _PyflakesChecker(tree, filename=path).messages
           if m.__class__.__name__ in _FATAL_PYFLAKES]
    if not bad:
        out.append(CheckResult(path, "undefined-names", True))
    for m in sorted(bad, key=lambda m: m.lineno)[:_MAX_PER_FILE]:
        out.append(CheckResult(path, "undefined-names", False, m.message % m.message_args, m.lineno))
    return out


def _json(path: str, text: str) -> List[CheckResult]:
    try:
        if path.endswith(".jsonl"):
            for i, ln in enumerate(text.splitlines(), 1):
                if ln.strip():
                    json.loads(ln)
        else:
            json.loads(text)
        return [CheckResult(path, "json", True)]
    except json.JSONDecodeError as e:
        return [CheckResult(path, "json", False, e.msg, e.lineno)]


def _yaml_check(path: str, text: str) -> List[CheckResult]:
    if _yaml is None:
        return [CheckResult(path, "yaml", True, "PyYAML not installed", skipped=True)]
    try:
        list(_yaml.safe_load_all(text))
        return [CheckResult(path, "yaml", True)]
    except _yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        return [CheckResult(path, "yaml", False, str(getattr(e, "problem", e)),
                            (mark.line + 1) if mark else None)]


def _toml(path: str, text: str) -> List[CheckResult]:
    if _tomllib is None:
        return [CheckResult(path, "toml", True, "tomllib needs Python 3.11+", skipped=True)]
    try:
        _tomllib.loads(text)
        return [CheckResult(path, "toml", True)]
    except Exception as e:
        return [CheckResult(path, "toml", False, str(e))]


def _jinja(path: str, text: str) -> List[CheckResult]:
    if "{{" not in text and "{%" not in text:
        return [CheckResult(path, "jinja", True, "no template syntax", skipped=True)]
    if _jinja2 is None:
        return [CheckResult(path, "jinja", True, "jinja2 not installed", skipped=True)]
    try:
        _jinja2.Environment().parse(text)
        return [CheckResult(path, "jinja", True)]
    except _jinja2.TemplateSyntaxError as e:
        return [CheckResult(path, "jinja", False, e.message, e.lineno)]


_JS_MAX_BYTES = 400_000          # bundles / minified vendor files are not worth a parse


def _first_syntax_error(root):
    """The first ERROR / missing node, depth first (None if the tree cannot be walked)."""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or getattr(n, "is_missing", False):
            return n
        if n.has_error:
            stack.extend(reversed(n.children))
    return None


def _js(path: str, text: str) -> List[CheckResult]:
    """Syntax only (nothing is executed): modern JS / JSX included, via the tree-sitter grammar the chunker already uses."""
    if _ts_parser is None:
        return [CheckResult(path, "js-syntax", True, "tree-sitter not installed", skipped=True)]
    if len(text) > _JS_MAX_BYTES:
        return [CheckResult(path, "js-syntax", True, "file too large to check", skipped=True)]
    try:
        root = _ts_parser("javascript").parse(text.encode("utf-8")).root_node
        if not root.has_error:
            return [CheckResult(path, "js-syntax", True)]
        node = _first_syntax_error(root)
        row = node.start_point[0] if node is not None else 0
        lines = text.splitlines()
        near = (lines[row].strip() if row < len(lines) else "")[:80]
        what = "missing token" if node is not None and getattr(node, "is_missing", False) else "unexpected syntax"
        return [CheckResult(path, "js-syntax", False, f"{what} near: {near}", row + 1)]
    except Exception as e:      # grammar not available offline, binding differences ...
        return [CheckResult(path, "js-syntax", True, f"parser unavailable ({type(e).__name__})", skipped=True)]


def _template_exists(name: str, from_path: str, known: set) -> bool:
    name = name.lstrip("/").replace("\\", "/")
    here = from_path.rsplit("/", 1)[0] + "/" if "/" in from_path else ""
    if name in known or ("templates/" + name) in known or (here + name) in known:
        return True
    return any(k.endswith("/" + name) for k in known)


def jinja_references(path: str, text: str, known_paths) -> List[CheckResult]:
    """{% extends / include / import / from %} targets that are not in the repo: TemplateNotFound at the first request.
    Dynamic names ({% extends layout %}) and `include ... ignore missing` are not judged."""
    if _jinja2 is None or _jnodes is None or "{%" not in text:
        return []
    try:
        ast_ = _jinja2.Environment().parse(text)
        wanted = []
        for node in ast_.find_all((_jnodes.Extends, _jnodes.Include, _jnodes.Import, _jnodes.FromImport)):
            if isinstance(node, _jnodes.Include) and getattr(node, "ignore_missing", False):
                continue
            if isinstance(node.template, _jnodes.Const) and isinstance(node.template.value, str):
                wanted.append(node.template.value)
    except Exception:
        return []
    known = set(known_paths or ())
    names = list(dict.fromkeys(wanted))
    out = [CheckResult(path, "jinja-ref", False, f"template '{n}' (extends/include/import) is not in the repo")
           for n in names if not _template_exists(n, path, known)]
    return out or ([CheckResult(path, "jinja-ref", True)] if names else [])


_BY_SUFFIX = {
    ".py": _py, ".json": _json, ".jsonl": _json,
    ".yaml": _yaml_check, ".yml": _yaml_check, ".toml": _toml,
    ".html": _jinja, ".htm": _jinja, ".jinja": _jinja, ".j2": _jinja,
    ".js": _js, ".mjs": _js, ".cjs": _js, ".jsx": _js,
}
_TEMPLATE_SUFFIXES = (".html", ".htm", ".jinja", ".j2")


def check_files(files: Dict[str, bytes], known_paths=None) -> List[CheckResult]:
    """`known_paths` = every file path of the repo view (+ the edited files): enables the template-reference check."""
    results: List[CheckResult] = []
    for path in sorted(files):
        suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
        fn = _BY_SUFFIX.get(suffix)
        if fn is None:
            results.append(CheckResult(path, "none", True, f"no checker for '{suffix or 'this file type'}'", skipped=True))
            continue
        try:
            text = files[path].decode("utf-8")
        except UnicodeDecodeError:
            results.append(CheckResult(path, "encoding", False, "not valid UTF-8"))
            continue
        results.extend(fn(path, text))
        if known_paths is not None and suffix in _TEMPLATE_SUFFIXES:
            results.extend(jinja_references(path, text, known_paths))
    return results


def new_failures(base: List[CheckResult], now: List[CheckResult]) -> List[CheckResult]:
    """Failures in `now` that the untouched base file did not already have (line numbers ignored)."""
    known = {(r.path, r.check, r.message) for r in base if not r.ok}
    return [r for r in now if not r.ok and (r.path, r.check, r.message) not in known]


def summarize(results: List[CheckResult]) -> Tuple[int, int, int]:
    """(checked, failed, skipped) counted per (file, check)."""
    checked = sum(1 for r in results if not r.skipped)
    failed = sum(1 for r in results if not r.ok)
    skipped = sum(1 for r in results if r.skipped)
    return checked, failed, skipped
