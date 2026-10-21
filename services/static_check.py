"""Fast in-process checks on edited files (verification level L0).

No sandbox needed: parse-level problems are caught before any test run.
Missing optional libraries (pyflakes, yaml, jinja2, tomllib) just mean that
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


_BY_SUFFIX = {
    ".py": _py, ".json": _json, ".jsonl": _json,
    ".yaml": _yaml_check, ".yml": _yaml_check, ".toml": _toml,
    ".html": _jinja, ".htm": _jinja, ".jinja": _jinja, ".j2": _jinja,
}


def check_files(files: Dict[str, bytes]) -> List[CheckResult]:
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
