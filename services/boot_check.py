"""App boot check. Deterministic, no model.

Import-smoke only proves a changed module imports. A 500 after restart comes from the app: a handler, a template, the
startup hook, a caller of a changed function. So the sandbox also boots the app and hits every GET route without
path params. Pass = no exception and no 5xx. Failures that also happen on the untouched base are subtracted by the
caller (same rule as for tests).
"""
import ast
import re
from typing import Dict, List

_APP_FUNCS = {"create_app", "get_app", "build_app", "make_app"}
_TEST = re.compile(r"(^|/)(tests?|venv|\.venv|node_modules|site-packages)/|(^|/)(test_[^/]*|[^/]*_test|conftest)\.py$")
_ROOT_FIRST = {"main", "app", "run", "server", "api_view"}


def _is_fastapi_call(n: ast.AST) -> bool:
    if not isinstance(n, ast.Call):
        return False
    f = n.func
    return (isinstance(f, ast.Name) and f.id == "FastAPI") or (isinstance(f, ast.Attribute) and f.attr == "FastAPI")


def find_entries(tree: Dict[str, bytes], limit: int = 3) -> List[str]:
    """['views.api_view:app', 'main:create_app()'] for FastAPI apps defined in the repo, most likely first."""
    found = []
    for path, data in tree.items():
        if not path.endswith(".py") or _TEST.search(path) or b"FastAPI" not in data:
            continue
        parts = path[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(re.fullmatch(r"[A-Za-z_]\w*", p) for p in parts):
            continue
        mod = ".".join(parts)
        try:
            body = ast.parse(data).body
        except SyntaxError:
            continue
        for n in body:
            if isinstance(n, ast.Assign) and _is_fastapi_call(n.value):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        found.append((len(parts), parts[-1] not in _ROOT_FIRST, f"{mod}:{t.id}"))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in _APP_FUNCS \
                    and any(_is_fastapi_call(x) for x in ast.walk(n)):
                found.append((len(parts), parts[-1] not in _ROOT_FIRST, f"{mod}:{n.name}()"))
    return [e for _, _, e in sorted(found)][:limit]


# Runs INSIDE the sandbox container from the workspace root. Output lines: BOOT|<step>|OK|ERR|SKIP|<message>
BOOT_PY = r'''import importlib, json, os, sys, threading, traceback

def out(step, state, msg=""):
    print("BOOT|%s|%s|%s" % (step, state, " ".join(str(msg).split())[:400]), flush=True)

def where(e):
    tb = [f for f in traceback.extract_tb(e.__traceback__) if "-packages" not in f.filename and "/usr/lib" not in f.filename]
    if not tb:
        return "%s: %s" % (type(e).__name__, e)
    f = tb[-1]
    try:
        rel = os.path.relpath(f.filename)
    except ValueError:      # Windows: file on another drive than the cwd
        rel = f.filename
    return "%s: %s (%s:%d in %s)" % (type(e).__name__, e, rel, f.lineno, f.name)

def walk(routes, prefix=""):
    # newer FastAPI wraps include_router() lazily (original_router + include_context); older flattens. Handle both.
    for r in routes:
        if hasattr(r, "original_router"):
            yield from walk(r.original_router.routes, prefix + (getattr(r.include_context, "prefix", "") or ""))
        elif getattr(r, "routes", None):
            yield from walk(r.routes, prefix + (getattr(r, "path", "") or ""))
        elif "GET" in (getattr(r, "methods", None) or ()):
            yield prefix + r.path

def get_paths(app):
    found = [p for p in walk(app.routes)]
    try:
        found += [p for p, ops in app.openapi().get("paths", {}).items() if "get" in ops]
    except Exception:
        pass
    return [p for p in found if "{" not in p]

def call(fn, seconds):
    # Unix alarm signals do not exist on Windows (LocalRunner runs this on the host): time-box with a thread instead.
    box = {}

    def run():
        try:
            box["ok"] = fn()
        except BaseException as e:
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError("no answer in %ds" % seconds)
    if "err" in box:
        raise box["err"]
    return box["ok"]

sys.path.insert(0, os.getcwd())
for spec in json.load(open(".sbx_boot.json"))["entries"]:
    mod, _, attr = spec.partition(":")
    try:
        app = getattr(importlib.import_module(mod), attr.rstrip("()"))
        app = app() if attr.endswith("()") else app
        out(spec + "#import", "OK")
    except BaseException as e:
        out(spec + "#import", "ERR", where(e))
        continue
    try:
        app.openapi()
        out(spec + "#openapi", "OK")
    except BaseException as e:
        out(spec + "#openapi", "ERR", where(e))
    try:
        from starlette.testclient import TestClient
    except Exception as e:
        out(spec + "#client", "SKIP", "TestClient unavailable (needs httpx): %s" % e)
        continue
    try:
        client = TestClient(app)
        client.__enter__()
        out(spec + "#startup", "OK")
    except BaseException as e:
        out(spec + "#startup", "ERR", where(e))
        continue
    paths = sorted(set(get_paths(app)))[:40]
    for p in paths:
        step = "%s#GET %s" % (spec, p)
        try:
            r = call(lambda: client.get(p, follow_redirects=True), 8)
            if r.status_code >= 500:
                out(step, "ERR", "HTTP %d %s" % (r.status_code, r.text[:200]))
            else:
                out(step, "OK", r.status_code)
        except TimeoutError as e:
            out(step, "SKIP", str(e))
        except BaseException as e:
            out(step, "ERR", where(e))
sys.stdout.flush()
os._exit(0)
'''
