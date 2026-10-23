"""Reproduce a sandbox run outside the chat and print EVERYTHING the isolated runtime says.

    python sandbox_diag.py                         # this repo, all test files, backend from settings (docker)
    python sandbox_diag.py C:\\path\\to\\repo         # another repo
    python sandbox_diag.py . --tests tests/test_phase2.py tests/test_plan.py
    python sandbox_diag.py . --probe               # what the container looks like: user, writable dirs, network, packages
    python sandbox_diag.py . --backend local       # same flow without Docker (NO isolation), to compare host vs container

It uses the engine's own code (same image recipe, same `docker run` flags, same pytest arguments), so a failure seen
here is the failure the engine sees. Nothing is written to the repo.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from config.settings import settings
from services import sandbox_service as sbx

_SKIP_DIRS = {".git", ".engine_storage", "__pycache__", "node_modules", ".venv", "venv", "env", "dist", "build",
              ".idea", ".vscode", ".pytest_cache", ".mypy_cache", "data", "workspaces"}
_SKIP_FILES = re.compile(r"(^|/)(\.env[\w.]*|.*\.(pyc|sqlite3?|db|png|jpg|jpeg|gif|zip|exe|dll|so|pdf))$", re.IGNORECASE)

PROBE = r'''
import os, sys, socket, importlib.util, tempfile
print("python      :", sys.version.split()[0], "| user:", os.getuid(), "| cwd:", os.getcwd(), "| HOME:", os.environ.get("HOME"))
for d in ("/work", "/tmp", "/opt", "/usr/local/lib"):
    try:
        f = os.path.join(d, ".w"); open(f, "w").close(); os.remove(f); print("writable    :", d, "yes")
    except Exception as e:
        print("writable    :", d, "no (%s)" % type(e).__name__)
try:
    socket.setdefaulttimeout(3); socket.gethostbyname("pypi.org"); print("network     : reachable (fine for --backend local; inside Docker this must say none)")
except Exception as e:
    print("network     : none (%s)" % type(e).__name__)
print("TIKTOKEN_CACHE_DIR:", os.environ.get("TIKTOKEN_CACHE_DIR"))
for mod in ("pytest", "jinja2", "yaml", "pyflakes", "tiktoken", "langchain", "langgraph", "chromadb", "fastapi", "httpx", "playwright"):
    print("package     : %-10s %s" % (mod, "ok" if importlib.util.find_spec(mod) else "MISSING"))
try:
    import tiktoken
    tiktoken.get_encoding("cl100k_base").encode("hello"); print("tiktoken    : cl100k_base loads offline")
except Exception as e:
    print("tiktoken    : cl100k_base FAILS offline ->", type(e).__name__, str(e)[:160])
'''


def collect_tree(root: Path) -> dict:
    tree = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in rel.split("/")[:-1]) or _SKIP_FILES.search(rel):
            continue
        try:
            if p.stat().st_size <= 2_000_000:
                tree[rel] = p.read_bytes()
        except OSError:
            pass
    return tree


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", default=".", help="repo directory (default: current)")
    ap.add_argument("--tests", nargs="*", help="test files to run (default: every test file)")
    ap.add_argument("--backend", choices=["docker", "local"], help="override settings.sandbox_backend")
    ap.add_argument("--probe", action="store_true", help="describe the container instead of running tests")
    ap.add_argument("--rebuild", action="store_true", help="remove this repo's sandbox image first, then build it again")
    a = ap.parse_args()
    if a.backend:
        settings.sandbox_backend = a.backend

    root = Path(a.repo).resolve()
    tree = collect_tree(root)
    print(f"repo        : {root} ({len(tree)} files)")
    requirements = next((tree[n].decode("utf-8", "replace") for n in settings.sandbox_requirements_files if n in tree), "")
    runner = sbx.make_runner()
    ok, info = await runner.available()
    print(f"backend     : {runner.name} -> {'ready' if ok else 'NOT AVAILABLE'}: {info}")
    if not ok:
        return 2

    if runner.name == "docker":
        tag = runner.image_tag(requirements)
        print(f"image       : {tag}")
        print("--- Dockerfile ---\n" + runner.dockerfile(requirements) + "------------------")
        if a.rebuild:
            await runner.remove_image(tag)
        err = await runner.prepare(requirements)
        if err:
            print("IMAGE BUILD FAILED:\n" + err)
            return 3
        print("image       : ready (build output of the last build is under <storage_dir>/sandbox_logs/_image_builds)")

    timeout = settings.sandbox_test_timeout_s
    if a.probe:
        raw = await runner.run(requirements, sbx.build_tar({"probe.py": PROBE.encode()}), ["python", "probe.py"], timeout)
        print("--- container probe ---\n" + (raw.error or raw.stdout.split("::SBX-EXIT::")[0]))
        return 0

    tests = a.tests or sorted(p for p in tree if sbx.is_test_file(p))
    if not tests:
        print("no test files found")
        return 4
    print(f"running     : {len(tests)} test file(s)")
    tar = sbx.build_tar(tree)
    raw = await runner.run(requirements, tar, sbx.pytest_argv(tests), timeout)
    print("--- raw output of the isolated run ---")
    print(raw.error or raw.stdout.split("::SBX-JUNIT-BEGIN::")[0].rstrip())
    rep = sbx.parse_run(raw, "pytest", runner.name)
    print("--- summary ---")
    if not rep.ran:
        print(f"NOT RUN: {rep.note}")
        return 5
    print(f"exit code {rep.exit_code} | {rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped")
    tops = {p.split("/")[0].removesuffix(".py") for p in tree}
    for c in rep.failures:
        print(f"\n[{c.kind}] {c.id}\n  {c.message}")
        hint = sbx.hint_for(f"{c.message}\n{c.detail}", tops)
        if hint:
            print(f"  likely cause: {hint}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
