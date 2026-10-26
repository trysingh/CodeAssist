"""Where do the ~20 s per sandbox pytest subprocess go?

Replays what LocalRunner does (same python, same stripped env from local_env(), same argv from pytest_argv()) on a
two-file project, and varies ONE thing at a time so the slow part is attributable.

Run from the repo root, inside the same conda env you run the suite in:

    python sandbox_pytest_timing.py            # ~1-3 min on a slow host
    python sandbox_pytest_timing.py --keep     # keep the temp dir

Read the output top to bottom: the first row that jumps is the culprit.
"""
import argparse
import importlib.metadata as md
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from services.sandbox_service import local_env, pytest_argv  # noqa: E402  (the real functions the sandbox uses)

CAP_S = 180
FILES = {
    "calc.py": "def add(a, b):\n    return a + b\n",
    "test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
}


def run(label, args, env, cwd, cap=CAP_S, show_err=False):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(args, capture_output=True, env=env, cwd=cwd, timeout=cap)
        rc, out = p.returncode, (p.stdout or b"") + (p.stderr or b"")
    except subprocess.TimeoutExpired as e:
        rc, out = "TIMEOUT", (e.stdout or b"") + (e.stderr or b"")
    dt = time.perf_counter() - t0
    print(f"{dt:7.2f}s  rc={rc!s:<7} {label}")
    if show_err and rc not in (0, "TIMEOUT"):
        print("          " + out.decode("utf-8", "replace").strip()[-300:].replace("\n", "\n          "))
    return dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    keep = ap.parse_args().keep

    tmp = Path(tempfile.mkdtemp(prefix="coderag_local_diag_")).resolve()
    for name, body in FILES.items():
        (tmp / name).write_text(body, encoding="utf-8")
    report = str(tmp / ".sbx_report.xml")
    argv = [sys.executable if a == "python" else a.replace("{REPORT}", report) for a in pytest_argv(["test_calc.py"])]
    sandbox_env = local_env(tmp)                      # exactly what LocalRunner passes
    host_env = dict(os.environ, PYTHONPATH=str(tmp))  # your normal shell environment

    print(f"python   : {sys.executable} ({sys.version.split()[0]})")
    print(f"temp dir : {tmp}")
    print(f"cpu      : {os.cpu_count()} logical cores\n")

    print("== 1. interpreter and import baselines (sandbox env) ==")
    run("python -c pass", [sys.executable, "-c", "pass"], sandbox_env, tmp)
    run("python -S -c pass   (no site-packages / .pth files)", [sys.executable, "-S", "-c", "pass"], sandbox_env, tmp)
    run("python -c 'import pytest'", [sys.executable, "-c", "import pytest"], sandbox_env, tmp)
    run("python -m pytest --version", [sys.executable, "-m", "pytest", "--version"], sandbox_env, tmp)

    print("\n== 2. the real sandbox command, one variable at a time ==")
    base, base_out = run("EXACT sandbox command (stripped env, plugin autoload ON)", argv, sandbox_env, tmp, show_err=True)
    run("  ... plugin autoload OFF (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1)", argv,
        {**sandbox_env, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, tmp)
    run("  ... run it a 2nd time (warm file cache / AV cache)", argv, sandbox_env, tmp)
    run("  ... your normal env instead of the stripped one", argv, host_env, tmp)
    run("  ... bytecode writing allowed (drop PYTHONDONTWRITEBYTECODE)", argv,
        {k: v for k, v in sandbox_env.items() if k != "PYTHONDONTWRITEBYTECODE"}, tmp)
    run("  ... no network (HTTP(S)_PROXY to a dead port, short connect)", argv,
        {**sandbox_env, "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}, tmp)

    print("\n== 3. pytest plugins that autoload in this env (pytest11 entry points) ==")
    plugins = sorted({(ep.name, ep.value) for d in md.distributions() for ep in d.entry_points if ep.group == "pytest11"})
    for name, value in plugins:
        print(f"   {name:<28} {value}")
    if not plugins:
        print("   (none)")

    print("\n== 4. biggest imports in the exact sandbox command (python -X importtime, cumulative) ==")
    imp_argv = [argv[0], "-X", "importtime", *argv[1:]]
    _, out = run("importtime run", imp_argv, sandbox_env, tmp, show_err=False)
    rows = []
    for ln in out.decode("utf-8", "replace").splitlines():
        m = re.match(r"import time:\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\s*)(\S.*)$", ln)
        if m:
            rows.append((int(m.group(2)), m.group(4).strip()))
    rows.sort(reverse=True)
    for cum, name in rows[:15]:
        print(f"   {cum / 1e6:7.2f}s  {name}")

    print("\n== how to read it ==")
    print("  * python -c pass already slow (>1 s)      -> AV/EDR scanning process start or .pth files; exclude python.exe + env dir")
    print("  * only the 'exact' row slow, 2nd run fast -> cold-cache/AV scan of new files; exclude %TEMP%\\coderag_local_*")
    print("  * plugin autoload OFF much faster         -> a pytest11 plugin from section 3 is the cost (see section 4)")
    print("  * stripped env slow, normal env fast      -> something keys on HOME/USERPROFILE/LOCALAPPDATA (all = temp dir here)")
    print("  * dead-proxy row fast, others slow        -> a network call with a long timeout at startup")
    print("  * importtime shows one package at ~10+ s  -> that package's import is the cost")
    if keep:
        print(f"\nkept: {tmp}")
    else:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
