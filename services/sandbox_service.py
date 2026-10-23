"""Isolated verification of a proposed change (Phase 2).

Flow for one proposal
  1. build the workspace IN MEMORY: repo view at the base revision + the edited files
  2. pick tests: those that import a changed module (else all tests; else import-smoke)
  3. run them in a throw-away container, stream the workspace in as a tar on stdin
  4. if something fails, run the same selection on the UNTOUCHED base once (cached)
     so failures that already existed are not blamed on the change

Docker design (why it looks like this)
  * `docker run -i --rm`, workspace arrives on stdin as a tar and is unpacked into a
    tmpfs: no bind mounts, so no Windows path/permission/line-ending surprises and no
    host directory is ever exposed to generated code
  * --network none, --read-only root, --cap-drop ALL, no-new-privileges, non-root user,
    memory / cpu / pid limits, hard timeout (in-container AND host side)
  * dependencies are baked into an image tagged by the hash of requirements, built once
    (the only step that needs network)

Pre-warm: that build no longer has to wait for the first change request.
SandboxService.prewarm() builds the image for a repo ahead of time (services/sandbox_warmer.py
calls it in the background at startup, after ingest and when a requirements file changes),
verify() then finds the tag in `_ready` and goes straight to `docker run`. A build that is
still running when a request arrives is awaited, never started twice (per-tag lock).
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from services.notifier import notifier

from config.settings import settings

_TEST_FILE = re.compile(r"(^|/)(test_[^/]+|[^/]+_test)\.py$")
_SMOKE_SKIP = re.compile(r"(^|/)(tests?|migrations|docs?|examples?|scripts)/|(^|/)(setup|conftest)\.py$")
_REPORT = "{REPORT}"
_MAX_TAIL = 3000


# --------------------------------------------------------------------------- results
@dataclass
class CaseResult:
    id: str
    kind: str                 # failed | error | timeout | killed
    message: str = ""
    detail: str = ""


@dataclass
class RunReport:
    mode: str                 # pytest | smoke | none
    ran: bool                 # infrastructure worked and something was executed
    note: str = ""
    backend: str = ""
    selected: List[str] = field(default_factory=list)
    strategy: str = ""        # affected | all | smoke
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: List[CaseResult] = field(default_factory=list)
    timed_out: bool = False
    duration: float = 0.0
    tail: str = ""


@dataclass
class SandboxVerdict:
    report: RunReport
    new_failures: List[CaseResult] = field(default_factory=list)
    preexisting: int = 0

    @property
    def ran(self) -> bool:
        return self.report.ran

    @property
    def ok(self) -> bool:
        return self.report.ran and not self.new_failures

    def to_dict(self) -> dict:
        r = self.report
        return {"mode": r.mode, "ran": r.ran, "ok": self.ok, "note": r.note, "backend": r.backend,
                "strategy": r.strategy, "selected": len(r.selected), "passed": r.passed, "failed": r.failed,
                "skipped": r.skipped, "new_failures": [c.id for c in self.new_failures],
                "preexisting": self.preexisting, "duration": round(r.duration, 1)}


# --------------------------------------------------------------------------- test selection
def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE.search(path))


def _module_names(path: str) -> Set[str]:
    parts = (path[:-3] if path.endswith(".py") else path).split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return {".".join(parts[i:]) for i in range(len(parts))} if parts else set()


def _imports(src: str) -> Set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if base:
                out.add(base)
            for a in n.names:
                out.add(f"{base}.{a.name}" if base else a.name)
    return out


def select_tests(tree: Dict[str, bytes], changed: Iterable[str]) -> Tuple[List[str], str]:
    """-> (test file paths, strategy). strategy: affected | all | none."""
    tests = sorted(p for p in tree if is_test_file(p))
    if not tests:
        return [], "none"
    changed = list(changed)
    direct = [p for p in changed if p in tests]
    cands: Set[str] = set()
    for c in changed:
        if c.endswith(".py"):
            cands |= _module_names(c)
    hit: Set[str] = set(direct)
    for t in tests:
        src = tree[t].decode("utf-8", errors="replace")
        imps = _imports(src)
        if any(i == c or i.startswith(c + ".") for i in imps for c in cands):
            hit.add(t)
            continue
        if any(not c.endswith(".py") and c.rsplit("/", 1)[-1] in src for c in changed):
            hit.add(t)               # non-python change (template, config) referenced by name
    if hit:
        return sorted(hit), "affected"
    return tests, "all"


def smoke_modules(tree: Dict[str, bytes], changed: Iterable[str]) -> List[str]:
    mods = []
    for c in changed:
        if not c.endswith(".py") or _SMOKE_SKIP.search(c) or is_test_file(c) or c not in tree:
            continue
        parts = c[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts and all(re.fullmatch(r"[A-Za-z_]\w*", x) for x in parts):
            mods.append(".".join(parts))
    return sorted(set(mods))[:30]


# --------------------------------------------------------------------------- workspace tar
def build_tar(files: Dict[str, bytes], extra: Optional[Dict[str, bytes]] = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for path, data in sorted({**files, **(extra or {})}.items()):
            ti = tarfile.TarInfo(path)
            ti.size, ti.mode, ti.mtime, ti.uid, ti.gid = len(data), 0o644, 0, 1000, 1000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


# --------------------------------------------------------------------------- output parsing
_SMOKE_PY = '''import importlib, json
for m in json.load(open(".sbx_smoke.json")):
    try:
        importlib.import_module(m)
        print("SMOKE|%s|OK" % m)
    except BaseException as e:
        print("SMOKE|%s|ERR|%s: %s" % (m, type(e).__name__, str(e).replace("\\n", " ")[:300]))
'''


def pytest_argv(paths: Sequence[str]) -> List[str]:
    return ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--maxfail=25", "--tb=short",
            "-o", "junit_family=xunit2", f"--junitxml={_REPORT}", *paths]


def _trim(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ..."


def parse_junit(xml_text: str) -> Tuple[int, int, int, List[CaseResult]]:
    passed = failed = skipped = 0
    cases: List[CaseResult] = []
    root = ET.fromstring(xml_text)
    for tc in root.iter("testcase"):
        cid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        bad = tc.find("failure")
        kind = "failed"
        if bad is None:
            bad = tc.find("error")
            kind = "error"
        if bad is not None:
            failed += 1
            cases.append(CaseResult(cid, kind, _trim(bad.get("message", ""), 300), _trim(bad.text or "", 1500)))
        elif tc.find("skipped") is not None:
            skipped += 1
        else:
            passed += 1
    return passed, failed, skipped, cases


@dataclass
class RawRun:
    stdout: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    error: str = ""           # infrastructure error (docker missing, tar failure ...)


_EXIT = re.compile(r"::SBX-EXIT:: (-?\d+)")
_JUNIT = re.compile(r"::SBX-JUNIT-BEGIN::\n(.*?)\n::SBX-JUNIT-END::", re.DOTALL)


def parse_run(raw: RawRun, mode: str, backend: str) -> RunReport:
    rep = RunReport(mode=mode, ran=False, backend=backend)
    if raw.error:
        rep.note = raw.error
        return rep
    out = raw.stdout
    m = _EXIT.search(out)
    rc = int(m.group(1)) if m else raw.exit_code
    console = out.split("::SBX-EXIT::")[0]
    rep.tail = console[-_MAX_TAIL:]
    if "::SBX-ERROR::" in out:
        rep.note = "sandbox could not unpack the workspace"
        return rep
    rep.ran = True
    if raw.timed_out or rc == 124:
        rep.timed_out = True
        rep.failures.append(CaseResult("sandbox::timeout", "timeout",
                                       f"run exceeded {settings.sandbox_test_timeout_s}s", _trim(rep.tail, 800)))
        return rep
    if mode == "smoke":
        for line in console.splitlines():
            if line.startswith("SMOKE|"):
                parts = line.split("|", 3)
                if len(parts) >= 3 and parts[2] == "OK":
                    rep.passed += 1
                elif len(parts) == 4:
                    rep.failed += 1
                    rep.failures.append(CaseResult(f"import::{parts[1]}", "error", _trim(parts[3], 300), parts[3]))
        if rc not in (0, None) and not rep.failures:
            rep.failures.append(CaseResult("smoke::crash", "error", f"exit {rc}", _trim(rep.tail, 800)))
        return rep
    jm = _JUNIT.search(out)
    if jm:
        try:
            rep.passed, rep.failed, rep.skipped, rep.failures = parse_junit(jm.group(1))
        except ET.ParseError:
            rep.failures.append(CaseResult("pytest::report", "error", "unreadable JUnit report", _trim(rep.tail, 800)))
    if rc == 5:
        rep.note = "pytest collected no tests"
    elif rc in (137, 9):
        rep.failures.append(CaseResult("sandbox::killed", "killed", "process killed (out of memory?)", _trim(rep.tail, 800)))
    elif rc not in (0, 1, None) and not rep.failures:
        rep.failures.append(CaseResult("pytest::run", "error", f"pytest exited with {rc}", _trim(rep.tail, 800)))
    return rep


# --------------------------------------------------------------------------- runners
def _exec(args: List[str], stdin: Optional[bytes], timeout: float,
          cwd: Optional[str] = None, env: Optional[dict] = None) -> Tuple[Optional[int], bytes, bool]:
    """Blocking subprocess helper. Runs in a worker thread because asyncio subprocesses are not
    available on every Windows event loop (uvicorn --reload uses the selector loop)."""
    try:
        p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout, cwd=cwd, env=env)
        return p.returncode, (p.stdout or b"") + (p.stderr or b""), False
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or b"") + (e.stderr or b""), True


ENTRY_SH = """#!/bin/sh
mkdir -p /work && cd /work || exit 97
tar -xf - || { echo "::SBX-ERROR:: could not extract workspace"; echo "::SBX-EXIT:: 97"; exit 0; }
export PYTHONPATH=/work PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
timeout -k 5 "${SBX_TIMEOUT:-180}" "$@"
rc=$?
echo
echo "::SBX-EXIT:: $rc"
if [ -f /tmp/sbx_report.xml ]; then
  echo "::SBX-JUNIT-BEGIN::"
  cat /tmp/sbx_report.xml
  echo
  echo "::SBX-JUNIT-END::"
fi
exit 0
"""


class DockerRunner:
    name = "docker"
    report_path = "/tmp/sbx_report.xml"
    _NEG_TTL_S = 10.0          # a "Docker is down" answer is trusted for this long at most

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._failed: Dict[str, Tuple[float, str]] = {}     # tag -> (monotonic time, error)
        self._avail: Optional[Tuple[float, bool, str]] = None
        self._procs: Set[subprocess.Popen] = set()          # running image builds, killable by stop()
        self._plock = threading.Lock()
        self._closing = False

    def image_tag(self, requirements: str) -> str:
        h = hashlib.sha256((settings.sandbox_base_image + "\n" + ENTRY_SH + "\n" + requirements).encode()).hexdigest()[:12]
        return f"{settings.sandbox_image_prefix}:{h}"

    def docker_args(self, tag: str, name: str, timeout: int, argv: List[str]) -> List[str]:
        return ["docker", "run", "--rm", "-i", "--name", name,
                "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", "1000:1000",
                "--memory", settings.sandbox_memory, "--cpus", settings.sandbox_cpus,
                "--pids-limit", str(settings.sandbox_pids),
                "--tmpfs", f"/work:rw,exec,nosuid,size={settings.sandbox_max_workspace_mb * 2}m,uid=1000,gid=1000",
                "--tmpfs", "/tmp:rw,nosuid,size=256m,uid=1000,gid=1000",
                "-e", f"SBX_TIMEOUT={timeout}",
                tag, *argv]

    async def available(self) -> Tuple[bool, str]:
        """Is the Docker daemon reachable? Cached: `docker version` costs 0.1-2 s (more on Docker Desktop)
        and it used to run before every single verification. A negative answer is cached only briefly
        so starting Docker Desktop after the app is noticed quickly."""
        ttl = settings.sandbox_available_ttl_s
        if ttl > 0 and self._avail is not None:
            ts, ok, info = self._avail
            if time.monotonic() - ts < (ttl if ok else min(ttl, self._NEG_TTL_S)):
                return ok, info
        ok, info = await self._probe()
        self._avail = (time.monotonic(), ok, info)
        return ok, info

    def invalidate_availability(self) -> None:
        self._avail = None

    async def _probe(self) -> Tuple[bool, str]:
        try:
            rc, out, _ = await asyncio.to_thread(
                _exec, ["docker", "version", "--format", "{{.Server.Version}}"], None, 15)
        except FileNotFoundError:
            return False, "Docker CLI not found on PATH"
        except Exception as e:  # noqa: BLE001
            return False, f"Docker check failed: {type(e).__name__}"
        if rc != 0:
            return False, "Docker is installed but the daemon is not reachable (is Docker Desktop running?)"
        return True, out.decode(errors="replace").strip()

    def _recent_failure(self, tag: str) -> Optional[str]:
        hit = self._failed.get(tag)
        if hit and time.monotonic() - hit[0] < settings.sandbox_build_retry_s:
            return hit[1]
        return None

    async def prepare(self, requirements: str) -> Optional[str]:
        """Make sure the dependency image exists. Returns an error string or None.

        A failed build is remembered for settings.sandbox_build_retry_s, then tried again: a network
        hiccup during a background build at startup must not switch tests off until the next restart."""
        tag = self.image_tag(requirements)
        err = self._recent_failure(tag)
        if err:
            return err
        async with self._locks.setdefault(tag, asyncio.Lock()):
            err = self._recent_failure(tag)        # a caller that waited for the lock must not redo a build that just failed
            if err:
                return err
            rc, _, _ = await asyncio.to_thread(_exec, ["docker", "image", "inspect", tag], None, 30)
            if rc == 0:
                self._failed.pop(tag, None)
                return None
            if self._closing:
                return "application is shutting down"
            err = await self._build(tag, requirements)
            if err is None:
                self._failed.pop(tag, None)
            elif not self._closing:                # a build killed by stop() is not a verdict on the requirements
                self._failed[tag] = (time.monotonic(), err)
            return err

    def _exec_tracked(self, args: List[str], timeout: float) -> Tuple[Optional[int], bytes, bool]:
        """Blocking like _exec, but the process can be killed from another thread (stop()).
        Without this a background `docker build` (minutes) would keep the application from exiting:
        asyncio waits for its worker threads, and the thread waits for docker."""
        with self._plock:                    # same lock as stop(): a build can not slip in after shutdown began
            if self._closing:
                return 1, b"application is shutting down", False
            p = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self._procs.add(p)
        try:
            out, _ = p.communicate(timeout=timeout)
            return p.returncode, out or b"", False
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            return None, out or b"", True
        finally:
            with self._plock:
                self._procs.discard(p)

    def stop(self) -> None:
        """Application shutdown: kill an in-flight image build and refuse to start new ones."""
        with self._plock:
            self._closing = True
            procs = list(self._procs)
        for p in procs:
            try:
                p.terminate()
            except Exception:  # noqa: BLE001 - already gone
                pass

    async def list_images(self) -> List[str]:
        """`repository:tag` of every image this sandbox built (settings.sandbox_image_prefix)."""
        prefix = settings.sandbox_image_prefix + ":"
        try:
            rc, out, _ = await asyncio.to_thread(
                _exec, ["docker", "image", "ls", settings.sandbox_image_prefix,
                        "--format", "{{.Repository}}:{{.Tag}}"], None, 30)
        except Exception:  # noqa: BLE001
            return []
        if rc != 0:
            return []
        lines = (ln.strip() for ln in out.decode(errors="replace").splitlines())
        return sorted({ln for ln in lines if ln.startswith(prefix) and not ln.endswith(":<none>")})

    async def remove_image(self, tag: str) -> bool:
        """`docker rmi` without --force: an image used by a running container is left alone."""
        if not tag.startswith(settings.sandbox_image_prefix + ":"):
            return False
        try:
            rc, _, _ = await asyncio.to_thread(_exec, ["docker", "rmi", tag], None, 60)
        except Exception:  # noqa: BLE001
            return False
        return rc == 0

    async def _build(self, tag: str, requirements: str) -> Optional[str]:
        with tempfile.TemporaryDirectory(prefix="coderag_build_") as ctx:
            c = Path(ctx)
            # write bytes: text mode would turn LF into CRLF on Windows and break the shebang
            (c / "sbx-entry").write_bytes(ENTRY_SH.encode("utf-8"))
            (c / "requirements.txt").write_bytes((requirements or "").encode("utf-8"))
            (c / "Dockerfile").write_bytes((
                f"FROM {settings.sandbox_base_image}\n"
                "COPY sbx-entry /usr/local/bin/sbx-entry\n"
                "RUN chmod 0755 /usr/local/bin/sbx-entry\n"
                "COPY requirements.txt /tmp/requirements.txt\n"
                "RUN pip install --no-cache-dir pytest -r /tmp/requirements.txt\n"
                'ENTRYPOINT ["/usr/local/bin/sbx-entry"]\n').encode("utf-8"))
            rc, out, timed = await asyncio.to_thread(
                self._exec_tracked, ["docker", "build", "-t", tag, ctx], settings.sandbox_build_timeout_s)
        if timed:
            return f"image build exceeded {settings.sandbox_build_timeout_s}s"
        if rc != 0:
            return "image build failed: " + out.decode(errors="replace")[-600:].strip()
        return None

    async def run(self, requirements: str, tar: bytes, argv: List[str], timeout: int) -> RawRun:
        tag = self.image_tag(requirements)
        name = f"coderag-sbx-{uuid.uuid4().hex[:10]}"
        argv = [a.replace(_REPORT, self.report_path) for a in argv]
        try:
            rc, out, timed = await asyncio.to_thread(
                _exec, self.docker_args(tag, name, timeout, argv), tar, timeout + 30)
        except FileNotFoundError:
            return RawRun(error="Docker CLI not found on PATH")
        if rc == 125:                          # docker itself failed (daemon gone ...): do not trust a cached "Docker is up"
            self.invalidate_availability()
        if timed:
            await asyncio.to_thread(_exec, ["docker", "rm", "-f", name], None, 30)   # CLI killed, container may live
        return RawRun(stdout=out.decode("utf-8", errors="replace"), exit_code=rc, timed_out=timed)


# Host variables a Python subprocess needs to start cleanly. Everything else is dropped on purpose.
_PASS_ENV = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
             "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")


def local_env(root: Path, host: Optional[dict] = None) -> dict:
    """Minimal environment for LocalRunner subprocesses.

    The home directory MUST be defined the way each OS looks it up: pathlib.Path.home() /
    os.path.expanduser read USERPROFILE (then HOMEDRIVE+HOMEPATH) on Windows and ignore HOME
    (since Python 3.8). With only HOME set, anything that asks for the home directory
    (pytest and its plugins included) dies with "RuntimeError: Could not determine home
    directory." and no test runs at all."""
    host = os.environ if host is None else host
    env = {k: host[k] for k in _PASS_ENV if k in host}
    r = str(root)
    drive, tail = os.path.splitdrive(r)
    env.update({
        "PYTHONPATH": r, "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",           # same output decoding on every host
        "HOME": r, "USERPROFILE": r,
        "HOMEDRIVE": drive or "C:", "HOMEPATH": tail or "\\",
        "APPDATA": r, "LOCALAPPDATA": r, "XDG_CACHE_HOME": r, "XDG_CONFIG_HOME": r,
    })
    return env


class LocalRunner:
    """NO ISOLATION. Runs on the host in a temp dir. For development and the test-suite only;
    enable explicitly with ENGINE_SANDBOX_BACKEND=local."""
    name = "local"

    async def available(self) -> Tuple[bool, str]:
        return True, "local (not isolated)"

    async def prepare(self, requirements: str) -> Optional[str]:
        return None

    async def run(self, requirements: str, tar: bytes, argv: List[str], timeout: int) -> RawRun:
        def work() -> RawRun:
            with tempfile.TemporaryDirectory(prefix="coderag_local_") as tmp:
                root = Path(tmp).resolve()
                with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
                    for m in tf.getmembers():
                        target = (root / m.name).resolve()
                        if root != target and root not in target.parents:
                            return RawRun(error="unsafe path in workspace tar")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(tf.extractfile(m).read())
                report = str(root / ".sbx_report.xml")
                cmd = [sys.executable if a == "python" else a.replace(_REPORT, report) for a in argv]
                rc, out, timed = _exec(cmd, None, timeout, cwd=str(root), env=local_env(root))
                text = out.decode("utf-8", errors="replace")
                text += f"\n::SBX-EXIT:: {124 if timed else rc}\n"
                if os.path.exists(report):
                    text += "::SBX-JUNIT-BEGIN::\n" + Path(report).read_text(encoding="utf-8") + "\n::SBX-JUNIT-END::\n"
                return RawRun(stdout=text, exit_code=rc, timed_out=timed)
        return await asyncio.to_thread(work)


def make_runner() -> object:
    return LocalRunner() if settings.sandbox_backend == "local" else DockerRunner()


def _unverified(report: RunReport, preexisting: int = 0) -> SandboxVerdict:
    """A run in which no test reached a result (pytest could not start, timed out, collected nothing,
    everything skipped) proves nothing about the change. It must never read as a pass, and the baseline
    must not be allowed to cancel it out: report 'not run' so callers fall back to static checks only."""
    why = report.note
    if not why and report.timed_out:
        why = f"run exceeded {settings.sandbox_test_timeout_s}s"
    if not why and report.failures:
        why = report.failures[0].message
    if not why and report.skipped:
        why = f"all {report.skipped} selected test(s) were skipped"
    last = next((ln.strip() for ln in reversed(report.tail.splitlines()) if ln.strip()), "")
    report.ran = False
    report.note = "no test produced a result (" + (why or "nothing executed") + ")" + (f": {_trim(last, 200)}" if last else "")
    return SandboxVerdict(report, preexisting=preexisting)


# --------------------------------------------------------------------------- service
class SandboxService:
    def __init__(self, store, runner=None, enabled: Optional[bool] = None, emit: Optional[Callable] = None):
        self.store = store
        self.runner = runner or make_runner()
        self.enabled = settings.sandbox_enabled if enabled is None else enabled
        self._emit_fn = emit
        self._baseline: Dict[tuple, Tuple[Set[str], bool]] = {}
        self._ready: Set[str] = set()

    def _emit(self, msg: str, level: str = "INFO") -> None:
        try:
            if self._emit_fn:
                self._emit_fn("Sandbox", msg, level)
            else:
                from services.notifier import notifier
                notifier.notify("Sandbox", msg, level=level)
        except Exception:
            pass

    @staticmethod
    def _requirements(tree: Dict[str, bytes]) -> str:
        for name in settings.sandbox_requirements_files:
            if name in tree:
                return tree[name].decode("utf-8", errors="replace")
        return ""

    # ------------------------------------------------------------------ pre-warm
    async def requirements_for(self, repo_id: str, rev: Optional[int] = None) -> str:
        """The requirements text verify() would use for the repo view at `rev` (default: latest).
        Same file order as _requirements(), but reads only those files, not the whole repo."""
        rev = await self.store.latest_rev(repo_id) if rev is None else rev
        for name in settings.sandbox_requirements_files:
            try:
                f = await self.store.get_file(repo_id, name, rev)
            except ValueError:
                continue
            if f is not None:
                return f.data.decode("utf-8", errors="replace")
        return ""

    async def prewarm(self, repo_id: str, rev: Optional[int] = None) -> dict:
        """Build the dependency image for a repo ahead of the first change request. Never raises for
        an expected situation; returns {repo_id, status, tag, note[, seconds][, retry]} where status is
        ready | skipped | failed. `retry` is set when trying again later can help (Docker not up yet)."""
        out: dict = {"repo_id": repo_id, "status": "skipped", "tag": "", "note": ""}
        if not self.enabled:
            out["note"] = "sandbox disabled"
            return out
        if self.runner.name != "docker":
            out["note"] = f"the {self.runner.name} backend needs no image"
            return out
        rev = await self.store.latest_rev(repo_id) if rev is None else rev
        if not any(p.endswith(".py") for p in await self.store.list_paths(repo_id, rev)):
            out["note"] = "no Python files in this repo (verify never starts a container for it)"
            return out
        ok, info = await self.runner.available()
        if not ok:
            out.update(note=info, retry=True)
            return out

        requirements = await self.requirements_for(repo_id, rev)
        tag = self.runner.image_tag(requirements)
        out["tag"] = tag
        if tag in self._ready:
            out.update(status="ready", note="image already prepared")
            return out
        self._emit(f"Preparing the sandbox image for repo {repo_id} in the background "
                   f"(a first build can take minutes, chat is not blocked)")
        t0 = time.monotonic()
        err = await self.runner.prepare(requirements)
        if err:
            out.update(status="failed", note=err)
            self._emit(f"Sandbox image for repo {repo_id} could not be built: {err[:300]}. "
                       f"Tests will be skipped for this repo until it builds.", "WARNING")
            return out
        self._ready.add(tag)
        secs = round(time.monotonic() - t0, 1)
        out.update(status="ready", seconds=secs, note="image present")
        if secs >= 2:                              # an image that already existed answers in milliseconds: stay quiet
            self._emit(f"Sandbox image for repo {repo_id} is ready ({secs}s)", "SUCCESS")
        return out

    async def prune_images(self, repo_ids: Iterable[str]) -> List[str]:
        """Remove sandbox images that no repo's current requirements map to (old requirement hashes,
        a changed base image). Only images under settings.sandbox_image_prefix are ever touched."""
        if not self.enabled or self.runner.name != "docker":
            return []
        ok, _ = await self.runner.available()
        if not ok:
            return []
        keep = {self.runner.image_tag(await self.requirements_for(r)) for r in repo_ids}
        removed = []
        for tag in await self.runner.list_images():
            if tag not in keep and await self.runner.remove_image(tag):
                removed.append(tag)
        if removed:
            self._ready.difference_update(removed)
            self._emit(f"Removed {len(removed)} unused sandbox image(s)")
        return removed

    def close(self) -> None:
        """Application shutdown: stop anything the runner has running in the background."""
        stop = getattr(self.runner, "stop", None)
        if stop:
            stop()

    async def _execute(self, tree: Dict[str, bytes], mode: str, selected: List[str],
                       smoke: List[str], requirements: str) -> RunReport:
        extra: Dict[str, bytes] = {}
        if mode == "smoke":
            extra = {".sbx_smoke.py": _SMOKE_PY.encode(), ".sbx_smoke.json": json.dumps(smoke).encode()}
            argv = ["python", ".sbx_smoke.py"]
        else:
            argv = pytest_argv(selected)
        tar = build_tar(tree, extra)
        if len(tar) > settings.sandbox_max_workspace_mb * 1024 * 1024:
            return RunReport(mode=mode, ran=False, backend=self.runner.name, note="workspace too large for the sandbox")
        t0 = time.monotonic()
        raw = await self.runner.run(requirements, tar, argv, settings.sandbox_test_timeout_s)
        rep = parse_run(raw, mode, self.runner.name)
        rep.duration = time.monotonic() - t0
        rep.selected = selected if mode == "pytest" else smoke
        return rep

    async def verify(self, repo_id: str, base_rev: int, working: Dict[str, bytes]) -> SandboxVerdict:
        """Run the proposal. `working` = {path: new bytes} on top of the repo at `base_rev`."""
        none = lambda note: SandboxVerdict(RunReport(mode="none", ran=False, note=note, backend=self.runner.name))
        if not self.enabled:
            return none("sandbox disabled")
        ok, info = await self.runner.available()
        if not ok:
            return none(info)

        base_tree = await self.store.snapshot_bytes(repo_id, base_rev)
        tree = {**base_tree, **working}
        changed = sorted(working)
        selected, strategy = select_tests(tree, changed)
        smoke: List[str] = []
        mode = "pytest"
        if not selected:
            smoke = smoke_modules(tree, changed)
            if not smoke:
                return none("no tests in the repo and no importable python module changed")
            mode, strategy = "smoke", "smoke"
        requirements = self._requirements(tree)

        if self.runner.name == "docker":
            tag = self.runner.image_tag(requirements)
            if tag not in self._ready:
                self._emit("Preparing sandbox image (a background build may already be running; "
                           "a first build can take minutes)")
                err = await self.runner.prepare(requirements)
                if err:
                    return none(err)
                self._ready.add(tag)

        self._emit(f"Running {len(selected) if mode == 'pytest' else len(smoke)} "
                   f"{'test file(s)' if mode == 'pytest' else 'module import(s)'} [{strategy}] in {self.runner.name}")
        report = await self._execute(tree, mode, selected, smoke, requirements)
        report.strategy = strategy
        if not report.ran:
            return SandboxVerdict(report)
        if not report.failures:
            return _unverified(report) if report.passed + report.failed == 0 else SandboxVerdict(report)

        # failures: subtract what was already broken before the change
        key = (repo_id, base_rev, mode, tuple(selected), tuple(smoke), hashlib.sha256(requirements.encode()).hexdigest())
        if key not in self._baseline:
            b_selected = [p for p in selected if p in base_tree]
            b_smoke = [m for m in smoke if (m.replace(".", "/") + ".py") in base_tree
                       or (m.replace(".", "/") + "/__init__.py") in base_tree]
            if (mode == "pytest" and not b_selected) or (mode == "smoke" and not b_smoke):
                self._baseline[key] = (set(), False)
            else:
                self._emit("Failures found, checking whether they already existed before the change", "WARNING")
                base_report = await self._execute(base_tree, mode, b_selected, b_smoke, requirements)
                self._baseline[key] = ({c.id for c in base_report.failures}, base_report.timed_out)
            if len(self._baseline) > 64:
                self._baseline.pop(next(iter(self._baseline)))
        base_ids, base_timed_out = self._baseline[key]
        fresh = [c for c in report.failures
                 if c.id not in base_ids and not (c.kind == "timeout" and base_timed_out)]
        if not fresh and report.passed + report.failed == 0:
            return _unverified(report, preexisting=len(report.failures))   # same crash/timeout on the base: not a pass
        return SandboxVerdict(report, new_failures=fresh, preexisting=len(report.failures) - len(fresh))