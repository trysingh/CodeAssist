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
_IMAGE_REV = "2"          # part of the image tag: bump it when the Dockerfile text changes so old images are rebuilt
_TIKTOKEN_REQ = re.compile(r"^\s*tiktoken\b", re.IGNORECASE | re.MULTILINE)
_GENERIC_MSG = {"", "collection failure", "test setup failure", "test teardown failure"}

_NET_ERR = re.compile(r"NameResolutionError|Name or service not known|Temporary failure in name resolution|getaddrinfo failed|"
                      r"Connection refused|ConnectionError|Max retries exceeded|Network is unreachable", re.IGNORECASE)
_HOST = re.compile(r"host='([^']+)'|Failed to resolve '([^']+)'|https?://([\w.\-]+)")
_NO_MOD = re.compile(r"No module named '([\w.]+)'")


def hint_for(text: str, local_tops: Optional[Set[str]] = None) -> str:
    """One actionable sentence for the usual reasons a test fails inside the sandbox but not on the host.
    '' = no idea. Text only, no side effects."""
    text = text or ""
    m = _NO_MOD.search(text)
    if m:
        top = m.group(1).split(".")[0]
        if not local_tops or top not in local_tops:
            return (f"package '{top}' is not installed in the sandbox image. It exists on your machine but is missing from "
                    f"the requirements file the sandbox uses; add it there and the image rebuilds by itself")
    if _NET_ERR.search(text):
        hm = _HOST.search(text)
        host = next((g for g in (hm.groups() if hm else ()) if g), "")
        extra = " (tiktoken downloads its vocabulary on first use)" if "tiktoken" in text else ""
        return (f"code tried to reach {host or 'the network'}{extra}, but the sandbox runs with --network none. Fetch it while "
                f"the image is built (settings.sandbox_build_run) or mock it in the test")
    if re.search(r"Read-only file system|\[Errno 30\]", text):
        return "code wrote to the read-only root filesystem; only /tmp and /work are writable in the sandbox"
    if re.search(r"PermissionError|Permission denied", text):
        return "permission denied inside the sandbox (non-root user 1000, read-only root): write under /tmp or tmp_path"
    return ""


def write_log(scope: str, label: str, text: str) -> str:
    """Keep the raw console output of a sandbox run on disk (best effort, rotated). -> absolute path or ''."""
    try:
        safe = lambda v: re.sub(r"[^\w.-]", "_", v or "_")[:60]
        d = (Path(settings.storage_dir) / "sandbox_logs" / safe(scope)).resolve()
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe(label)}-{uuid.uuid4().hex[:4]}.log"
        f.write_text((text or "")[-2_000_000:], encoding="utf-8", errors="replace")
        for old in sorted(d.glob("*.log"), key=lambda x: x.stat().st_mtime)[:-max(1, settings.sandbox_logs_keep)]:
            old.unlink(missing_ok=True)
        return str(f)
    except Exception:  # noqa: BLE001 - a log is a convenience, never a reason to fail a run
        return ""


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
    exit_code: Optional[int] = None
    passed_ids: List[str] = field(default_factory=list)
    log_path: str = ""


def _case_dict(c: CaseResult) -> dict:
    return {"id": c.id, "kind": c.kind, "message": _trim(c.message, 240)}


@dataclass
class SandboxVerdict:
    report: RunReport
    new_failures: List[CaseResult] = field(default_factory=list)
    preexisting: int = 0
    preexisting_cases: List[CaseResult] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    proof: dict = field(default_factory=dict)       # fail->pass evidence for generated tests

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
                "preexisting": self.preexisting, "duration": round(r.duration, 1),
                "exit_code": r.exit_code, "log": r.log_path,
                "new_failure_details": [_case_dict(c) for c in self.new_failures[:8]],
                "preexisting_details": [_case_dict(c) for c in self.preexisting_cases[:8]],
                "hints": self.hints[:3], "proof": self.proof}


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

def closest_tests(tests: Iterable[str], changed: Iterable[str]) -> List[str]:
    """Narrow a too-big 'affected' set to changed test files and tests named after a changed file
    (services/contract.py -> tests/test_contract.py)."""
    changed = list(changed)
    stems = {c.rsplit("/", 1)[-1].rsplit(".", 1)[0] for c in changed if c.endswith(".py")} - {"__init__"}
    return sorted(t for t in tests
                  if t in changed or any(s in t.rsplit("/", 1)[-1] for s in stems))


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
    # --continue-on-collection-errors: one test file that cannot be imported (offline download, missing package) must not
    # abort the whole session, otherwise NOT A SINGLE test runs and the run proves nothing
    return ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--maxfail=25", "--tb=short",
            "--continue-on-collection-errors",
            "-o", "junit_family=xunit2", f"--junitxml={_REPORT}", *paths]


def _trim(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ..."


_PROGRESS = re.compile(r"^[.FEsxX]+(?:\s+\[\s*\d+%\])?\s*$")


def pytest_progress(console: str) -> Tuple[int, int]:
    """(passed, failed) counted from pytest -q's progress characters. A run killed at the time cap writes no JUnit report,
    but the dots it printed before it died still say how far it got."""
    ok = bad = 0
    for ln in (console or "").splitlines():
        if _PROGRESS.match(ln.strip()):
            body = ln.strip().split()[0]
            ok += body.count(".")
            bad += body.count("F") + body.count("E")
    return ok, bad


def _tail(s: str, n: int) -> str:
    """Last n chars: in a traceback the cause is at the END."""
    s = (s or "").strip()
    return s if len(s) <= n else "... " + s[-n:]


def _headline(body: str) -> str:
    """The line that says what went wrong: the last 'E   ...' line of a pytest traceback, else the last non-empty line."""
    lines = [ln.rstrip() for ln in (body or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("E "):
            return _trim(ln[1:].strip(), 300)
    return _trim(lines[-1].strip(), 300) if lines else ""


def junit_passed_ids(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    return [f"{tc.get('classname', '')}::{tc.get('name', '')}" for tc in root.iter("testcase")
            if tc.find("failure") is None and tc.find("error") is None and tc.find("skipped") is None]


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
            body, msg = (bad.text or "").strip(), (bad.get("message") or "").strip()
            if msg == "collection failure":
                kind = "collect"                      # the test file could not even be imported: no test in it ran
            if msg in _GENERIC_MSG and body:
                msg = f"{msg}: {_headline(body)}" if msg else _headline(body)
            cases.append(CaseResult(cid, kind, _trim(msg, 300), _tail(body, 2400)))
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
    rep.exit_code = rc
    if m is None and rc in (125, 126, 127):          # `docker run` itself failed: the container never started
        last = next((ln.strip() for ln in reversed(console.splitlines()) if ln.strip()), "")
        rep.note = f"docker could not start the sandbox container (exit {rc}): {_trim(last, 300)}"
        return rep
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
            rep.passed_ids = junit_passed_ids(jm.group(1))
        except ET.ParseError:
            rep.failures.append(CaseResult("pytest::report", "error", "unreadable JUnit report", _trim(rep.tail, 800)))
    if rc == 5:
        rep.note = "pytest collected no tests"
    elif rc in (137, 9):
        rep.failures.append(CaseResult("sandbox::killed", "killed",
                                       f"process killed (exit {rc}, out of memory? the sandbox limit is {settings.sandbox_memory})",
                                       _trim(rep.tail, 800)))
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

    def dockerfile(self, requirements: str) -> str:
        """The image recipe. The build is the ONLY step with network: whatever the code downloads at import or test
        time (tiktoken vocabularies, nltk data ...) has to be fetched here, the run itself has --network none."""
        lines = [f"FROM {settings.sandbox_base_image}",
                 "COPY sbx-entry /usr/local/bin/sbx-entry",
                 "RUN chmod 0755 /usr/local/bin/sbx-entry",
                 "COPY requirements.txt /tmp/requirements.txt",
                 "RUN pip install --no-cache-dir pytest -r /tmp/requirements.txt"]
        if _TIKTOKEN_REQ.search(requirements or ""):
            lines += ["ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken",
                      "RUN mkdir -p /opt/tiktoken && python -c \"import tiktoken; "
                      "[tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]\" && chmod -R a+rX /opt/tiktoken "
                      "|| echo 'tiktoken vocabulary prefetch skipped (no network while building?)'"]
        lines += [f"RUN {c}" for c in settings.sandbox_build_run]
        lines.append('ENTRYPOINT ["/usr/local/bin/sbx-entry"]')
        return "\n".join(lines) + "\n"

    def image_tag(self, requirements: str) -> str:
        h = hashlib.sha256("\n".join((_IMAGE_REV, settings.sandbox_base_image, ENTRY_SH, self.dockerfile(requirements),
                                      requirements)).encode()).hexdigest()[:12]
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
            (c / "Dockerfile").write_bytes(self.dockerfile(requirements).encode("utf-8"))
            rc, out, timed = await asyncio.to_thread(
                self._exec_tracked, ["docker", "build", "-t", tag, ctx], settings.sandbox_build_timeout_s)
        text = out.decode(errors="replace")
        log = write_log("_image_builds", tag.split(":")[-1] + ("-FAILED" if rc != 0 or timed else ""), text)
        if timed:
            return f"image build exceeded {settings.sandbox_build_timeout_s}s" + (f" (full build log: {log})" if log else "")
        if rc != 0:
            return "image build failed: " + text[-1500:].strip() + (f"\n(full build log: {log})" if log else "")
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
        ok, bad = pytest_progress(report.tail)
        if ok or bad:
            why += (f" after {ok + bad} test(s) ({ok} passed, {bad} failed so far); raise ENGINE_SANDBOX_TEST_TIMEOUT_S "
                    f"or run fewer tests")
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

    @staticmethod
    def _hints(failures: List[CaseResult], tree: Dict[str, bytes]) -> List[str]:
        tops = {p.split("/")[0].removesuffix(".py") for p in tree}
        out: List[str] = []
        for c in failures[:6]:
            h = hint_for(f"{c.message}\n{c.detail}", tops)
            if h and h not in out:
                out.append(h)
        return out[:3]

    def _announce(self, rep: RunReport, what: str = "") -> None:
        """What happened inside the container, in the notification stream: counts, the failing ids and WHY, the log file."""
        if not rep.ran:
            self._emit(f"{what}the sandbox could not run the tests: {rep.note}", "WARNING")
            return
        line = f"{what}{rep.passed} passed, {rep.failed} failed, {rep.skipped} skipped in {rep.duration:.1f}s"
        if rep.failures:
            shown = " | ".join(f"{c.id} [{c.kind}]: {_trim((c.message or '').splitlines()[0] if c.message else '', 160)}"
                               for c in rep.failures[:3])
            line += f" - {shown}" + (f" (+{len(rep.failures) - 3} more)" if len(rep.failures) > 3 else "")
        if rep.log_path:
            line += f" [log: {rep.log_path}]"
        self._emit(line, "WARNING" if rep.failures else "INFO")

    async def _execute(self, tree: Dict[str, bytes], mode: str, selected: List[str],
                       smoke: List[str], requirements: str, repo_id: str = "", label: str = "") -> RunReport:
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
        rep.log_path = write_log(repo_id or "_", label or mode, raw.stdout or raw.error)
        return rep

    @staticmethod
    def _case_file(case_id: str, paths: Sequence[str]) -> Optional[str]:
        """Which of `paths` (test files) a JUnit case id belongs to: 'tests.test_x::test_y' or '::tests.test_x' (collection)."""
        head = case_id.split("::")[0] or (case_id.split("::")[1] if "::" in case_id else "")
        for p in paths:
            mod = p[:-3].replace("/", ".")
            if head == mod or head.startswith(mod + "."):
                return p
        return None

    async def _prove(self, repo_id: str, base_tree: Dict[str, bytes], tree: Dict[str, bytes],
                     new_tests: Sequence[str], requirements: str, after: RunReport) -> dict:
        """fail->pass proof: the GENERATED tests against the OLD code. A test that fails there and passes on the new code
        demonstrates the change; one that passes on both is only a regression test and says nothing about it.
        Called only when everything passed on the new code (otherwise the fix loop is busy anyway)."""
        old = {**base_tree, **{p: tree[p] for p in new_tests}}
        rep = await self._execute(old, "pytest", list(new_tests), [], requirements, repo_id=repo_id, label="fail-to-pass-on-old-code")
        mods = {p: p[:-3].replace("/", ".") for p in new_tests}
        n_after = sum(1 for cid in after.passed_ids if self._case_file(cid, new_tests))
        out = {"checked": rep.ran, "files": list(new_tests), "tests": n_after}
        if not rep.ran:
            out["note"] = rep.note or "the old-code run did not execute"
            return out
        failing = {self._case_file(c.id, new_tests) for c in rep.failures} - {None}
        out["fail_on_old"] = sorted(failing)
        out["pass_on_old"] = [p for p in new_tests if p not in failing]
        out["proves"] = len(failing)
        self._emit(f"fail->pass on the old code: {len(failing)} of {len(new_tests)} generated test file(s) fail without the change "
                   f"(they demonstrate it)" + (f"; pass on old code too: {', '.join(out['pass_on_old'])}" if out["pass_on_old"] else ""),
                   "SUCCESS" if failing else "WARNING")
        return out

    async def verify_red(self, repo_id: str, base_rev: int, overlay: Dict[str, bytes], tests: Sequence[str],
                         label: str = "red-check") -> RunReport:
        """Run only `tests` on the base tree plus `overlay`: red check (tests + contract stubs) and per-task green check."""
        none = lambda note: RunReport(mode="none", ran=False, note=note, backend=self.runner.name)
        if not self.enabled:
            return none("sandbox disabled")
        ok, info = await self.runner.available()
        if not ok:
            return none(info)
        tree = {**await self.store.snapshot_bytes(repo_id, base_rev), **overlay}
        requirements = self._requirements(tree)
        if self.runner.name == "docker":
            tag = self.runner.image_tag(requirements)
            if tag not in self._ready:
                err = await self.runner.prepare(requirements)
                if err:
                    return none(err)
                self._ready.add(tag)
        return await self._execute(tree, "pytest", list(tests), [], requirements, repo_id=repo_id, label=label)

    async def verify(self, repo_id: str, base_rev: int, working: Dict[str, bytes],
                     new_tests: Sequence[str] = ()) -> SandboxVerdict:
        """Run the proposal. `working` = {path: new bytes} on top of the repo at `base_rev`.
        `new_tests` = generated test files inside `working`: they always run, and (when everything passes) are re-run
        against the OLD code to prove they fail without the change."""
        none = lambda note: SandboxVerdict(RunReport(mode="none", ran=False, note=note, backend=self.runner.name))
        if not self.enabled:
            return none("sandbox disabled")
        ok, info = await self.runner.available()
        if not ok:
            return none(info)

        base_tree = await self.store.snapshot_bytes(repo_id, base_rev)
        tree = {**base_tree, **working}
        changed = sorted(working)
        gen = sorted(p for p in new_tests if p in working)
        src_changed = [c for c in changed if c not in gen] or changed
        selected, strategy = select_tests(tree, src_changed)       # existing tests chosen by what the SOURCE change touches
        cap = settings.sandbox_all_max_files
        if strategy == "all" and cap and len(selected) > cap:
            self._emit(f"No test refers to the changed file(s) and the suite has {len(selected)} test files (limit {cap}), so it is "
                       f"not run in full: it would hit the {settings.sandbox_test_timeout_s}s cap and prove nothing. "
                       f"ENGINE_SANDBOX_ALL_MAX_FILES=0 always runs everything", "WARNING")
            selected, strategy = [], "limited"
        elif strategy == "affected" and cap and len(selected) > cap:
            n = len(selected)
            selected = closest_tests(selected, src_changed)
            strategy = "narrowed"
            self._emit(f"{n} test files import the changed file(s) (limit {cap}); running the {len(selected)} closest instead. "
                        f"ENGINE_SANDBOX_ALL_MAX_FILES=0 runs all {n}", "WARNING")
        if gen:
            selected = sorted(set(selected) | set(gen))
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
        report = await self._execute(tree, mode, selected, smoke, requirements, repo_id=repo_id, label="change")
        report.strategy = strategy
        self._announce(report)
        if not report.ran:
            return SandboxVerdict(report)
        if not report.failures:
            if report.passed + report.failed == 0:
                return _unverified(report)
            verdict = SandboxVerdict(report)
            if gen and mode == "pytest":
                verdict.proof = await self._prove(repo_id, base_tree, tree, gen, requirements, report)
            return verdict

        if report.timed_out and all(c.kind == "timeout" for c in report.failures):
            ok, bad = pytest_progress(report.tail)
            if ok and not bad:
                # Tests were passing and the cap simply cut the run off (the suite is too big for it): no verdict either way, and
                # the untouched base would only hit the same cap, so do not wait for a second one. A run that produced NO
                # progress (a hang at import or in the first test) still goes to the base comparison below.
                return _unverified(report)

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
                base_report = await self._execute(base_tree, mode, b_selected, b_smoke, requirements,
                                                  repo_id=repo_id, label="base")
                self._announce(base_report, "untouched base: ")
                self._baseline[key] = ({c.id for c in base_report.failures}, base_report.timed_out)
            if len(self._baseline) > 64:
                self._baseline.pop(next(iter(self._baseline)))
        base_ids, base_timed_out = self._baseline[key]
        fresh = [c for c in report.failures
                 if c.id not in base_ids and not (c.kind == "timeout" and base_timed_out)]
        old_cases = [c for c in report.failures if c not in fresh]
        hints = self._hints(report.failures, tree)
        if old_cases:
            self._emit(f"{len(old_cases)} of {len(report.failures)} failure(s) also occur on the untouched base, so they are not "
                       f"caused by this change: {', '.join(c.id for c in old_cases[:4])}", "WARNING")
        for h in hints:
            self._emit(f"Likely cause: {h}", "WARNING")
        if not fresh and report.passed == 0:
            # Nothing passed and nothing new failed: the same breakage exists without the change, so not one test says anything
            # about it. That is 'not run', never a pass (this is the all-tests-fail-offline case).
            v = _unverified(report, preexisting=len(report.failures))
            v.preexisting_cases, v.hints = old_cases, hints
            if hints:
                v.report.note += f" Likely cause: {hints[0]}"
            return v
        v = SandboxVerdict(report, new_failures=fresh, preexisting=len(old_cases), preexisting_cases=old_cases, hints=hints)
        if v.ok and gen and mode == "pytest":
            v.proof = await self._prove(repo_id, base_tree, tree, gen, requirements, report)
        return v