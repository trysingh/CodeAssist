"""Generated patches must survive `git apply`; a plan file the patcher skipped gets a focused retry.
Run from the repo root:  pytest tests/test_patch_and_gap.py -q   (no Ollama, Chroma or network; git must be on PATH)
"""
import asyncio
import shutil
import subprocess
import sys
import uuid

import pytest

from config.settings import settings
from services.patch_service import diff_stats, join_diffs, make_diff
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore

run = asyncio.run
HAVE_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAVE_GIT, reason="git not installed")


def _git(args, cwd, autocrlf="false"):
    # -c core.autocrlf=false: on Windows the default (true) makes `git apply` write CRLF into the working tree, which would
    # change the bytes this test compares. A separate test below covers every autocrlf setting on a CRLF tree.
    return subprocess.run(["git", "-c", f"core.autocrlf={autocrlf}", "-c", "core.safecrlf=false"] + args,
                          cwd=cwd, capture_output=True)


def _round_trip(tmp_path, files):
    """files = {path: (old|None, new|None)}. Build the joined patch, `git apply` it on a repo holding the old bytes,
    and require the new bytes on disk."""
    _git(["init", "-q"], tmp_path)
    for p, (old, _new) in files.items():
        if old is not None:
            f = tmp_path / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(old)
    patch = join_diffs([make_diff(p, o, n) for p, (o, n) in files.items()])
    assert patch.endswith("\n")
    pf = tmp_path.parent / f"{uuid.uuid4().hex}.patch"
    pf.write_bytes(patch.encode("utf-8"))
    chk = _git(["apply", "--check", str(pf)], tmp_path)
    assert chk.returncode == 0, chk.stderr.decode()
    ap = _git(["apply", "--whitespace=nowarn", str(pf)], tmp_path)
    assert ap.returncode == 0, ap.stderr.decode()
    for p, (_old, new) in files.items():
        f = tmp_path / p
        assert (f.read_bytes() if f.exists() else None) == new, p


@needs_git
@pytest.mark.parametrize("name,old,new", [
    ("plain",            b"a\nb\nc\nd\ne\nf\ng\n",           b"a\nb\nc\nX\ne\nf\ng\n"),
    ("no_eol_old",       b"a\nb\nc\nd",                      b"a\nb\nc\nd\ne\n"),
    ("no_eol_new",       b"a\nb\nc\nd\n",                    b"a\nb\nc\nd"),
    ("no_eol_both",      b"a\nb\nc\nd",                      b"a\nb\nc\nX"),
    ("crlf",             b"a\r\nb\r\nc\r\nd\r\ne\r\n",       b"a\r\nb\r\nc\r\nX\r\ne\r\n"),
    ("crlf_to_lf",       b"a\r\nb\r\n",                      b"a\nb\n"),
    ("mixed_endings",    b"a\r\nb\nc\r\n",                   b"a\r\nB\nc\r\n"),
    ("blank_context",    b"a\n\nb\n\nc\n",                   b"a\n\nb\n\nZ\n"),
    ("form_feed",        b"a\n\x0cb\nc\n",                   b"a\n\x0cb\nC\n"),
    ("unicode_separator", "a\nb\u2028c\nd\n".encode(),       "a\nb\u2028c\nD\n".encode()),
    ("dash_lines",       b"-- keep\nSELECT 1;\n",            b"-- keep\n-- new\nSELECT 2;\n"),
    ("bom",              b"\xef\xbb\xbfa\nb\n",              b"\xef\xbb\xbfa\nB\n"),
])
def test_edit_applies_with_git(tmp_path, name, old, new):
    _round_trip(tmp_path, {f"{name}.txt": (old, new)})


@needs_git
def test_create_delete_and_odd_names_apply_with_git(tmp_path):
    _round_trip(tmp_path, {
        "pkg/new.py": (None, b"hello\nworld\n"),
        "pkg/new_no_eol.py": (None, b"hello"),
        "pkg/empty.py": (None, b""),
        "gone.py": (b"bye\n", None),
        "gone_empty.py": (b"", None),
        "templates/index copy.html": (b"<a>\n<b>\n", b"<a>\n<c>\n"),      # a real file name in this repo
        "templates/new copy.html": (None, b"x\n"),
        "caf\u00e9/\u00fcber.py": (b"x\n", b"y\n"),
        **({} if sys.platform == "win32" else {'we"ird.py': (b"x\n", b"y\n")}),     # '"' is not a legal file name on Windows
    })


@needs_git
def test_non_utf8_file_is_skipped_not_corrupting_the_patch(tmp_path):
    _git(["init", "-q"], tmp_path)
    (tmp_path / "x.py").write_bytes(b"1\n2\n3\n")
    patch = join_diffs([make_diff("a.bin", b"\xff\xfe", b"\xff\xff"), make_diff("x.py", b"1\n2\n3\n", b"1\nTWO\n3\n")])
    pf = tmp_path.parent / f"{uuid.uuid4().hex}.patch"
    pf.write_bytes(patch.encode("utf-8"))
    assert _git(["apply", str(pf)], tmp_path).returncode == 0
    assert (tmp_path / "x.py").read_bytes() == b"1\nTWO\n3\n"


@needs_git
@pytest.mark.parametrize("autocrlf", ["true", "false", "input"])
def test_patch_of_a_crlf_tree_applies_under_every_autocrlf_setting(tmp_path, autocrlf):
    """A Windows checkout: working tree CRLF, repository LF. The patch keeps the tree's own line endings and must apply
    strictly (no --ignore-whitespace) whatever core.autocrlf is."""
    old = b"a\r\nb\r\nc\r\nd\r\ne\r\nf\r\ng\r\n"
    new = old.replace(b"d\r\n", b"X\r\n")
    _git(["init", "-q"], tmp_path)
    (tmp_path / "f.txt").write_bytes(old.replace(b"\r\n", b"\n"))
    _git(["add", "f.txt"], tmp_path)
    (tmp_path / "f.txt").write_bytes(old)
    pf = tmp_path.parent / f"{uuid.uuid4().hex}.patch"
    pf.write_bytes(make_diff("f.txt", old, new).encode("utf-8"))
    res = _git(["apply", "--check", str(pf)], tmp_path, autocrlf=autocrlf)
    assert res.returncode == 0, res.stderr.decode()


@needs_git
def test_conflict_comment_lines_before_the_first_file_do_not_break_git_apply(tmp_path):
    _git(["init", "-q"], tmp_path)
    (tmp_path / "x.py").write_bytes(b"1\n2\n3\n")
    head = "# WARNING: these files differ on disk\n#   other.py: file changed on disk since ingest\n"
    pf = tmp_path.parent / f"{uuid.uuid4().hex}.patch"
    pf.write_bytes((head + make_diff("x.py", b"1\n2\n3\n", b"1\nTWO\n3\n")).encode("utf-8"))
    assert _git(["apply", str(pf)], tmp_path).returncode == 0
    assert (tmp_path / "x.py").read_bytes() == b"1\nTWO\n3\n"


def test_diff_shape_and_stats():
    assert make_diff("a.py", b"x\n", b"x\n") == ""
    d = make_diff("sql.sql", b"-- keep\nSELECT 1;\n", b"-- keep\n-- new\nSELECT 2;\n")
    assert d.startswith("diff --git a/sql.sql b/sql.sql\n") and d.endswith("\n")
    assert diff_stats(d) == (2, 1)                       # '-- new' is an added line, not a header
    assert "\\ No newline at end of file\n" in make_diff("n.py", b"a", b"b")


# --------------------------------------------------------------------------- plan gap -> focused retry
PLAN = ("FEASIBLE: yes, small\nFILES:\n- a.py - add the field\n- b.js - show the field in the UI\n- c.py - extra\n"
        "NEW FILES:\n- none\nRISKS:\n- none")
FILES = [("a.py", b"X = 1\n"), ("b.js", b"const y = 1;\n"), ("c.py", b"Z = 1\n")]


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class FakeDoc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class FakeVector:
    def similarity_search(self, repo_id, q, k=4):
        return [FakeDoc(p) for p, _ in FILES]


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


async def _no_json(*a, **k):
    return None


async def _service(llm):
    s, rid = RepoStore(), f"r_{uuid.uuid4().hex[:8]}"
    await s.upsert_base_files(rid, FILES)
    return RepoChangeService(None, FakeVector(), s, llm_text=llm, llm_json=_no_json, emit=lambda *a: None), s, rid


def test_skipped_plan_file_gets_a_focused_retry(monkeypatch):
    monkeypatch.setattr(settings, "change_max_files", 6)

    async def go():
        llm = ScriptedLLM(block("a.py", "X = 1", "X = 2") + block("c.py", "Z = 1", "Z = 2"),     # skips b.js
                          block("b.js", "const y = 1;", "const y = 2;"))
        svc, s, rid = await _service(llm)
        r = await svc.run(rid, "add the field in a.py, b.js and c.py", plan=PLAN)
        assert r.ok and r.attempts == 2
        assert {f["path"] for f in r.files} == {"a.py", "b.js", "c.py"}
        first, second = llm.prompts
        assert "=== a.py ===" in first and "=== b.js ===" in first                 # full view on attempt 1
        assert "FOCUSED PASS" in second and "=== b.js ===" in second
        assert "=== a.py ===" not in second and "=== c.py ===" not in second       # only the missing file is shown ...
        assert "diff --git a/a.py b/a.py" in second and "+X = 2" in second         # ... plus what is already done
        assert "NO_CHANGE: <path>" in second and "EXISTING file(s) b.js" in second  # both ways out are offered
        assert len(second) < len(first) + 2000
    run(go())


def test_no_change_answer_survives_a_later_retry(monkeypatch):
    monkeypatch.setattr(settings, "change_max_files", 6)

    async def go():
        llm = ScriptedLLM(block("a.py", "X = 1", "X = 2") + "NO_CHANGE: b.js - already renders it\n",   # c.py skipped
                          block("c.py", "Z = 1", "Z = 2"))                      # b.js must not be demanded again
        svc, s, rid = await _service(llm)
        r = await svc.run(rid, "add the field in a.py, b.js and c.py", plan=PLAN)
        assert r.ok and r.attempts == 2
        assert {f["path"] for f in r.files} == {"a.py", "c.py"}
        assert "FOCUSED PASS" in llm.prompts[1] and "EXISTING file(s) c.py" in llm.prompts[1]
        assert "EXISTING file(s) b.js" not in llm.prompts[1]               # the NO_CHANGE answer was remembered
        assert "=== b.js ===" not in llm.prompts[1]
    run(go())


def test_chat_markdown_has_patch_link_and_clean_fence():
    async def go():
        llm = ScriptedLLM(block("a.py", "X = 1", "X = 2"))
        svc, s, rid = await _service(llm)
        r = await svc.run(rid, "change X in a.py")
        md = r.to_markdown()
        assert r.ok and f"/revisions/{r.revision}/patch" in md
        assert "\n\n```\n" not in md and r.diff.endswith("\n")
    run(go())
