"""Phase 1 tests: raw store, patch applier, static checks, router, change graph.
Run from the repo root:  pytest tests -q
No Ollama, Chroma or network needed (LLM and vector search are faked).
"""
import asyncio
import io
import uuid
import zipfile

import pytest

from config.settings import settings
from services import static_check
from services.intent_router import route_by_rules, route_intent
from services.patch_service import (apply_edits, diff_stats, make_diff, parse_edits,
                                    resolve_edit_paths)
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore, StaleRevision, safe_rel_path

run = asyncio.run


def block(path, search, replace):
    return f"FILE: {path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


# ----------------------------------------------------------------------------- patching
def test_parse_and_apply_exact():
    src = b"def a():\n    return 1\n\ndef b():\n    return 2\n"
    edits, errs = parse_edits(block("m.py", "def b():\n    return 2", "def b():\n    return 3"))
    assert not errs and len(edits) == 1
    res = apply_edits(edits, {"m.py": src})
    assert res.ok and res.new_files["m.py"] == b"def a():\n    return 1\n\ndef b():\n    return 3\n"
    assert res.outcomes[0].mode == "exact"


def test_crlf_preserved():
    src = b"x = 1\r\ny = 2\r\n"
    edits, _ = parse_edits(block("m.py", "y = 2", "y = 5"))
    res = apply_edits(edits, {"m.py": src})
    assert res.new_files["m.py"] == b"x = 1\r\ny = 5\r\n"


def test_substring_is_not_a_match():
    src = b"max = 10\n"
    edits, _ = parse_edits(block("m.py", "x = 1", "x = 2"))
    res = apply_edits(edits, {"m.py": src})
    assert not res.ok and "not found" in res.errors[0]


def test_ambiguous_search_rejected_and_file_untouched():
    src = b"a = 1\nb = 1\na = 1\n"
    edits, _ = parse_edits(block("m.py", "a = 1", "a = 9"))
    res = apply_edits(edits, {"m.py": src})
    assert not res.ok and "matches 2 places" in res.errors[0] and "m.py" not in res.new_files


def test_indent_insensitive_match_reindents():
    src = b"class A:\n    def f(self):\n        return 1\n"
    edits, _ = parse_edits(block("m.py", "def f(self):\n    return 1", "def f(self):\n    return 2"))
    res = apply_edits(edits, {"m.py": src})
    assert res.ok and res.outcomes[0].mode == "indent"
    assert res.new_files["m.py"] == b"class A:\n    def f(self):\n        return 2\n"


def test_failed_block_means_no_partial_file():
    src = b"a = 1\nb = 2\n"
    text = block("m.py", "a = 1", "a = 10") + block("m.py", "zzz", "q")
    edits, _ = parse_edits(text)
    res = apply_edits(edits, {"m.py": src})
    assert not res.ok and res.new_files == {}


def test_new_file_and_existing_file_guard():
    edits, _ = parse_edits("FILE: new.py\n<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n")
    assert apply_edits(edits, {"new.py": None}).new_files["new.py"] == b"print('hi')\n"
    res = apply_edits(edits, {"new.py": b"x\n"})
    assert not res.ok and "already exists" in res.errors[0]


def test_path_resolution_basename_and_unknown():
    known = ["services/a.py", "views/b.py"]
    edits, _ = parse_edits(block("wrong/dir/a.py", "x", "y"))
    fixed, notes, errs = resolve_edit_paths(edits, known)
    assert fixed[0].path == "services/a.py" and notes and not errs
    edits, _ = parse_edits(block("nope.py", "x", "y"))
    _, _, errs = resolve_edit_paths(edits, known)
    assert errs and "does not exist" in errs[0]


def test_windows_separators_are_normalised():
    assert safe_rel_path("services\\calc.py") == "services/calc.py"
    assert safe_rel_path(".\\a\\b.py") == "a/b.py"


def test_unsafe_and_secret_paths():
    for bad in ("../x.py", "/etc/passwd", "C:/x.py", "a/../../b"):
        with pytest.raises(ValueError):
            safe_rel_path(bad)
    edits, _ = parse_edits("FILE: prod.env\n<<<<<<< SEARCH\n=======\nA=1\n>>>>>>> REPLACE\n")
    _, _, errs = resolve_edit_paths(edits, ["x.py"])
    assert errs and "secret" in errs[0]


def test_diff_stats():
    d = make_diff("m.py", b"a\nb\n", b"a\nc\n")
    assert diff_stats(d) == (1, 1) and "--- a/m.py" in d


# ----------------------------------------------------------------------------- static checks
def test_static_checks():
    ok = static_check.check_files({"a.py": b"x = 1\n"})
    assert all(r.ok for r in ok)
    bad = static_check.check_files({"a.py": b"def f(:\n  pass\n"})
    assert any(not r.ok and r.check == "syntax" for r in bad)
    undef = static_check.check_files({"a.py": b"def f():\n    return nope\n"})
    assert any(not r.ok and r.check == "undefined-names" for r in undef)
    assert any(not r.ok for r in static_check.check_files({"a.json": b"{bad"}))
    assert any(not r.ok for r in static_check.check_files({"t.html": b"{% if x %}oops"}))
    assert static_check.check_files({"t.html": b"<p>plain</p>"})[0].skipped


def test_preexisting_failures_ignored():
    base = static_check.check_files({"a.py": b"def f():\n    return nope\n"})
    now = static_check.check_files({"a.py": b"def f():\n    # touched\n    return nope\n"})
    assert static_check.new_failures(base, now) == []


# ----------------------------------------------------------------------------- store
def _repo():
    return f"r_{uuid.uuid4().hex[:8]}"


def test_store_base_and_revisions_and_revert():
    async def go():
        s, rid = RepoStore(), _repo()
        out = await s.upsert_base_files(rid, [
            ("a.py", b"a=1\r\n"), ("sub/b.py", b"b=1\n"),
            (".env", b"K=v"), ("prod.env", b"K=v"), ("bin.dat", b"\x00\x01"), ("db_cookies.txt", b"x")])
        assert sorted(out["written"]) == ["a.py", "sub/b.py"]
        assert {p for p, _ in out["skipped"]} == {".env", "prod.env", "bin.dat", "db_cookies.txt"}
        assert (await s.get_file(rid, "a.py")).data == b"a=1\r\n"          # exact bytes
        assert (await s.upsert_base_files(rid, [("a.py", b"a=1\r\n")]))["unchanged"] == ["a.py"]

        r1 = await s.create_revision(rid, 0, "req", {"a.py": b"a=2\r\n", "new.py": b"n\n"}, {"k": 1})
        assert r1 == 1 and await s.latest_rev(rid) == 1
        assert (await s.get_file(rid, "a.py")).data == b"a=2\r\n"            # latest wins
        assert (await s.get_file(rid, "a.py", 0)).data == b"a=1\r\n"         # time travel
        assert await s.list_paths(rid, 0) == ["a.py", "sub/b.py"]
        assert await s.list_paths(rid) == ["a.py", "new.py", "sub/b.py"]
        assert (await s.changed_files(rid, 1))["a.py"] == (b"a=1\r\n", b"a=2\r\n")

        with pytest.raises(StaleRevision):
            await s.create_revision(rid, 0, "stale", {"a.py": b"x"}, {})

        # disk changes under a pending revision are flagged
        out = await s.upsert_base_files(rid, [("a.py", b"a=99\r\n")])
        assert out["shadowed"] == ["a.py"]

        r2 = await s.revert_to(rid, 0)
        assert r2 == 2
        assert (await s.get_file(rid, "a.py")).data == b"a=99\r\n"        # rev-0 view restored
        assert await s.get_file(rid, "new.py") is None                     # removed by revert
        assert await s.revert_to(rid, 2) == 2                              # no-op

        z = zipfile.ZipFile(io.BytesIO(await s.export_zip(rid, 1, extra={"changes.diff": b"d"})))
        assert set(z.namelist()) == {"a.py", "new.py", "changes.diff"}
    run(go())


def test_materialize_and_prune(tmp_path):
    async def go():
        s, rid = RepoStore(), _repo()
        await s.upsert_base_files(rid, [("p/a.py", b"1\r\n"), ("b.py", b"2\n")])
        assert await s.materialize(rid, None, tmp_path / "ws") == 2
        assert (tmp_path / "ws" / "p" / "a.py").read_bytes() == b"1\r\n"
        assert await s.prune_base(rid, ["b.py"]) == 1
        assert await s.list_paths(rid) == ["b.py"]
    run(go())


# ----------------------------------------------------------------------------- router
def test_rules_router():
    assert route_by_rules("fix the null pointer bug in login.py").intent == "change"
    assert route_by_rules("Can you add pagination to the repos endpoint").intent == "change"
    assert route_by_rules("how does the watcher work?").intent == "ask"
    assert route_by_rules("How do I fix this bug?").intent == "ask"
    assert route_by_rules("hmm engine").intent == "ask"


def test_router_prefix_model_and_fallback():
    async def go():
        async def boom(*a, **k):
            raise AssertionError("model must not be called for a prefix")
        d = await route_intent("/ask add pagination?", noul_fn=boom, model="m")
        assert (d.intent, d.query, d.source) == ("ask", "add pagination?", "prefix")

        async def likely_change(*a, **k):
            return 0.9                                    # P(wants change)
        d = await route_intent("what is this", noul_fn=likely_change, model="m")
        assert (d.intent, d.source) == ("change", "model")

        async def likely_ask(*a, **k):
            return 0.1
        d = await route_intent("fix the bug in a.py", noul_fn=likely_ask, model="m")
        assert (d.intent, d.source) == ("ask", "model") and d.confidence == 0.9

        async def unsure(*a, **k):
            return 0.55                                   # confidence 0.55 < 0.6 -> rules decide
        d = await route_intent("what is this", noul_fn=unsure, model="m")
        assert (d.intent, d.source) == ("ask", "rules")

        async def none(*a, **k):
            return None
        d = await route_intent("fix the bug in a.py", noul_fn=none, model="m")
        assert (d.intent, d.source) == ("change", "rules")
    run(go())


# ----------------------------------------------------------------------------- change graph
class FakeDoc:
    def __init__(self, p):
        self.metadata = {"file_path": p}


class FakeVector:
    def __init__(self, paths):
        self.paths = paths

    def similarity_search(self, repo_id, q, k=4):
        return [FakeDoc(p) for p in self.paths]


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def __call__(self, model, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


async def _no_json(*a, **k):
    return None


async def _service(llm, files, semantic=(), llm_json=None):
    s, rid = RepoStore(), _repo()
    await s.upsert_base_files(rid, files)
    svc = RepoChangeService(None, FakeVector(list(semantic)), s, llm_text=llm,
                            llm_json=llm_json or _no_json, emit=lambda *a: None)
    return svc, s, rid


def test_change_happy_path_creates_revision_not_disk_write():
    async def go():
        llm = ScriptedLLM(block("calc.py", "    return a - b", "    return a + b"))
        svc, s, rid = await _service(llm, [("calc.py", b"def add(a, b):\n    return a - b\n")])
        r = await svc.run(rid, "fix the bug in calc.py add function")
        assert r.ok and r.revision == 1 and r.attempts == 1
        assert (await s.get_file(rid, "calc.py")).data == b"def add(a, b):\n    return a + b\n"
        assert (await s.get_file(rid, "calc.py", 0)).data.endswith(b"a - b\n")
        assert "revision 1" in r.to_markdown() and "```diff" in r.to_markdown()
        assert "Tests not run" in r.to_markdown()
    run(go())


def test_change_retries_after_bad_search_then_succeeds():
    async def go():
        llm = ScriptedLLM(block("calc.py", "return a * b  # wrong", "return a + b"),
                          block("calc.py", "    return a - b", "    return a + b"))
        svc, s, rid = await _service(llm, [("calc.py", b"def add(a, b):\n    return a - b\n")])
        r = await svc.run(rid, "fix calc.py")
        assert r.ok and r.attempts == 2
        assert "SEARCH text not found" in llm.prompts[1]          # feedback reached the model
    run(go())


def test_change_retries_after_syntax_error_and_keeps_applied_state():
    async def go():
        llm = ScriptedLLM(block("calc.py", "    return a - b", "    return a +"),
                          block("calc.py", "    return a +", "    return a + b"))
        svc, s, rid = await _service(llm, [("calc.py", b"def add(a, b):\n    return a - b\n")])
        r = await svc.run(rid, "fix calc.py")
        assert r.ok and r.attempts == 2
        assert "syntax" in llm.prompts[1] and "return a +" in llm.prompts[1]   # model sees CURRENT text
    run(go())


def test_change_fails_cleanly_after_max_attempts_and_saves_nothing():
    async def go():
        bad = block("calc.py", "nothing like this", "x")
        llm = ScriptedLLM(*[bad] * settings.change_max_attempts)
        svc, s, rid = await _service(llm, [("calc.py", b"def add(a, b):\n    return a - b\n")])
        r = await svc.run(rid, "fix calc.py")
        assert not r.ok and r.attempts == settings.change_max_attempts
        assert await s.latest_rev(rid) == 0
        assert "could not produce a safe change" in r.to_markdown()
    run(go())


def test_change_cannot_complete_and_no_files_and_new_file():
    async def go():
        svc, s, rid = await _service(ScriptedLLM("CANNOT_COMPLETE: needs db schema"),
                                     [("calc.py", b"x = 1\n")])
        r = await svc.run(rid, "fix calc.py")
        assert not r.ok and "needs db schema" in r.errors[0]

        svc, s, rid = await _service(ScriptedLLM("x"), [("calc.py", b"x = 1\n")])
        r = await svc.run(rid, "make everything better")          # names no file, no semantic hits
        assert not r.ok and "Name the file" in r.errors[0]

        new = "FILE: util.py\n<<<<<<< SEARCH\n=======\nVALUE = 1\n>>>>>>> REPLACE\n"
        svc, s, rid = await _service(ScriptedLLM(new), [("calc.py", b"x = 1\n")], semantic=["calc.py"])
        r = await svc.run(rid, "add a util module")
        assert r.ok and (await s.get_file(rid, "util.py")).data == b"VALUE = 1\n"
        assert r.files[0]["new"]
    run(go())


def test_followup_builds_on_latest_revision_and_stale_base_rejected():
    async def go():
        llm = ScriptedLLM(block("a.py", "x = 1", "x = 2"), block("a.py", "x = 2", "x = 3"))
        svc, s, rid = await _service(llm, [("a.py", b"x = 1\n")])
        assert (await svc.run(rid, "change a.py")).revision == 1
        r2 = await svc.run(rid, "change a.py again")
        assert r2.ok and r2.revision == 2 and r2.base_rev == 1
        assert (await s.get_file(rid, "a.py")).data == b"x = 3\n"
        stale = await svc.run(rid, "x", base_rev=0)
        assert not stale.ok and "not the latest" in stale.errors[0]
    run(go())


def test_preexisting_problem_in_edited_file_does_not_block_unrelated_edit():
    async def go():
        base = b"def f():\n    return nope\n\nY = 1\n"          # already has an undefined name
        llm = ScriptedLLM(block("old.py", "Y = 1", "Y = 2"))
        svc, s, rid = await _service(llm, [("old.py", base)])
        r = await svc.run(rid, "change Y in old.py")
        assert r.ok and any("already existed" in n for n in r.notes)

        # but a NEW undefined name introduced by the edit is still caught
        llm = ScriptedLLM(block("old.py", "Y = 1", "Y = other_missing"),
                          block("old.py", "Y = other_missing", "Y = 2"))
        svc, s, rid = await _service(llm, [("old.py", base)])
        r = await svc.run(rid, "change Y in old.py")
        assert r.ok and r.attempts == 2
        assert "other_missing" in llm.prompts[1]
    run(go())


def test_incomplete_raw_store_gives_actionable_error_and_trace():
    async def go():
        # search finds app.js, but only calc.py was ever stored (ingest not re-run)
        svc, s, rid = await _service(ScriptedLLM("x"), [("calc.py", b"x = 1\n")], semantic=["static\\js\\app.js"])
        r = await svc.run(rid, "make code blocks gray")
        assert not r.ok and "Re-run ingest" in r.errors[0] and "static/js/app.js" in r.errors[0]
        assert r.trace and "not in raw store" in r.trace[1]
        assert "**Details:**" in r.to_markdown()
    run(go())


def test_file_tree_pick_rescues_vague_request():
    async def go():
        async def pick(model, prompt, **kw):
            assert "static/js/app.js" in prompt
            return {"files": ["static/js/app.js", "not/a/file.js"]}
        llm = ScriptedLLM(block("static/js/app.js", "const c = 'green';", "const c = 'gray';"))
        svc, s, rid = await _service(llm, [("static/js/app.js", b"const c = 'green';\n"),
                                           ("calc.py", b"x = 1\n")], llm_json=pick)
        r = await svc.run(rid, "make code blocks gray")           # names no file, search finds nothing
        assert r.ok and r.files[0]["path"] == "static/js/app.js"
        assert any("file-tree pick: 1" in t for t in r.trace)
    run(go())
