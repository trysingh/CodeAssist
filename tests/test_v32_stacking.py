"""v3.2: stacked revisions are visible (cumulative line), repeated requests are not regenerated, plain summary on top.
Pure parts run directly; the repo_change / controller call sites are EXECUTED against fakes (tests/stubbed_imports.py) when their
third-party packages are missing."""
import asyncio
import hashlib
import types

import pytest

from stubbed_imports import stubbed_imports
from services.reasoning import IntakeAgent, VerifierAgent
from services.reasoning.crosscheck import facts_from_sandbox
from services.reasoning.stacking import (change_summary, find_repeat, is_repeat_request, normalize_request, plan_file_lines,
                                         repeat_reply, request_similarity, stack_note, wants_redo)

REQUEST = ("give a more modern look to the landing page with color schemes for red-green color blindness friendly users. "
           "Add audible effects when starting and stopping services and ensure it is responsive")
PATH = "templates/dashboard.html"
V0 = "\n".join(f"line {i}" for i in range(40)) + "\n"
V22 = (V0.replace("line 3\n", "line 3 new colour\nextra a\nextra b\n").replace("line 9\n", "line 9 audio\nextra c\n").replace("line 20\n", "")
       + "".join(f"added {i}\n" for i in range(10)))
V23 = V22.replace("line 3 new colour", "line 3 new colour tweaked")


def run(coro):
    return asyncio.run(coro)


class FakeStore:
    """Just enough of RepoStore: revisions = {rev: (request, report, {path: bytes})}; views are cumulative."""
    def __init__(self, revs, base=None):
        self.base = base or {PATH: V0.encode()}
        self.revs = revs

    def view(self, rev):
        out = dict(self.base)
        for r in sorted(self.revs):
            if r <= rev:
                out.update(self.revs[r][2])
        return out

    async def latest_rev(self, repo_id):
        return max(self.revs, default=0)

    async def list_revisions(self, repo_id):
        return [{"rev": r, "parent": r - 1, "request": self.revs[r][0], "created_at": "", "files": len(self.revs[r][2])}
                for r in sorted(self.revs, reverse=True)]

    async def get_revision(self, repo_id, rev):
        if rev not in self.revs:
            return None
        return {"rev": rev, "parent": rev - 1, "request": self.revs[rev][0], "report": self.revs[rev][1]}

    async def index(self, repo_id, rev=None):
        return {p: hashlib.sha256(b).hexdigest() for p, b in self.view(rev if rev is not None else 10 ** 6).items()}

    async def get_file(self, repo_id, path, rev=None):
        b = self.view(rev if rev is not None else 10 ** 6).get(path)
        return types.SimpleNamespace(data=b) if b is not None else None

    async def changed_files(self, repo_id, rev):
        out = {}
        for p, b in self.revs[rev][2].items():
            prev = self.view(rev - 1).get(p)
            out[p] = (prev, b)
        return out

    async def list_paths(self, repo_id, rev=None):
        return list(self.view(rev or 0))


def edit_rev(request, data, applied=False, kind="edit"):
    rep = {"kind": kind, "files": [PATH]}
    if applied:
        rep["applied_at"] = "2026-11-10T00:00:00Z"
    return (request, rep, {PATH: data.encode()})


def rev23_store(**kw):
    return FakeStore({22: edit_rev(REQUEST, V22, **kw), 23: edit_rev(REQUEST, V23)})


@pytest.fixture(scope="module")
def mods():
    """repo_change + controller, imported for real when their packages exist, otherwise against inert stand-ins."""
    with stubbed_imports():
        import importlib
        yield types.SimpleNamespace(rc=importlib.import_module("services.repo_change"),
                                    ec=importlib.import_module("controllers.engine_controller"),
                                    settings=importlib.import_module("config.settings").settings)


# ---------------------------------------------------------------- 1. stacked revisions
def test_stack_note_names_the_stack_and_the_cumulative_change():
    cum = [{"path": PATH, "added": 31, "removed": 5, "new": False}]
    note = stack_note([22], 23, cum)
    assert "Builds on unapplied revision(s) 22." in note
    assert "Shown below: this revision's changes only." in note
    assert f"`{PATH}` (+31 −5)" in note and "cumulative change against your disk" in note
    assert "Download .patch (cumulative)" in note and "apply 22 and 23 together" in note
    assert stack_note([], 23, cum) == ""
    assert "apply 21, 22 and 23 together" in stack_note([21, 22], 23, cum)


def test_unapplied_chain_cumulative_stats_and_stack_info(mods):
    from services.reasoning.stacking import cumulative_stats, stack_info, unapplied_edit_revisions
    st = rev23_store()
    assert run(unapplied_edit_revisions(st, "r", 22)) == [22]
    cum = run(cumulative_stats(st, "r", 23))
    assert [f["path"] for f in cum] == [PATH] and cum[0]["added"] > 6 and not cum[0]["new"]
    chain, cum2 = run(stack_info(st, "r", 22, 23))
    assert chain == [22] and cum2 == cum
    # 22 applied: 23 builds on your disk, nothing to say
    assert run(stack_info(rev23_store(applied=True), "r", 22, 23)) == ([], [])
    # built on rev 0 / nothing unapplied
    assert run(stack_info(st, "r", 0, 1)) == ([], [])
    # disk-sync and revert revisions are skipped, an applied one stops the walk
    st2 = FakeStore({20: edit_rev("old", V0, applied=True), 21: edit_rev("sync", V0, kind="disk-sync"), 22: edit_rev(REQUEST, V22),
                     23: edit_rev(REQUEST, V23)})
    assert run(unapplied_edit_revisions(st2, "r", 22)) == [22]


def sample_result(mods, **kw):
    rc = mods.rc
    diff = "diff --git a/templates/dashboard.html b/templates/dashboard.html\n--- a/templates/dashboard.html\n+++ b/templates/dashboard.html\n@@ -1 +1 @@\n-a\n+b\n"
    return rc.ChangeResult(ok=True, repo_id="r", request=REQUEST, revision=23, base_rev=22, attempts=2,
                           files=[{"path": PATH, "added": 6, "removed": 6, "new": False}], diff=diff, plan="FILES:\n- x", checked=1,
                           verification="L1-boot-smoke",
                           tests={"ran": True, "ok": True, "mode": "boot", "passed": 9, "backend": "docker", "on_unapplied_n": 1,
                                  "on_unapplied": [PATH]}, **kw)


def test_change_result_renders_the_stack_line_once_and_not_a_second_diff(mods):
    cum = [{"path": PATH, "added": 31, "removed": 5, "new": False}]
    md = sample_result(mods, stacked_on=[22], cumulative=cum).to_markdown()
    assert "**Builds on unapplied revision(s) 22.**" in md and f"`{PATH}` (+31 −5)" in md
    assert md.count("```diff") == 1                                    # this revision's delta only, never both diffs
    assert "(+6 −6)" in md and "Verified on top of 1 file(s) from earlier proposals" in md   # the existing sandbox message stays
    assert md.index("Builds on unapplied") < md.index("**Plan:**") < md.index("```diff")
    plain = sample_result(mods).to_markdown()
    assert "Builds on unapplied" not in plain
    assert plain == sample_result(mods, stacked_on=[], cumulative=[]).to_markdown()
    failed = sample_result(mods, stacked_on=[22], cumulative=cum)
    failed.ok = False
    assert "Builds on unapplied" not in failed.to_markdown()


def _service(mods, store, out, base=22):
    svc = object.__new__(mods.rc.RepoChangeService)
    svc.store, svc._emit_fn = store, None
    svc._open_log = lambda *a, **k: None
    svc.graph = types.SimpleNamespace(ainvoke=None)

    async def ainvoke(init, config=None):
        store.revs[23] = edit_rev(REQUEST, V23)       # the graph's save step creates the revision before returning
        return out
    svc.graph.ainvoke = ainvoke
    return svc


def test_run_fills_the_stack_for_a_revision_built_on_an_unapplied_one(mods):
    st = FakeStore({22: edit_rev(REQUEST, V22)})
    out = {"status": "ok", "revision": 23, "attempt": 2, "working": {PATH: V23.encode()}, "plan": "FILES:\n- x",
           "tests": {"ran": True, "ok": True, "mode": "boot", "passed": 9, "backend": "docker"}}
    svc = _service(mods, st, out)
    res = run(svc._run("r", REQUEST, 22, "", ""))
    assert res.ok and res.stacked_on == [22] and res.cumulative and res.cumulative[0]["path"] == PATH
    assert res.files and (res.files[0]["added"], res.files[0]["removed"]) == (1, 1)         # the 23-vs-22 delta stays the shown diff
    assert "Builds on unapplied revision(s) 22" in res.to_markdown()


def test_run_survives_a_failing_stack_lookup(mods):
    st = FakeStore({22: edit_rev(REQUEST, V22)})

    async def boom(*a, **k):
        raise RuntimeError("store down")
    st.list_revisions = boom
    out = {"status": "ok", "revision": 23, "attempt": 1, "working": {PATH: V23.encode()}, "tests": {}}
    res = run(_service(mods, st, out)._run("r", REQUEST, 22, "", ""))
    assert res.ok and res.stacked_on == [] and "Builds on unapplied" not in res.to_markdown()


def test_prompts_carry_the_no_churn_rules(mods):
    assert "already satisfied by the current code, make no edits" in mods.rc._PROMPT
    assert "never change unrelated values (timings, shadows, spacing)" in mods.rc._PROMPT
    assert "timings, shadows or spacing" in mods.rc._REVIEW_PROMPT


# ---------------------------------------------------------------- 2. repeated request
def test_similarity_and_override_words():
    assert normalize_request("  Give a MORE modern look!  ") == "give a more modern look"
    assert is_repeat_request(REQUEST, REQUEST.upper() + " ")
    assert is_repeat_request(REQUEST, REQUEST + ".")
    assert is_repeat_request(REQUEST, REQUEST.replace("responsive", "responsive!!"))
    assert not is_repeat_request(REQUEST, "add retries to the client")
    assert request_similarity("", "x") == 0.0
    for w in ("redo", "Again", "regenerate", "from scratch"):
        assert wants_redo(f"{REQUEST} {w}") and wants_redo(f"{w.upper()}: {REQUEST}")
    assert not wants_redo(REQUEST) and not wants_redo("the redoubt")


def test_find_repeat_hit_miss_and_edge_cases():
    st = rev23_store()
    hit = run(find_repeat(st, "r", REQUEST))
    assert hit["rev"] == 23 and hit["files"][0]["path"] == PATH and hit["files"][0]["added"] == 1
    assert run(find_repeat(st, "r", REQUEST.lower() + "."))["rev"] == 23                          # near-identical
    assert run(find_repeat(st, "r", "add retries to client.py")) is None                          # different request
    for w in ("redo", "again", "regenerate", "from scratch"):
        assert run(find_repeat(st, "r", f"{REQUEST}, {w}")) is None                                # override word
    assert run(find_repeat(st, "r", "")) is None
    # the latest unapplied revision was made for another request: not a repeat
    st2 = FakeStore({22: edit_rev(REQUEST, V22), 23: edit_rev("make the header sticky", V23)})
    assert run(find_repeat(st2, "r", REQUEST)) is None
    # latest edit already applied: nothing pending to repeat
    assert run(find_repeat(FakeStore({22: edit_rev(REQUEST, V22, applied=True)}), "r", REQUEST)) is None
    # a disk-sync revision on top is skipped
    st3 = FakeStore({22: edit_rev(REQUEST, V22), 23: edit_rev("disk sync", V23, kind="disk-sync")})
    assert run(find_repeat(st3, "r", REQUEST))["rev"] == 22
    assert run(find_repeat(FakeStore({}), "r", REQUEST)) is None


def test_missing_stored_request_proceeds_normally():
    for missing in ("", None, "   "):
        st = FakeStore({22: edit_rev(missing, V22)})
        assert run(find_repeat(st, "r", REQUEST)) is None
    st = FakeStore({22: ("x", {"kind": "edit"}, {PATH: V22.encode()})})
    st.get_revision = lambda *a, **k: asyncio.sleep(0, result=None)                              # revision record not found
    assert run(find_repeat(st, "r", REQUEST)) is None


def test_repeat_reply_text_and_links():
    hit = run(find_repeat(rev23_store(), "ff46", REQUEST))
    text = repeat_reply(hit, "ff46")
    assert text.startswith("This is the same request as revision 23 (not applied yet), which already implements it: ")
    assert f"`{PATH}` (+1 −1)" in text
    assert "Apply it, or tell me what to change. Say 'redo' to regenerate from scratch." in text
    for part in ("[Apply to repo](/api/v1/repos/ff46/revisions/23/apply)", "[Download changed files](/api/v1/repos/ff46/revisions/23/download)",
                 "[Full diff](/api/v1/repos/ff46/revisions/23/diff)", "[Download .patch](/api/v1/repos/ff46/revisions/23/patch)"):
        assert part in text


# ---------------------------------------------------------------- controller: executed end to end against fakes
def make_controller(mods, store, monkeypatch, guard=True, tests=None):
    ec, rc = mods.ec, mods.rc
    monkeypatch.setattr(mods.settings, "repeat_request_guard", guard)
    monkeypatch.setattr(mods.settings, "reasoning_supervisor", True)
    ctl = object.__new__(ec.EngineController)
    calls = {"run": 0, "model": 0}

    class DB:
        async def init_db(self): pass
        async def get_repo(self, rid): return types.SimpleNamespace(id=rid, root_path="/tmp/none")
        async def get_thread_messages(self, tid): return []
    ctl.db, ctl.repo_store = DB(), store

    async def warm(rid): pass
    ctl._warm_agentic = warm

    class Change:
        async def run(self, repo_id, request, base_rev, context="", plan=""):
            calls["run"] += 1
            rev = (await store.latest_rev(repo_id)) + 1
            data = V23.replace("tweaked", f"gen{rev}")
            store.revs[rev] = edit_rev(request, data)
            res = rc.ChangeResult(ok=True, repo_id=repo_id, request=request, revision=rev, base_rev=base_rev or rev - 1, attempts=1,
                                  files=[{"path": PATH, "added": 3, "removed": 1, "new": False}], diff="diff --git a/x b/x\n",
                                  plan=f"FILES:\n- {PATH} - restyle the palette and add audio effects; make it responsive\n\nNEW FILES:\n- none",
                                  verification="L1-boot-smoke", checked=1,
                                  tests=tests if tests is not None else {"ran": True, "ok": True, "mode": "boot", "passed": 9, "backend": "docker"})
            res.stacked_on, res.cumulative = await __import__("services.reasoning.stacking", fromlist=["x"]).stack_info(store, repo_id, res.base_rev, rev)
            return res
    ctl.change_service = Change()
    ctl.supervisor = ctl._build_supervisor()
    ctl.rag_manager = types.SimpleNamespace(invoke_chat=None)
    return ctl, calls


def chat(ctl, q):
    return run(ctl._chat_impl("repo1", "t1", q, "off", None, "", None))


def test_controller_repeated_request_does_not_call_the_change_service(mods, monkeypatch):
    st = rev23_store()
    ctl, calls = make_controller(mods, st, monkeypatch)
    out = chat(ctl, "/change " + REQUEST)
    assert calls["run"] == 0
    assert out.startswith("This is the same request as revision 23 (not applied yet)")
    assert "Delivery notes" not in out and "**Summary**" not in out and "Verified" not in out
    assert "[Apply to repo](/api/v1/repos/repo1/revisions/23/apply)" in out
    assert max(st.revs) == 23                                                                       # no revision was created


@pytest.mark.parametrize("word", ["redo", "again", "regenerate", "from scratch"])
def test_controller_override_words_regenerate(mods, monkeypatch, word):
    st = rev23_store()
    ctl, calls = make_controller(mods, st, monkeypatch)
    out = chat(ctl, f"/change {REQUEST} - {word}")
    assert calls["run"] == 1 and "same request as revision" not in out and max(st.revs) == 24


def test_controller_guard_off_and_different_request_build_normally(mods, monkeypatch):
    ctl, calls = make_controller(mods, rev23_store(), monkeypatch, guard=False)
    chat(ctl, "/change " + REQUEST)
    assert calls["run"] == 1
    ctl, calls = make_controller(mods, rev23_store(), monkeypatch)
    chat(ctl, "/change make the page header sticky when scrolling")
    assert calls["run"] == 1


def test_controller_missing_stored_request_builds_normally(mods, monkeypatch):
    st = FakeStore({22: edit_rev(None, V22)})
    ctl, calls = make_controller(mods, st, monkeypatch)
    chat(ctl, "/change " + REQUEST)
    assert calls["run"] == 1


def test_controller_guard_error_falls_back_to_building(mods, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("db locked")
    monkeypatch.setattr(mods.ec, "find_repeat", boom)
    ctl, calls = make_controller(mods, rev23_store(), monkeypatch)
    out = chat(ctl, "/change " + REQUEST)
    assert calls["run"] == 1 and "Change proposal saved" in out


def test_rev23_regression_summary_stack_line_and_honest_notes(mods, monkeypatch):
    """The reply the user got for revision 23, rebuilt: stacked on unapplied 22, boot check only."""
    st = FakeStore({22: edit_rev(REQUEST, V22)})
    ctl, calls = make_controller(mods, st, monkeypatch, guard=False)
    out = chat(ctl, "/change " + REQUEST)
    assert calls["run"] == 1
    assert out.startswith("**Summary**")
    head, _, rest = out.partition("**Change proposal saved")
    lines = [l for l in head.strip().splitlines() if l.strip()]
    assert len(lines) <= 6
    assert any(l.startswith("- Changed:") and PATH in l and "+3 −1" in l for l in lines)
    assert any(l.startswith("- Apply:") and "builds on unapplied 22" in l and "cumulative .patch" in l and "apply 22 and 23 together" in l for l in lines)
    assert any(l.startswith("- Verified:") and "boot check only" in l and "Achieved: L1 (boot smoke)" in l for l in lines)
    assert any(l.startswith("- Next:") and "add a test" in l and "L3 isn't available in this version" in l for l in lines)
    assert "**Builds on unapplied revision(s) 22.**" in rest and rest.count("```diff") <= 1
    notes = rest.split("### Delivery notes", 1)[1]
    nv = notes.split("**Not verified**", 1)[1].split("**Requirements", 1)[0] if "**Not verified**" in notes else ""
    assert "boot check passed" not in nv and "Runtime behaviour beyond" not in nv
    assert "boot check only (no tests cover this change)" in notes
    assert notes.count("only the boot check ran") + notes.count("Runtime behaviour beyond the boot check") == 1
    assert "tests passed" not in notes


# ---------------------------------------------------------------- 3. summary (pure)
def _notes_and_facts(request=REQUEST, tests=None, **extra):
    brief = IntakeAgent().analyze(request, "change")
    facts = {**facts_from_sandbox(tests if tests is not None else {"ran": True, "ok": True, "mode": "boot", "passed": 9, "backend": "docker"}),
             "achieved_level": "L1-boot-smoke", "static_checked": True, "files_changed": [PATH], "revision": 23,
             "plan": f"FEASIBLE: yes\n\nFILES:\n- {PATH} - restyle palette, add audio effects, make responsive\n\nNEW FILES:\n- none\n\nRISKS:\n- x",
             "change_files": [{"path": PATH, "added": 61, "removed": 20, "new": False}], **extra}
    return brief, facts, VerifierAgent().check(brief, "saved", facts)


def test_plan_file_lines_parse():
    p = plan_file_lines("FEASIBLE: yes\nFILES:\n- a/b.py - change x\n- `c.html` – other\nNEW FILES:\n- n.js - new\nRISKS:\n- r.py - no\n")
    assert p == {"a/b.py": "change x", "c.html": "other", "n.js": "new"}
    assert plan_file_lines("") == {}


def test_summary_is_short_deterministic_and_complete():
    brief, facts, notes = _notes_and_facts()
    s1, s2 = change_summary(brief, facts, notes), change_summary(brief, facts, notes)
    assert s1 == s2 and s1.startswith("**Summary**\n")
    lines = [l for l in s1.strip().splitlines() if l.strip()]
    assert 4 <= len(lines) <= 6
    assert f"`{PATH}` (+61 −20)" in lines[1] and lines[1].startswith("- Changed:")
    assert lines[2] == "- Apply: use the Apply to repo link below (revision 23), or the .patch."
    assert lines[3].startswith("- Verified: Verified – executed in container: boot check only") and "Achieved: L1 (boot smoke)" in lines[3]
    assert lines[4] == "- Next: add a test that fails before and passes after; check the UI in a browser (L3 isn't available in this version)"


def test_summary_stacked_and_tested_variants():
    brief, facts, notes = _notes_and_facts(stacked_on=[22])
    assert "revision 23 builds on unapplied 22: use the cumulative .patch or apply 22 and 23 together." in change_summary(brief, facts, notes)
    t = {"ran": True, "ok": True, "mode": "pytest", "passed": 4, "failed": 0, "backend": "docker", "proof": {"proves": 1}}
    brief, facts, notes = _notes_and_facts(request="add retries to client.py", tests=t)
    facts["achieved_level"] = "L3-fail-to-pass"
    s = change_summary(brief, facts, VerifierAgent().check(brief, "saved", facts))
    assert "4 sandbox tests passed" in s and "add a test" not in s and "- Next: review the diff, then apply" in s


def test_summary_only_for_a_saved_change():
    brief, facts, notes = _notes_and_facts()
    assert change_summary(brief, {**facts, "revision": None}, notes) == ""
    ask = IntakeAgent().analyze("how does the dashboard work?", "ask")
    assert change_summary(ask, facts, notes) == ""
    assert change_summary(brief, {}, notes) == ""


def test_supervisor_without_facts_or_summary_keeps_working():
    from services.reasoning import ReasoningSupervisor

    async def execute(b, p):
        return "ANSWER", {}
    out = run(ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {}).run(
        query="add retries to client.py", intent="change", repo_files=[], execute=execute))
    assert out.startswith("ANSWER") and "**Summary**" not in out and "### Delivery notes" in out


def test_supervisor_skip_notes_returns_the_bare_answer():
    from services.reasoning import ReasoningSupervisor

    async def execute(b, p):
        return "SAME REQUEST", {"skip_notes": True}
    out = run(ReasoningSupervisor(IntakeAgent(), VerifierAgent(), lambda: {}).run(
        query="add retries to client.py", intent="change", repo_files=[], execute=execute))
    assert out == "SAME REQUEST"
