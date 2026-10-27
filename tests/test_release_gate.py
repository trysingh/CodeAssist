"""Release gate + app boot check.  Run: pytest tests/test_release_gate.py -q
No Docker, Ollama or network: the sandbox runs through LocalRunner (needs fastapi + httpx + jinja2 on the host).
"""
import pytest

from config.settings import settings
from services import sandbox_service as sbx
from services.boot_check import BOOT_PY, find_entries
from services.gates import release_gate
from services.repo_change import RepoChangeService
from services.repo_store import RepoStore
from test_phase2 import Script, _change_svc, block, rid, run

MAIN = b"from fastapi import FastAPI\nfrom controllers.home import router\napp = FastAPI()\napp.include_router(router)\n"
HOME = (b"from fastapi import APIRouter, Request\nfrom fastapi.templating import Jinja2Templates\nfrom services.items import load\n"
        b"router = APIRouter()\ntemplates = Jinja2Templates(directory='templates')\n\n\n@router.get('/')\n"
        b"def index(request: Request):\n    return templates.TemplateResponse(request, 'index.html', {'items': load(), 'user': {'name': 'x'}})\n")
ITEMS = b"def load():\n    return [{'name': 'a', 'qty': 2}]\n"
PAGE = b"<h1>Hello {{ user.name }}</h1>\n"
APP = [("main.py", MAIN), ("controllers/home.py", HOME), ("services/items.py", ITEMS), ("templates/index.html", PAGE)]
HOME_LINE = "    return templates.TemplateResponse(request, 'index.html', {'items': load(), 'user': {'name': 'x'}})"
HOME_BAD = "    return templates.TemplateResponse(request, 'index.html', {'items': load()})"
HOME_FIX = "    return templates.TemplateResponse(request, 'index.html', {'items': load(), 'user': {'name': 'y'}})"


async def _svc(files):
    s, r = RepoStore(), rid()
    await s.upsert_base_files(r, files)
    return sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None), s, r


# ----------------------------------------------------------------------------- detection and the pure gate
def test_find_entries_finds_app_and_factory_and_skips_tests():
    t = {"main.py": MAIN, "svc/api.py": b"from fastapi import FastAPI\ndef create_app():\n    return FastAPI()\n",
         "tests/test_x.py": b"from fastapi import FastAPI\napp = FastAPI()\n", "notes.py": b"x = 1\n"}
    assert find_entries(t) == ["main:app", "svc.api:create_app()"]


def test_boot_script_runs_without_unix_signals():
    # LocalRunner runs it on the host, which may be Windows: no SIGALRM / signal.alarm there.
    compile(BOOT_PY, "<boot>", "exec")
    assert not any(w in BOOT_PY for w in ("SIGALRM", "signal.alarm", "import signal", ", signal,"))


GREEN = {"ran": True, "ok": True, "mode": "pytest", "passed": 3, "proof": {"proves": 1},
         "boot": {"entries": ["main:app"], "ran": True, "ok": True}}


def test_gate_matrix():
    f = ["services/items.py"]
    assert release_gate({}, f, "off") == []
    assert release_gate(GREEN, f, "strict") == [] and release_gate(GREEN, f, "smoke") == []
    assert "did not verify" in release_gate({"ran": False, "note": "Docker CLI not found"}, f, "smoke")[0]
    assert "did not verify" in release_gate({**GREEN, "ok": False, "inconclusive": True}, f, "smoke")[0]
    assert "boot" in release_gate({**GREEN, "boot": {"entries": ["main:app"], "ran": False, "why": "no httpx"}}, f, "smoke")[0]
    assert "boot" in release_gate({**GREEN, "boot": {"entries": ["main:app"], "ran": True, "ok": False}}, f, "smoke")[0]
    no_proof = {**GREEN, "proof": {}}
    assert release_gate(no_proof, f, "smoke") == []                                   # smoke does not ask for a proving test
    assert "fails on the old code" in release_gate(no_proof, f, "strict")[0]
    assert release_gate(no_proof, ["templates/index.html"], "strict") == []           # no python changed: boot is the evidence
    assert release_gate({"ran": False}, ["README.md", "static/css/a.css"], "strict") == []   # inert files cannot 500
    assert "dropped" in release_gate({**GREEN, "generated_dropped": [{"id": "x"}]}, f, "strict")[0]


# ----------------------------------------------------------------------------- boot check through the sandbox
FAULTS = [
    ("services/items.py", ITEMS.replace(b"def load()", b"def load_items()"), "cannot import name 'load'"),
    ("controllers/home.py", HOME.replace(b", 'user': {'name': 'x'}", b""), "'user' is undefined"),
    ("main.py", MAIN + b"\n@app.on_event('startup')\ndef _i():\n    import os\n    os.environ['NEEDED_KEY']\n", "NEEDED_KEY"),
    ("templates/index.html", b"<h1>{{ user.name|currency }}</h1>\n", "No filter named 'currency'"),
]


@pytest.mark.parametrize("path,broken,needle", FAULTS)
def test_boot_check_catches_what_import_smoke_cannot(path, broken, needle):
    async def go():
        svc, s, r = await _svc(APP)
        healthy = {"services/items.py": ITEMS + b"# t\n", "controllers/home.py": HOME + b"# t\n",
                   "main.py": MAIN + b"# t\n", "templates/index.html": PAGE + b"<!-- t -->\n"}[path]
        v = await svc.verify(r, 0, {path: healthy})
        assert v.ok and v.boot["ran"] and v.boot["ok"], v.boot
        v = await svc.verify(r, 0, {path: broken})
        assert not v.ok and v.boot["ok"] is False
        assert any(needle in c.message and c.id.startswith("boot::") for c in v.new_failures), [c.message for c in v.new_failures]
    run(go())


def test_template_only_change_gets_a_boot_verdict_and_its_own_level():
    async def go():
        svc, s, r = await _svc(APP)
        v = await svc.verify(r, 0, {"templates/index.html": PAGE + b"<p>more</p>\n"})
        assert v.ran and v.ok and v.report.mode == "boot"          # before: "no tests and nothing importable changed"
        assert RepoChangeService._verification(v.to_dict()) == "L1-boot-smoke"
    run(go())


def test_failures_that_already_exist_on_the_base_are_not_blamed_on_the_change():
    async def go():
        broken_base = HOME + b"\n\n@router.get('/boom')\ndef boom():\n    raise RuntimeError('db is down')\n"
        svc, s, r = await _svc([("main.py", MAIN), ("controllers/home.py", broken_base), ("services/items.py", ITEMS),
                                ("templates/index.html", PAGE)])
        v = await svc.verify(r, 0, {"services/items.py": ITEMS + b"# t\n"})
        assert v.ok and v.boot["ok"] and v.boot["preexisting"] == 1
    run(go())


def test_an_app_that_cannot_boot_even_before_the_change_is_not_called_verified():
    async def go():
        dead = b"import package_that_does_not_exist_xyz\n" + MAIN
        svc, s, r = await _svc([("main.py", dead), ("controllers/home.py", HOME), ("services/items.py", ITEMS),
                                ("templates/index.html", PAGE)])
        v = await svc.verify(r, 0, {"services/items.py": ITEMS + b"# t\n"})
        assert v.boot["ran"] is False and "even without the change" in v.boot["why"]
        assert release_gate(v.to_dict(), ["services/items.py"], "smoke")           # the gate refuses it, with a reason
    run(go())


# ----------------------------------------------------------------------------- the whole pipeline
CALC = [("calc.py", b"def add(a, b):\n    return a + b\n"),
        ("tests/test_calc.py", b"from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")]
EDIT = block("calc.py", "    return a + b", "    return a + b + 0")


@pytest.mark.parametrize("mode,saved", [("strict", False), ("smoke", True), ("off", True)])
def test_pipeline_holds_back_a_change_that_has_no_proving_test(monkeypatch, mode, saved):
    monkeypatch.setattr(settings, "change_gate", mode)

    async def go():
        svc, s, r = await _change_svc(Script(EDIT), CALC)
        res = await svc.run(r, "refactor add in calc.py")
        assert res.ok is saved
        if saved:
            assert res.revision and res.diff
        else:      # nothing saved, nothing shown: the user never receives an unverified diff
            assert res.revision is None and res.diff == "" and await s.latest_rev(r) == 0
            assert any("Not shared" in e for e in res.errors) and "not shown" in " ".join(res.notes)
    run(go())


def test_pipeline_holds_back_when_the_sandbox_cannot_run(monkeypatch):
    monkeypatch.setattr(settings, "change_gate", "smoke")

    class Down:
        name = "docker"

        async def available(self):
            return False, "Docker CLI not found on PATH"

    async def go():
        svc, s, r = await _change_svc(Script(EDIT), CALC, runner=Down())
        res = await svc.run(r, "refactor add in calc.py")
        assert not res.ok and res.diff == "" and "Docker CLI not found" in " ".join(res.errors)
    run(go())


def test_pipeline_repairs_a_template_crash_from_the_real_traceback(monkeypatch):
    monkeypatch.setattr(settings, "change_gate", "smoke")

    async def go():
        llm = Script(block("controllers/home.py", HOME_LINE, HOME_BAD),      # attempt 1: drops `user`, imports fine, 500s on GET /
                     block("controllers/home.py", HOME_BAD, HOME_FIX))       # attempt 2: repaired after seeing the traceback
        svc, s, r = await _change_svc(llm, APP)
        res = await svc.run(r, "change the index handler in controllers/home.py")
        assert res.ok and res.attempts == 2, (res.errors, res.trace)
        assert "UndefinedError" in llm.prompts[1] and "boot::" in llm.prompts[1]
        assert res.tests["boot"]["ok"]
    run(go())


def test_pipeline_gives_up_without_sharing_when_the_app_stays_broken(monkeypatch):
    monkeypatch.setattr(settings, "change_gate", "smoke")

    async def go():
        steps = [HOME_LINE, HOME_BAD, HOME_BAD.replace("load()", "load() or []"), HOME_BAD.replace("load()", "load() or {}")]
        llm = Script(*[block("controllers/home.py", steps[i], steps[i + 1]) for i in range(settings.change_max_attempts)])
        svc, s, r = await _change_svc(llm, APP)
        res = await svc.run(r, "change the index handler in controllers/home.py")
        assert not res.ok and await s.latest_rev(r) == 0
        assert res.diff == "" and "```diff" not in res.to_markdown()        # a broken attempt is not shown, not even as "not saved"
    run(go())
