"""What goes INTO the sandbox: secret filter, env templates, merged requirements, network switch.
Run from the repo root:  pytest tests/test_sandbox_inputs.py -q      (no Docker, Ollama or network needed)
"""
import asyncio
import io
import tarfile
import uuid

import pytest

from config.settings import settings
from controllers.engine_controller import EngineController
from services import sandbox_service as sbx
from services.repo_store import RepoStore, is_secret_path, needs_content_check, secret_content_reason
from services.requirements_merge import collect, include_targets, merge_requirements, needs_rewrite

run = asyncio.run
ROOTS = ["requirements-test.txt", "requirements-dev.txt", "requirements.txt"]


def rid():
    return f"r_{uuid.uuid4().hex[:8]}"


# ============================================================================= 1. secret filter
@pytest.mark.parametrize("path", ["tokenizer.py", "auth/token_service.py", "secrets_manager.py", "credentials.py",
                                  "utils/cookie_utils.py", "web/token.js", ".env.example", "cfg/.env.sandbox"])
def test_source_files_with_secret_looking_names_are_stored_after_a_content_scan(path):
    assert not is_secret_path(path) and needs_content_check(path)


@pytest.mark.parametrize("path", ["token.json", "credentials.yaml", "secrets.txt", ".env", ".env.local", "server.key",
                                  "id_rsa", "x.pem", "fix.patch", "api/credentials.json"])
def test_data_and_key_files_stay_name_filtered(path):
    assert is_secret_path(path)


def test_ordinary_names_are_untouched():
    assert not is_secret_path("config.py") and not needs_content_check("config.py")


def test_content_scan_catches_real_looking_credentials_and_ignores_placeholders():
    assert secret_content_reason(b'KEY = "sk-' + b"a" * 40 + b'"')
    assert secret_content_reason(b"-----BEGIN RSA PRIVATE KEY-----\n")
    assert secret_content_reason(b'token = "' + b"Zq3" * 10 + b'"\n')
    assert secret_content_reason(b"AWS=AKIA" + b"A1" * 8 + b"\n")
    for clean in (b"API_KEY=your-api-key-here-please-change-me\n", b"SECRET_KEY=\n", b"OLLAMA_URL=http://localhost:11434\n",
                  b"def token(): return 1\n", b"password = get_password()\n"):
        assert secret_content_reason(clean) is None


def test_raw_store_keeps_token_service_but_withholds_one_with_a_hard_coded_key():
    async def go():
        s, r = RepoStore(), rid()
        out = await s.upsert_base_files(r, [
            ("auth/token_service.py", b"def issue():\n    return 1\n"),
            ("auth/token_leak.py", b'API_TOKEN = "' + b"Qx9" * 10 + b'"\n'),
            (".env.example", b"API_URL=http://x\nOPENAI_API_KEY=your-key-here\n"),
            (".env.real", b"X=1\n"),
            ("token.json", b"{}"),
        ])
        assert sorted(out["written"]) == [".env.example", "auth/token_service.py"]
        why = dict(out["skipped"])
        assert "credential" in why["auth/token_leak.py"] and why[".env.real"].startswith("secret-like")
        assert why["token.json"].startswith("secret-like")
        assert (await s.get_file(r, "auth/token_service.py", 0)).data.startswith(b"def issue")
    run(go())


def test_env_templates_pass_the_folder_walk_but_other_dot_files_do_not(tmp_path):
    c = EngineController.__new__(EngineController)            # is_excluded needs no state
    (tmp_path / "sub").mkdir()
    for f in (".env.example", "sub/.env.example", ".env", ".gitignore"):
        (tmp_path / f).write_text("X=1")
    assert not c.is_excluded(tmp_path / ".env.example", tmp_path)
    assert not c.is_excluded(tmp_path / "sub" / ".env.example", tmp_path)
    assert c.is_excluded(tmp_path / ".env", tmp_path) and c.is_excluded(tmp_path / ".gitignore", tmp_path)
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / ".env.example").write_text("X=1")
    assert c.is_excluded(tmp_path / "venv" / ".env.example", tmp_path)       # excluded directories still win


# ============================================================================= 2. requirements
def test_a_single_plain_file_is_returned_verbatim_so_existing_image_tags_survive():
    text = "fastapi>=0.1   # web\n\nrequests==2\n"
    files = {"requirements.txt": text}
    assert merge_requirements(collect(files.get, ROOTS), ROOTS) == text
    assert not needs_rewrite(text)


def test_dev_file_with_dash_r_gets_the_real_requirements_inlined():
    files = {"requirements-dev.txt": "-r requirements.txt\npytest==8\n", "requirements.txt": "fastapi\nrequests==2\n"}
    out = merge_requirements(collect(files.get, ROOTS), ROOTS)
    assert {"fastapi", "requests==2", "pytest==8"} <= set(out.splitlines())
    assert "-r " not in out.replace("# ", "")


def test_every_root_file_is_merged_and_the_first_file_wins_a_version_clash():
    files = {"requirements-test.txt": "pytest==8\nrequests==3\n", "requirements.txt": "Requests==2\nfastapi\n"}
    lines = merge_requirements(collect(files.get, ROOTS), ROOTS).splitlines()
    assert "requests==3" in lines and "Requests==2" not in lines and "fastapi" in lines and "pytest==8" in lines


def test_nested_includes_resolve_relative_to_the_including_file_and_cycles_are_safe():
    files = {"requirements.txt": "-r requirements/base.txt\n", "requirements/base.txt": "-r extra.txt\nrich\n",
             "requirements/extra.txt": "-r base.txt\nhttpx\n"}
    assert include_targets("requirements/base.txt", files["requirements/base.txt"]) == ["requirements/extra.txt"]
    out = merge_requirements(collect(files.get, ROOTS), ROOTS)
    assert "rich" in out.splitlines() and "httpx" in out.splitlines() and "circular" in out


def test_lines_the_build_cannot_satisfy_are_dropped_with_a_note():
    files = {"requirements.txt": "-c constraints.txt\n-e .\n./local_pkg\n-r missing.txt\nflask>=2\n"}
    out = merge_requirements(collect(files.get, ROOTS), ROOTS)
    assert [ln for ln in out.splitlines() if not ln.startswith("#")] == ["flask>=2"]
    assert out.count("dropped") == 4


def test_environment_markers_and_options_are_not_deduplicated_away():
    files = {"requirements.txt": 'numpy==1; python_version<"3.9"\nnumpy==2; python_version>="3.9"\n--extra-index-url https://x\n'}
    assert needs_rewrite("-e .\n") and not needs_rewrite(files["requirements.txt"])
    files["requirements-dev.txt"] = "pytest\n"
    out = merge_requirements(collect(files.get, ROOTS), ROOTS).splitlines()
    assert 'numpy==1; python_version<"3.9"' in out and 'numpy==2; python_version>="3.9"' in out


def test_requirements_for_equals_verify_side_requirements_for_dash_r_layouts():
    layouts = [
        [("requirements.txt", b"a==1\n"), ("requirements-dev.txt", b"-r requirements.txt\npytest\n")],
        [("requirements.txt", b"-r requirements/base.txt\n"), ("requirements/base.txt", b"b==2\n")],
        [("requirements.txt", b"a==1\n"), ("requirements-test.txt", b"pytest==8\n")],
        [],
    ]

    async def go():
        for extra in layouts:
            s, r = RepoStore(), rid()
            await s.upsert_base_files(r, [("calc.py", b"x = 1\n")] + extra)
            svc = sbx.SandboxService(s, sbx.LocalRunner(), enabled=True, emit=lambda *a: None)
            tree = await s.snapshot_bytes(r, await s.latest_rev(r))
            assert await svc.requirements_for(r) == svc._requirements(tree)
    run(go())


# ============================================================================= 3. network + env
def test_docker_run_has_no_network_unless_the_operator_opts_in(monkeypatch):
    d = sbx.DockerRunner()
    for value, expected in (("none", "none"), ("bridge", "bridge"), (" BRIDGE ", "bridge"), ("host", "none"), ("", "none")):
        monkeypatch.setattr(settings, "sandbox_network", value)
        args = d.docker_args("img:1", "n", 60, ["pytest"])
        assert args[args.index("--network") + 1] == expected


def test_hint_names_the_opt_in_only_while_the_network_is_off(monkeypatch):
    err = "requests.exceptions.ConnectionError: Max retries exceeded host='api.example.com'"
    monkeypatch.setattr(settings, "sandbox_network", "none")
    assert "ENGINE_SANDBOX_NETWORK=bridge" in sbx.hint_for(err)
    monkeypatch.setattr(settings, "sandbox_network", "bridge")
    assert "host.docker.internal" in sbx.hint_for(err) and "--network none" not in sbx.hint_for(err)


def test_sandbox_env_reads_the_templates_first_file_wins_and_skips_empty_and_reserved():
    tree = {".env.example": b'# c\nAPI_URL=http://x\nOPENAI_KEY=\nNAME="a b" # tail\nPATH=/evil\nSBX_X=1\nPLAIN=v # c\n',
            ".env.sandbox": b"API_URL=http://sandbox\nEXTRA=1\n"}
    assert sbx.sandbox_env(tree) == {"API_URL": "http://sandbox", "EXTRA": "1", "NAME": "a b", "PLAIN": "v"}
    assert sbx.sandbox_env({}) == {}


class _Spy(sbx.LocalRunner):
    def __init__(self):
        super().__init__()
        self.tars = []

    async def run(self, requirements, tar, argv, timeout):
        self.tars.append(tar)
        return await super().run(requirements, tar, argv, timeout)


def _members(tar):
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


APP = b'import os\n\ndef base():\n    return os.environ["API_URL"]\n'
TEST = b'from app import base\n\ndef test_base():\n    assert base() == "http://placeholder"\n'


def test_code_that_needs_env_vars_passes_in_the_sandbox_thanks_to_the_env_template():
    async def go():
        s, r = RepoStore(), rid()
        await s.upsert_base_files(r, [("app.py", APP), ("tests/test_app.py", TEST), (".env.example", b"API_URL=http://placeholder\n")])
        spy = _Spy()
        svc = sbx.SandboxService(s, spy, enabled=True, emit=lambda *a: None)
        v = await svc.verify(r, 0, {"app.py": APP + b"\n# touched\n"})
        assert v.ran and v.ok and v.report.passed == 1, v.report.note
        files = _members(spy.tars[0])
        assert files[".env"] == b'API_URL="http://placeholder"\n' and b"setdefault" in files["sitecustomize.py"]

        # no template -> nothing is added and the same test fails with a KeyError, as before
        s2, r2 = RepoStore(), rid()
        await s2.upsert_base_files(r2, [("app.py", APP), ("tests/test_app.py", TEST)])
        spy2 = _Spy()
        v2 = await sbx.SandboxService(s2, spy2, enabled=True, emit=lambda *a: None).verify(r2, 0, {"app.py": APP + b"\n# t\n"})
        assert not v2.ok and ".env" not in _members(spy2.tars[0]) and "sitecustomize.py" not in _members(spy2.tars[0])
    run(go())
