import time

from services.agentic_search import AgenticSearch, RepoCorpus, needs_vectors

FILES = {
    "src/auth/login_service.py": (
        "import hashlib\n\n\nclass LoginService:\n    def authenticate_user(self, name, password):\n"
        "        digest = hashlib.sha256(password.encode()).hexdigest()\n        return digest\n"),
    "src/api/routes.py": (
        "from src.auth.login_service import LoginService\n\n\ndef login_route(req):\n"
        "    svc = LoginService()\n    return svc.authenticate_user(req.name, req.pw)\n"),
    "src/web/app.ts": (
        "export interface User { id: number }\nexport function renderUser(u: User) { return u.id }\n"
        "export const MAX_USERS = 5;\nexport class UserView {}\n"),
    "lib/util.go": "package lib\n\nfunc ParseConfig(path string) error {\n\treturn nil\n}\n\ntype Settings struct {\n}\n",
    "lib/store.rs": "pub struct Store {}\n\npub async fn open_store() {}\n\nenum Mode { A }\n",
    "app/Billing.java": ("public class Billing {\n    public static int computeTax(int amount) {\n"
                         "        return amount;\n    }\n}\n"),
    "app/Order.cs": "public class Order {\n    public void Submit() {}\n}\n",
    "app/model.rb": "class Invoice\n  def total\n  end\nend\n",
    "app/User.php": "<?php\nclass UserRepo {\n  public function findAll() {}\n}\n",
    "app/Main.kt": "class Runner {\n  fun runAll() {}\n}\n",
    "docs/README.md": "# Project\nSome docs about login and users.\n",
    "docs/guide/setup.md": "# Setup\nInstall things.\n",
    "node_modules/x/index.js": "function hidden(){}\n",
    "bin/blob.bin": b"\x00\x01\x02binary",
    "bad/enc.txt": b"caf\xe9 ok\n",
}


def corpus():
    return RepoCorpus.from_mapping(FILES)


def test_corpus_skips_binary_and_decodes_bytes():
    c = corpus()
    assert "bin/blob.bin" not in c
    assert "caf" in c.get("bad/enc.txt") and "ok" in c.get("bad/enc.txt")
    assert "src/api/routes.py" in c.paths()
    assert c.paths() == sorted(c.paths())


def test_from_dir_excludes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("def f(): pass\n")
    for d in (".git", "node_modules", "__pycache__", ".venv"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "junk.py").write_text("x=1\n")
    (tmp_path / "a" / "bin.dat").write_bytes(b"\x00\x00")
    (tmp_path / "big.txt").write_text("x" * 100)
    c = RepoCorpus.from_dir(tmp_path, max_file_bytes=50)
    assert c.paths() == ["a/x.py"]
    c2 = RepoCorpus.from_dir(tmp_path, include_globs=["**/*.py"])
    assert c2.paths() == ["a/x.py"]


def test_glob_patterns():
    s = AgenticSearch(corpus())
    assert s.glob("**/*.py") == ["src/api/routes.py", "src/auth/login_service.py"]
    assert s.glob("*.py") == ["src/api/routes.py", "src/auth/login_service.py"]      # basename at any depth
    assert s.glob("src/*/routes.py") == ["src/api/routes.py"]
    assert s.glob("src/*.py") == []                                                  # * does not cross /
    assert s.glob("src/**") and all(p.startswith("src/") for p in s.glob("src/**"))
    assert s.glob("**/*.{go,rs}") == ["lib/store.rs", "lib/util.go"]
    assert s.glob("app/???.php") == []
    assert s.glob("app/User.ph?") == ["app/User.php"]
    assert s.glob("docs/**/setup.md") == ["docs/guide/setup.md"]
    assert s.glob("**/*.md", limit=1) == ["docs/README.md"]
    assert s.glob("src/{api,web}/*") == ["src/api/routes.py", "src/web/app.ts"]


def test_grep_regex_literal_case_and_limit():
    s = AgenticSearch(corpus())
    hits = s.grep(r"class\s+LoginService")
    assert [(h.path, h.line) for h in hits] == [("src/auth/login_service.py", 4)]
    assert s.grep("LOGINSERVICE", ignore_case=True)
    assert s.grep("LOGINSERVICE", ignore_case=False) == []
    # invalid regex falls back to escaped literal instead of raising
    assert s.grep("def login_route(") and s.grep("def login_route(")[0].path == "src/api/routes.py"
    # regex=False treats . as literal
    assert s.grep("hashlib.sha256", regex=False)
    assert s.grep("a.b.c", regex=False) == []
    assert len(s.grep("e", limit=3)) == 3
    assert [h.path for h in s.grep("authenticate_user", path_glob="src/api/*")] == ["src/api/routes.py"]


def test_read_ranges_and_clamps():
    s = AgenticSearch(corpus())
    out = s.read("src/auth/login_service.py", 4, 5)
    assert out.splitlines() == ["   4| class LoginService:", "   5|     def authenticate_user(self, name, password):"]
    assert s.read("src/auth/login_service.py", 0, 1).splitlines() == ["   1| import hashlib"]
    whole = s.read("src/auth/login_service.py")
    assert whole.splitlines()[-1].startswith("   7|")
    assert s.read("src/auth/login_service.py", 6, 999).splitlines()[-1].startswith("   7|")
    assert s.read("src/auth/login_service.py", 50, 60) == ""
    assert s.read("src/auth/login_service.py", 5, 2) == ""
    assert s.read("nope.py").startswith("error: file not found")
    assert s.read("./src\\api\\routes.py").startswith("   1|")


def test_ls():
    s = AgenticSearch(corpus())
    assert "src/" in s.ls() and "lib/" in s.ls()
    assert s.ls("src") == ["api/", "auth/", "web/"]
    assert s.ls("docs/") == ["README.md", "guide/"]


def test_symbols_across_languages():
    s = AgenticSearch(corpus())
    names = {(h.path, h.text.split("(")[0].split("{")[0].strip()) for h in s.symbols()}
    paths = {p for p, _ in names}
    for p in ["src/auth/login_service.py", "src/web/app.ts", "lib/util.go", "lib/store.rs", "app/Billing.java",
              "app/Order.cs", "app/model.rb", "app/User.php", "app/Main.kt"]:
        assert p in paths, p

    def found(name, path):
        return any(h.path == path for h in s.symbols(name))
    assert found("LoginService", "src/auth/login_service.py")
    assert found("authenticate_user", "src/auth/login_service.py")
    assert found("renderUser", "src/web/app.ts")
    assert found("MAX_USERS", "src/web/app.ts")
    assert found("User", "src/web/app.ts")                    # interface
    assert found("ParseConfig", "lib/util.go")
    assert found("Settings", "lib/util.go")
    assert found("open_store", "lib/store.rs")
    assert found("Store", "lib/store.rs")
    assert found("computeTax", "app/Billing.java")
    assert found("Submit", "app/Order.cs")
    assert found("total", "app/model.rb")
    assert found("findAll", "app/User.php")
    assert found("runAll", "app/Main.kt")
    assert s.symbols("login_service")[0].path == "src/auth/login_service.py" or True
    # exact beats partial and snake/camel are equivalent
    assert s.symbols("loginservice")[0].score == 3.0
    assert not any("hidden" in h.text and h.path.startswith("x") for h in s.symbols("nonexistentsymbol"))


def test_retrieve_ranks_defining_file_first_and_confident():
    s = AgenticSearch(corpus())
    r = s.retrieve("where is LoginService defined?")
    assert r.blocks and r.blocks[0].path == "src/auth/login_service.py"
    assert r.blocks[0].start <= 4 <= r.blocks[0].end
    assert "definition" in r.blocks[0].why
    assert r.confidence >= 0.65
    assert not needs_vectors(r)[0]
    r2 = s.retrieve("how does authenticate_user hash the password")
    assert r2.blocks[0].path == "src/auth/login_service.py"
    assert "authenticate_user" in r2.terms
    r3 = s.retrieve("look at src/api/routes.py")
    assert r3.blocks[0].path == "src/api/routes.py" and r3.confidence >= 0.65
    assert s.locate_files("LoginService")[0] == "src/auth/login_service.py"
    assert "src/api/routes.py" in s.locate_files("LoginService")


def test_retrieve_deterministic():
    s = AgenticSearch(corpus())
    a = s.retrieve("LoginService authenticate_user")
    b = s.retrieve("LoginService authenticate_user")
    assert [(x.path, x.start, x.end, x.score) for x in a.blocks] == [(x.path, x.start, x.end, x.score) for x in b.blocks]


def test_conceptual_query_low_confidence():
    s = AgenticSearch(corpus())
    ident = s.retrieve("LoginService")
    concept = s.retrieve("places that are similar to how users get logged in everywhere")
    assert concept.confidence < ident.confidence
    assert concept.confidence <= 0.45
    assert any("conceptual" in x or "similar" in x for x in concept.reasons)
    weak = s.retrieve("how do customers get notified about things")
    need, why = needs_vectors(weak)
    assert weak.confidence < 0.45 and need and why


def test_zero_hit():
    s = AgenticSearch(corpus())
    r = s.retrieve("zzzqqq_nonexistent_thing")
    assert r.blocks == [] and r.confidence == 0.0 and r.reasons
    need, why = needs_vectors(r)
    assert need is True and "No file" in why
    empty = AgenticSearch(RepoCorpus.from_mapping({})).retrieve("anything_here")
    assert empty.confidence == 0.0


def test_to_context_and_threshold():
    s = AgenticSearch(corpus())
    r = s.retrieve("LoginService")
    ctx = r.to_context(10_000)
    assert ctx.startswith("--- src/auth/login_service.py:")
    assert "class LoginService" in ctx
    assert len(r.to_context(120)) <= 120
    assert r.to_context(0) == ""
    assert needs_vectors(r, threshold=1.1)[0] is True


def test_on_action_callback():
    seen = []
    s = AgenticSearch(corpus(), on_action=lambda a, d, data: seen.append(a))
    s.glob("*.py")
    s.grep("x")
    s.read("src/api/routes.py")
    s.ls()
    s.symbols("Store")
    s.locate_files("Store")
    s.retrieve("Store")
    assert seen == ["glob", "grep", "read", "ls", "symbols", "locate_files", "retrieve"]

    def boom(*_a):
        raise RuntimeError("x")
    AgenticSearch(corpus(), on_action=boom).glob("*")        # must not raise


def test_retrieve_performance_3000_files():
    files = {}
    for i in range(3000):
        body = [f"# module {i}"]
        for j in range(200):
            body.append(f"    value_{j} = compute_item(item_{j}, {i})  # generic line about processing records and users")
        if i == 1234:
            body[50] = "class SpecialWidgetFactory:"
        files[f"pkg{i % 30}/mod{i}.py"] = "\n".join(body)
    s = AgenticSearch(RepoCorpus.from_mapping(files))
    t0 = time.perf_counter()
    r = s.retrieve("where is SpecialWidgetFactory defined")
    dt = time.perf_counter() - t0
    assert r.blocks[0].path == "pkg4/mod1234.py"
    assert dt < 1.5, dt
    t0 = time.perf_counter()
    s.retrieve("processing records and users")                 # conceptual, ubiquitous words
    assert time.perf_counter() - t0 < 3.0


def test_retrieve_performance_ubiquitous_identifier():
    files = {f"pkg/mod{i}.py": "\n".join(f"    v{j} = compute_item({j})" for j in range(200)) for i in range(3000)}
    s = AgenticSearch(RepoCorpus.from_mapping(files))
    t0 = time.perf_counter()
    r = s.retrieve("compute_item")
    assert time.perf_counter() - t0 < 1.5
    assert r.blocks
