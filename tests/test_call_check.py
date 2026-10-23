"""Call check: methods no repo class defines are caught; resolvable, existing calls are not."""
from services import call_check as cc

DB = b"class DatabaseManager:\n    async def get_repo(self, rid):\n        return rid\n"
SVC = (b"class Ctl:\n    def __init__(self):\n        self.db = DatabaseManager()\n"
       b"    async def ok(self, r):\n        return await self.db.get_repo(r)\n")
API = b"from c import Ctl\ncontroller = Ctl()\n\nasync def f(r):\n    return await controller.db.get_repo(r)\n"


def _base():
    return {"models/database.py": DB, "controllers/ctl.py": SVC, "views/api.py": API}


def test_existing_calls_are_clean():
    assert cc.new_problems(_base(), _base()) == ([], [])


def test_invented_method_is_flagged_with_its_owner_file():
    after = _base()
    after["controllers/ctl.py"] = SVC + b"    async def bad(self, r):\n        await self.db.delete_repo_chunks(r)\n"
    probs, owners = cc.new_problems(_base(), after)
    assert len(probs) == 1 and "delete_repo_chunks" in probs[0] and "DatabaseManager" in probs[0]
    assert owners == ["models/database.py"]


def test_defining_the_method_in_the_owner_clears_it():
    after = _base()
    after["controllers/ctl.py"] = SVC + b"    async def bad(self, r):\n        await self.db.delete_repo_chunks(r)\n"
    after["models/database.py"] = DB + b"    async def delete_repo_chunks(self, rid):\n        return rid\n"
    assert cc.new_problems(_base(), after) == ([], [])


def test_removing_a_method_breaks_untouched_callers():
    after = _base()
    after["models/database.py"] = DB.replace(b"get_repo", b"get_repo2")
    probs, _ = cc.new_problems(_base(), after)
    assert any("get_repo" in p for p in probs)


def test_module_level_instance_and_self_calls():
    after = _base()
    after["views/api.py"] = API + b"\nasync def g():\n    await controller.missing_method()\n"
    after["controllers/ctl.py"] = SVC + b"    async def h(self):\n        return self._nope()\n"
    probs, _ = cc.new_problems(_base(), after)
    joined = "\n".join(probs)
    assert "missing_method" in joined and "_nope" in joined


def test_unresolvable_receivers_and_open_classes_are_ignored():
    src = (b"class Child(Base):\n    def run(self):\n        self.anything()\n"
           b"class Holder:\n    def __init__(self, client):\n        self.client = client\n"
           b"    def go(self):\n        self.client.whatever()\n")
    assert cc.new_problems({}, {"x.py": src}) == ([], [])


def test_preexisting_problem_is_not_blamed_on_the_change():
    bad = SVC + b"    async def legacy(self, r):\n        await self.db.long_gone(r)\n"
    base = {"models/database.py": DB, "controllers/ctl.py": bad}
    assert cc.new_problems(base, base) == ([], [])
