"""Call check, arity: calls that resolve to a repo method / function / constructor must fit its signature."""
from services import call_check as cc

LIB = b"class Repo:\n    def get(self, rid, rev=0, *, deep=False):\n        return rid\n\n\ndef total(a, b):\n    return a + b\n"
USE = (b"from lib import Repo, total\n\n\nclass Ctl:\n    def __init__(self):\n        self.repo = Repo()\n"
       b"    def ok(self):\n        self.repo.get(1)\n        self.repo.get(1, 2, deep=True)\n        self.repo.get(rid=1)\n        return total(1, 2)\n")


def _base():
    return {"lib.py": LIB, "use.py": USE}


def _with(extra, base=None):
    base = base or _base()
    return cc.new_problems(base, {**base, "use.py": USE + extra})


def test_calls_that_fit_are_clean():
    assert cc.new_problems(_base(), _base()) == ([], [])


def test_method_called_with_too_many_missing_or_unknown_arguments():
    for call, why in [("self.repo.get(1, 2, 3)", "at most 2"), ("self.repo.get()", "missing required"),
                      ("self.repo.get(1, nope=1)", "unexpected keyword argument 'nope'")]:
        probs, owners = _with(b"    def bad(self):\n        " + call.encode() + b"\n")
        assert len(probs) == 1 and why in probs[0], (call, probs)
        assert owners == ["lib.py", "use.py"]                     # the callee and the caller must both be visible to the patcher


def test_function_and_constructor_arity():
    probs, _ = _with(b"\n\ndef bad():\n    return total(1)\n")
    assert len(probs) == 1 and "total" in probs[0] and "missing required" in probs[0]
    probs, _ = _with(b"\n\ndef bad2():\n    return Repo(1, 2)\n")
    assert len(probs) == 1 and "takes no arguments" in probs[0]
    local = b"class K:\n    def __init__(self, a):\n        self.a = a\n\n\ndef make():\n    return K()\n"
    probs, _ = cc.new_problems({}, {"k.py": local})
    assert len(probs) == 1 and "K.__init__" in probs[0]


def test_changing_a_signature_breaks_untouched_callers():
    after = {**_base(), "lib.py": LIB.replace(b"def total(a, b)", b"def total(values)")}
    probs, owners = cc.new_problems(_base(), after)
    assert len(probs) == 1 and probs[0].startswith("use.py:") and "total" in probs[0] and owners == ["lib.py", "use.py"]
    after = {**_base(), "lib.py": LIB.replace(b"def get(self, rid,", b"def get(self, rid, tenant,")}
    probs, _ = cc.new_problems(_base(), after)
    assert any("missing required" in p and "tenant" in p for p in probs)


def test_what_cannot_be_judged_is_left_alone():
    star = b"    def s(self, args, kw):\n        self.repo.get(*args)\n        self.repo.get(**kw)\n        return total(*args)\n"
    assert _with(star) == ([], [])
    deco = {"d.py": b"import functools\n\n\n@functools.cache\ndef f(a):\n    return a\n\n\ndef g():\n    return f(1, 2, 3)\n"}
    assert cc.new_problems({}, deco) == ([], [])
    shadow = {"s.py": b"def total(a, b):\n    return a\n\n\ndef g(total):\n    return total(1)\n"}
    assert cc.new_problems({}, shadow) == ([], [])
    data = {"m.py": b"from dataclasses import dataclass\n\n\n@dataclass\nclass P:\n    x: int\n\n\ndef g():\n    return P(1)\n"}
    assert cc.new_problems({}, data) == ([], [])
    static = {"c.py": b"class C:\n    @staticmethod\n    def s(a):\n        return a\n\n    def run(self):\n        return self.s(1)\n"}
    assert cc.new_problems({}, static) == ([], [])


def test_a_mismatch_that_already_existed_is_not_blamed_on_the_change():
    bad = USE + b"    def old(self):\n        self.repo.get(1, 2, 3)\n"
    base = {"lib.py": LIB, "use.py": bad}
    assert cc.new_problems(base, base) == ([], [])
