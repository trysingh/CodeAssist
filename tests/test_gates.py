from services import gates

KNOWN = {"services/__init__.py", "services/rag_graph.py", "controllers/engine_controller.py"}
OLD = b"class A:\n    def __init__(self):\n        self.db = 1\n        self.web = 2\n    def run(self):\n        return self.web\n"
NEW = b"class A:\n    def __init__(self):\n        self.db = 1\n    def run(self):\n        return self.web\n"


def test_invented_local_module_is_flagged():
    assert gates.unresolved_local_imports(b"from services.cancellation import reg\n", KNOWN)


def test_real_and_third_party_imports_pass():
    src = b"import asyncio\nimport httpx\nfrom services import rag_graph\nfrom services.rag_graph import X\n"
    assert gates.unresolved_local_imports(src, KNOWN) == []


def test_dropped_attribute_still_read_is_flagged():
    assert "self.web" in gates.lost_attributes(OLD, NEW, "A.__init__")[0]


def test_untouched_class_passes():
    assert gates.lost_attributes(OLD, OLD, "A.__init__") == []
