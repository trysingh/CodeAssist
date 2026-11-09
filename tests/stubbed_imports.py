"""Import the controller / change service / ask agent in an environment that lacks their third-party packages (langchain, langgraph,
httpx, sqlalchemy, fastapi, aiosqlite, pydantic_settings, ...), so call sites can be EXECUTED against fakes.

Only packages that really are missing are stubbed (the real import is tried first). The stand-ins are inert: nothing that talks to a
model, a database or the network works through them, so a test must hand the code under test its own fakes. Every module that was
first imported while the stubs were installed is removed again on exit, so other tests never see a stub-bound class."""
import contextlib
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types

import pydantic


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Dummy()

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return _Dummy()

    def __class_getitem__(cls, item):
        return cls

    def __iter__(self):
        return iter(())


class _AutoModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        return type(n, (_Dummy,), {})


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        return _AutoModule(spec.name)

    def exec_module(self, module):
        pass


class _BaseSettings(pydantic.BaseModel):
    """pydantic_settings.BaseSettings stand-in: ENGINE_* environment variables fill the fields."""
    def __init__(self, **kw):
        env = {k[7:].lower(): v for k, v in os.environ.items() if k.startswith("ENGINE_")}
        super().__init__(**{**env, **kw})


@contextlib.contextmanager
def stubbed_imports():
    before = set(sys.modules)
    finder = _Finder()
    try:
        import pydantic_settings  # noqa: F401
    except ImportError:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings, ps.SettingsConfigDict = _BaseSettings, dict
        sys.modules["pydantic_settings"] = ps
    sys.meta_path.append(finder)            # last: only consulted when nothing real can be found
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name not in before:
                del sys.modules[name]


def import_module(name):
    return importlib.import_module(name)
