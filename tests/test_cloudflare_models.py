"""Cloudflare runner: the UI model picker lists the account's Workers AI text-generation models."""
import asyncio

import pytest

from config.settings import settings
from services.runners import cloudflare_runner as cfr
from services.runners.cloudflare_runner import CloudflareRunner


@pytest.fixture(autouse=True)
def _cf(monkeypatch):
    monkeypatch.setattr(settings, "cf_account_id", "acc1")
    monkeypatch.setattr(settings, "cf_api_token", "tok")
    monkeypatch.setattr(settings, "cf_models", "@cf/google/gemma-4-26b-a4b-it")
    monkeypatch.setattr(settings, "cf_list_all_models", True)
    monkeypatch.setattr(settings, "cf_models_cache_s", 600)
    cfr._CACHE.update(names=[], at=0.0, key="")
    yield
    cfr._CACHE.update(names=[], at=0.0, key="")


def _pages(monkeypatch, pages, calls=None):
    async def fake(self, page):
        if calls is not None:
            calls.append(page)
        if isinstance(pages, Exception):
            raise pages
        return pages[page - 1] if page <= len(pages) else {"success": True, "result": []}
    monkeypatch.setattr(CloudflareRunner, "_search_page", fake)


def _row(name, task="Text Generation"):
    return {"name": name, "task": {"name": task}}


def test_lists_hosted_text_generation_models_pinned_first(monkeypatch):
    _pages(monkeypatch, [{"success": True, "result_info": {"total_count": 4}, "result": [
        _row("@cf/meta/llama-3.3-70b-instruct-fp8-fast"), _row("@cf/google/gemma-4-26b-a4b-it"),
        _row("@cf/baai/bge-base-en-v1.5", "Text Embeddings"), _row("anthropic/claude-opus-4.7")]}])
    got = asyncio.run(CloudflareRunner().list_models())
    assert got == ["@cf/google/gemma-4-26b-a4b-it", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"]


def test_pages_until_total_and_caches(monkeypatch):
    calls = []
    _pages(monkeypatch, [{"success": True, "result_info": {"total_count": 150}, "result": [_row(f"@cf/a/m{i}") for i in range(100)]},
                         {"success": True, "result_info": {"total_count": 150}, "result": [_row(f"@hf/b/m{i}") for i in range(50)]}], calls)
    first = asyncio.run(CloudflareRunner().list_models())
    second = asyncio.run(CloudflareRunner().list_models())
    assert len(first) == 151 and first == second and calls == [1, 2]


def test_api_error_falls_back_to_cf_models(monkeypatch):
    _pages(monkeypatch, RuntimeError("403"))
    assert asyncio.run(CloudflareRunner().list_models()) == ["@cf/google/gemma-4-26b-a4b-it"]


def test_switch_off_shows_only_cf_models(monkeypatch):
    calls = []
    _pages(monkeypatch, [{"success": True, "result": [_row("@cf/x/y")]}], calls)
    monkeypatch.setattr(settings, "cf_list_all_models", False)
    assert asyncio.run(CloudflareRunner().list_models()) == ["@cf/google/gemma-4-26b-a4b-it"] and calls == []
