"""Who does the web search for a chat: the app (SearXNG + scraper, services/web_search_service.py) or the runner's own native tool.

Pure logic, no langchain / network import. Setting: settings.web_provider (WEB_PROVIDER):
    app    always the app's own SearXNG path (the behaviour before runners could search)
    native the runner's own search when it has one (supports["web_search"]), else the app path
    auto   native ONLY for a chat that is not repo-scoped AND whose runner can search; a repo-scoped chat keeps the app path so code, file
           and client names never end up in search queries the model writes itself. Unknown values mean auto.
"""
from __future__ import annotations

PROVIDERS = ("app", "native", "auto")


def normalize_web_provider(value: object) -> str:
    v = str(value or "").strip().lower()
    return v if v in PROVIDERS else "auto"


def choose_web_provider(provider_setting: object, runner_supports_web: bool, repo_scoped: bool) -> str:
    """"app" | "native" for one chat turn."""
    mode = normalize_web_provider(provider_setting)
    if mode == "app" or not runner_supports_web:
        return "app"
    if mode == "native":
        return "native"
    return "app" if repo_scoped else "native"
