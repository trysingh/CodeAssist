"""Which speech-to-text providers to try, in which order. Independent of the chat runner (like the embedding model), but `auto` follows it."""
from __future__ import annotations

from typing import List

from config.settings import settings
from services.speech.providers import SttProvider, make_provider

# runner -> the provider that needs no extra account for that runner's user. ollama / claude_code / unknown: none.
AUTO_BY_RUNNER = {"aistudio": "gemini", "cloudflare": "cloudflare", "openai": "openai_compat", "chatgpt_codex": "openai"}
PROVIDER_NAMES = ("gemini", "cloudflare", "openai_compat", "openai", "local")
_ALIASES = {"aistudio": "gemini", "gemini": "gemini"}
_NO_SERVER = ("browser", "off", "none", "disabled")


def effective_runner(runner: str = "") -> str:
    """'' = the server's default runner (settings.llm_runner)."""
    return ((runner or "").strip() or (settings.llm_runner or "ollama")).strip().lower()


def resolve_chain(runner: str = "") -> List[SttProvider]:
    """Available providers in trial order. Empty list = server-side transcription is not possible: the client uses the browser's Web Speech API.

    stt_provider != auto: that provider first. auto: the one matching the runner. Then the self-hosted STT_BASE_URL ("local"), then (unless
    VOICE_CLOUD_FALLBACK=false) every other configured cloud provider. Providers that are not configured are dropped."""
    choice = (settings.stt_provider or "auto").strip().lower()
    if choice in _NO_SERVER:
        return []
    order: List[str] = []
    if choice != "auto" and (_ALIASES.get(choice, choice) in PROVIDER_NAMES):
        order.append(_ALIASES.get(choice, choice))
    else:
        first = AUTO_BY_RUNNER.get(effective_runner(runner))
        if first:
            order.append(first)
    if settings.stt_base_url:
        order.append("local")
    if settings.voice_cloud_fallback:
        order.extend(n for n in PROVIDER_NAMES if n != "local")
    chain: List[SttProvider] = []
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        p = make_provider(name)
        if p is not None and p.available()[0]:
            chain.append(p)
    return chain
