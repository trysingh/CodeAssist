"""Shared pieces for calls to the configured vision model (attachment OCR, web screenshot OCR). Light: heavy imports are lazy."""
from __future__ import annotations

import base64
from typing import Any

from config.settings import settings


def image_message(prompt: str, data: "bytes | list[bytes]", mime: str = "image/png") -> Any:
    """One HumanMessage: instruction text + the image(s) as data URIs (the format every runner's vision model accepts)."""
    from langchain_core.messages import HumanMessage
    imgs = [data] if isinstance(data, (bytes, bytearray)) else list(data)
    parts: list = [{"type": "text", "text": prompt}]
    parts += [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(b).decode()}"}} for b in imgs]
    return HumanMessage(content=parts)


def vision_chat(phase: str) -> Any:
    """Chat client on the configured vision model (settings.vision_model, empty = llm_model) of the request's runner."""
    from services.runners import make_chat
    try:
        from langchain_ollama import ChatOllama
    except ImportError:     # pragma: no cover - only the Ollama runner needs it
        ChatOllama = None
    return make_chat(settings.vision_model or settings.llm_model, temperature=0, vision=True, ollama_cls=ChatOllama, phase=phase)


def vision_available() -> bool:
    """True when the active runner reports the 'vision' capability. Unknown / failing runner lookup = False (fail closed)."""
    try:
        from services.runners import get_runner
        return bool(get_runner().supports.get("vision"))
    except Exception:  # noqa: BLE001
        return False
