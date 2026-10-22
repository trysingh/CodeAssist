import sys
import json
from typing import Generator
from plyer import notification
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio
from config.settings import settings
class NotificationDispatcher:
    """Dispatches real-time re-ingestion notifications to Logs, Windows OS, and Web Toasts."""
    
    def __init__(self):
        self.subscribers: list[asyncio.Queue] = []

    def notify(self, title: str, message: str, level: str = "INFO"):
        # Handle cases where message might be passed as a dictionary (e.g. Telemetry payload)
        if isinstance(message, dict):
            message = json.dumps(message)

        # 1. Logs / Stdout
        timestamp_log = f"\n[NOTIFICATION [{level}]] {title} — {message}\n"
        sys.stdout.write(timestamp_log)
        sys.stdout.flush()

        # 2. Windows OS Level Desktop Notification
        try:
            if settings.notify_at_os_level: 
                notification.notify(
                    title=f"CodeRAG Engine: {title}",
                    message=message,
                    app_name="CodeRAG Engine",
                    timeout=5
                )
        except Exception:
            pass  # Fallback gracefully if running on headless server / container

        # 3. Web App SSE Queues Broadcast
        event_payload = {"title": title, "message": message, "level": level}
        for q in self.subscribers:
            try:
                q.put_nowait(event_payload)
            except Exception:
                pass

    async def subscribe(self) -> Generator:
        q = asyncio.Queue()
        self.subscribers.append(q)
        try:
            while True:
                data = await q.get()
                yield {"event": "notification", "data": json.dumps(data)}
        finally:
            self.subscribers.remove(q)

notifier = NotificationDispatcher()