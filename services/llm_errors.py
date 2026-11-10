"""Provider-side quota failures of the chat model (Ollama Cloud: HTTP 402 "Add usage credits ...", or 429 at a usage limit).

A cloud model tag (gemma4:31b-cloud, gpt-oss:120b-cloud, ...) is proxied by the local Ollama daemon to ollama.com and billed
there. When the free allowance or the credits run out, ollama.com answers 402 and the langchain-ollama call raises
`ollama.ResponseError(status_code=402)`. That is not a bug in this app and waiting does not help, so this module:

  * recognises it (`as_quota_error`), duck-typed on `.status_code` so it needs neither the ollama package nor a server,
  * retries ONCE on `settings.llm_fallback_model` (a LOCAL model) when one is configured (`with_model_fallback`), and
    remembers for `settings.llm_quota_cooldown_s` that the cloud model is out, so a multi-turn agent does not knock on it
    again for every turn,
  * otherwise raises `ModelQuotaError`, which the API turns into HTTP 402 with a readable message (`quota_error_response`)
    instead of an unhandled 500 with a 200-line traceback.

Runner-aware: besides ollama's ResponseError it recognises OpenAI-style errors (openai.APIStatusError / httpx: `.status_code`
or `.response.status_code`), which is what the Cloudflare Workers AI runner raises (daily neuron allocation used up = 429 with a
quota text; bad / unauthorised token = 401 / 403 -> `ModelAuthError`). Google AI Studio (Gemini API) uses the same shape: 429
RESOURCE_EXHAUSTED / "exceeded your current quota", 400 API_KEY_INVALID, 401 / 403 PERMISSION_DENIED. Messages name the active
runner (the request's runner_scope, else settings.llm_runner).

Timeouts and transport failures get the same treatment (never a bare 500, never a silent UI): a request that times out
(openai.APITimeoutError / langchain_openai's OpenAITimeoutError, httpx.ReadTimeout / ConnectTimeout, asyncio.TimeoutError, ... matched by
CLASS NAME along the exception's MRO, so no provider package is imported) becomes `ModelTimeoutError` (HTTP 504 {"detail", "kind": "timeout",
"runner", "model", "timeout_s"}); a refused / unreachable connection (APIConnectionError, httpx.ConnectError) becomes `ModelUnavailableError`
(HTTP 503); anything else that escapes a route is turned into HTTP 500 {"detail": "Type: message", "kind": "internal"} by
`InternalErrorMiddleware` (the traceback is logged once).

Imports only config.settings at module level (notifier / action_log are imported lazily), so there is no import cycle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

from config.settings import settings
from services.runners.base import MODEL_NAME as _MODEL_NAME     # same shape model_calls.model_scope accepts ('@' allowed: Cloudflare ids)

T = TypeVar("T")
log = logging.getLogger("coderag.errors")

PLANS_URL = "https://ollama.com/plans"
AISTUDIO_KEY_URL = "https://aistudio.google.com/apikey"
AISTUDIO_LIMITS_URL = "https://ai.google.dev/gemini-api/docs/rate-limits"
_CLOUD = re.compile(r"[-:]cloud$", re.IGNORECASE)
_QUOTA_TEXT = re.compile(r"credit|usage|quota|subscri|upgrade|/plans|neuron|allocation|daily limit|resource_exhausted|resource has been exhausted", re.IGNORECASE)   # a busy LOCAL Ollama also answers 429 ("maximum pending requests"): that is not a quota


def is_cloud_model(model: str) -> bool:
    """gemma4:31b-cloud, gpt-oss:120b-cloud, gpt-oss:cloud ... (billed by ollama.com, not run locally)."""
    return bool(_CLOUD.search((model or "").strip()))


def _active_runner() -> str:
    try:
        from services.runners import current_runner_name
        return current_runner_name()
    except Exception:  # noqa: BLE001
        return settings.llm_runner


class ModelQuotaError(RuntimeError):
    """The model endpoint refused the call because credits / usage ran out. Waiting does not fix it."""

    def __init__(self, model: str, status: int = 402, provider_message: str = "",
                 fallback_tried: str = "", fallback_error: str = "", runner: str = ""):
        self.model, self.status = model, status
        self.runner = (runner or _active_runner() or "ollama").strip().lower()
        self.provider_message = (provider_message or "").strip()
        self.fallback_tried, self.fallback_error = fallback_tried, fallback_error
        super().__init__(self.user_message())

    def user_message(self) -> str:
        if self.runner == "cloudflare":
            msg = (f"Model '{self.model}' was refused (HTTP {self.status}): Cloudflare Workers AI says the daily free allocation "
                   "(10,000 neurons) or the plan limit is used up.")
            if self.fallback_tried:
                return (msg + f" The fallback model '{self.fallback_tried}' could not answer either ({self.fallback_error or 'no reply'}). "
                        "Pick another model in the model selector or set LLM_RUNNER=ollama.")
            return msg + (" Wait for the daily reset, move the account to Workers Paid, pick another model in the model selector,"
                          " or set LLM_RUNNER=ollama in .env.")
        if self.runner == "aistudio":
            msg = (f"Model '{self.model}' was refused (HTTP {self.status}): Google AI Studio says the quota or rate limit is used up "
                   "(RESOURCE_EXHAUSTED: requests / tokens per minute or per day; limits differ per model and plan).")
            if self.fallback_tried:
                msg += f" The fallback model '{self.fallback_tried}' could not answer either ({self.fallback_error or 'no reply'})."
            return msg + (f" Wait a minute and retry, pick another model in the model selector, check your limits in AI Studio ({AISTUDIO_LIMITS_URL}),"
                          f" enable billing for the key's project ({AISTUDIO_KEY_URL}), or set LLM_RUNNER=ollama.")
        if self.runner == "openai":
            msg = f"Model '{self.model}' was refused (HTTP {self.status}): the OpenAI-compatible endpoint says the credits, quota or rate limit are used up."
            if self.fallback_tried:
                msg += f" The fallback model '{self.fallback_tried}' could not answer either ({self.fallback_error or 'no reply'})."
            return msg + " Check the provider's dashboard (OPENAI_COMPAT_BASE_URL), pick another model, or set LLM_RUNNER=ollama."
        who = "Ollama Cloud" if is_cloud_model(self.model) else "the model endpoint"
        msg = f"Model '{self.model}' was refused (HTTP {self.status}): {who} says the usage credits or limit are used up."
        if self.fallback_tried:
            msg += f" The fallback model '{self.fallback_tried}' could not answer either ({self.fallback_error or 'no reply'})."
            return msg + " Check that it is pulled (`ollama list`), or pick another model in the model selector."
        return msg + (" Pick a local model in the model selector, set ENGINE_LLM_FALLBACK_MODEL=<a local model from `ollama list`>"
                      f" in .env to switch automatically, or add credits ({PLANS_URL}).")


def _chain(exc: BaseException):
    seen = 0
    while exc is not None and seen < 6:
        yield exc
        exc = exc.__cause__ or exc.__context__
        seen += 1


def _text_of(exc: BaseException) -> str:
    body = getattr(exc, "body", None)                       # openai errors keep the provider's JSON here
    return f"{getattr(exc, 'error', '') or ''} {getattr(exc, 'message', '') or ''} {exc} {body if body else ''}"


def _status_of(e: BaseException):
    """HTTP status of a provider error: `.status_code` (ollama.ResponseError, openai.APIStatusError) or `.response.status_code` (httpx)."""
    status = getattr(e, "status_code", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    return status


def quota_status(exc: BaseException) -> Optional[int]:
    """402 always; 429 only when its text talks about usage / credits / plans / neurons. None = not a quota failure."""
    for e in _chain(exc):
        status = _status_of(e)
        if status == 402:
            return 402
        if status == 429 and _QUOTA_TEXT.search(_text_of(e)):
            return 429
    return None


def as_quota_error(exc: BaseException, model: str) -> Optional[ModelQuotaError]:
    """exc as a ModelQuotaError, or None when it is some other failure (network, bad model name, ...)."""
    if isinstance(exc, ModelQuotaError):
        return exc
    status = quota_status(exc)
    if status is None:
        return None
    for e in _chain(exc):
        if _status_of(e) == status:
            return ModelQuotaError(model, status, str(getattr(e, "error", "") or e)[:300])
    return ModelQuotaError(model, status, str(exc)[:300])


class ModelAuthError(RuntimeError):
    """The model endpoint refused the credentials (HTTP 401 / 403). Retrying or a fallback model does not help."""

    def __init__(self, model: str, status: int = 401, provider_message: str = "", runner: str = ""):
        self.model, self.status = model, status
        self.runner = (runner or _active_runner() or "ollama").strip().lower()
        self.provider_message = (provider_message or "").strip()
        super().__init__(self.user_message())

    def user_message(self) -> str:
        head = f"Model '{self.model}' was refused (HTTP {self.status}): "
        if self.runner == "cloudflare":
            return head + ("Cloudflare Workers AI rejected the credentials. Check CF_API_TOKEN (needs Workers AI Read + Edit, no stray quotes "
                           "or spaces) and CF_ACCOUNT_ID; see docs/CLOUDFLARE_SETUP.md.")
        if self.runner == "aistudio":
            return head + ("Google AI Studio rejected the API key (API_KEY_INVALID / PERMISSION_DENIED). Check GEMINI_API_KEY (no stray quotes or "
                           f"spaces; create one at {AISTUDIO_KEY_URL}), that the Generative Language API is allowed for its project, and that the "
                           "model is available to your region and plan; see docs/GOOGLE_AI_STUDIO_SETUP.md.")
        if self.runner == "openai":
            return head + "the OpenAI-compatible endpoint rejected the credentials. Check OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL."
        return head + "the endpoint rejected the credentials. For an Ollama Cloud model run `ollama signin` (or set OLLAMA_API_KEY)."


_BAD_KEY_TEXT = re.compile(r"API_KEY_INVALID|API key not valid|API key expired", re.IGNORECASE)     # Google answers a wrong key with HTTP 400


def auth_status(exc: BaseException) -> Optional[int]:
    """401 / 403 of a provider error (or anything it was raised from); 400 when its text says the API key is invalid. Else None."""
    for e in _chain(exc):
        status = _status_of(e)
        if status in (401, 403):
            return status
        if status == 400 and _BAD_KEY_TEXT.search(_text_of(e)):
            return 400
    return None


def as_auth_error(exc: BaseException, model: str) -> Optional[ModelAuthError]:
    if isinstance(exc, ModelAuthError):
        return exc
    status = auth_status(exc)
    return None if status is None else ModelAuthError(model, status, str(exc)[:300])


# ---------------------------------------------------------------- timeouts / unreachable endpoint
# Matched by class name along the MRO: openai.APITimeoutError, langchain_openai's OpenAITimeoutError, httpx.ReadTimeout / WriteTimeout /
# ConnectTimeout / PoolTimeout (base TimeoutException), asyncio.TimeoutError / the builtin TimeoutError.
_TIMEOUT_NAMES = frozenset({"APITimeoutError", "OpenAITimeoutError", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout",
                            "TimeoutException", "TimeoutError"})
# openai.APIConnectionError (its APITimeoutError subclass is tested FIRST), httpx.ConnectError, the builtin ConnectionError family.
_CONNECT_NAMES = frozenset({"APIConnectionError", "ConnectError", "ConnectionError"})


def _mro_names(e: BaseException) -> set:
    return {c.__name__ for c in type(e).__mro__}


def is_timeout(exc: BaseException) -> bool:
    return any(_mro_names(e) & _TIMEOUT_NAMES for e in _chain(exc))


def is_unreachable(exc: BaseException) -> bool:
    return not is_timeout(exc) and any(_mro_names(e) & _CONNECT_NAMES for e in _chain(exc))


def _runner_facts(model: str) -> dict:
    """What the active runner is called, which id it really calls for `model`, its per-attempt timeout, attempts and timeout setting."""
    name = _active_runner()
    facts = {"runner": name, "label": name, "model": model, "timeout_s": None, "attempts": 1, "setting": ""}
    try:
        from services.runners import get_runner
        r = get_runner()
        facts["runner"] = str(getattr(r, "name", name) or name)
        facts["label"] = str(getattr(r, "label", "") or facts["runner"])
        try:
            facts["model"] = r.map_model(model) or model
        except Exception:  # noqa: BLE001
            pass
        t = getattr(r, "timeout_s", None)
        t = t() if callable(t) else None
        facts["timeout_s"] = float(t) if t else None
        n = getattr(r, "max_retries", None)
        facts["attempts"] = (int(n()) if callable(n) else 0) + 1
        facts["setting"] = str(getattr(r, "timeout_setting", "") or "")
    except Exception:  # noqa: BLE001 - the message is best effort; the error itself must still come out
        pass
    return facts


def _fmt_s(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


class ModelTimeoutError(RuntimeError):
    """The model endpoint did not answer in time (on every attempt). Retrying the same call mostly repeats the wait."""

    def __init__(self, model: str, timeout_s: Optional[float] = None, attempts: Optional[int] = None, runner: str = "",
                 provider_message: str = "", setting: Optional[str] = None):
        f = _runner_facts(model)
        self.model = f["model"]                 # the id the runner really called
        self.runner = (runner or f["runner"] or "ollama").strip().lower()
        self.label = f["label"] if self.runner == f["runner"] else self.runner
        self.timeout_s = f["timeout_s"] if timeout_s is None else timeout_s
        self.attempts = f["attempts"] if attempts is None else attempts
        self.setting = f["setting"] if setting is None else setting
        self.provider_message = (provider_message or "").strip()
        super().__init__(self.user_message())

    def user_message(self) -> str:
        how = f"within {_fmt_s(self.timeout_s)}s" if self.timeout_s else "in time"
        tail = f"raise {self.setting}" if self.setting else "check that the server is reachable"
        return (f"{self.label} did not answer {how} (model {self.model}, {self.attempts} attempt(s)). "
                f"Try a faster model (e.g. a Gemini Flash model), Lean mode, or {tail}.")


class ModelUnavailableError(RuntimeError):
    """The model endpoint could not be reached at all (connection refused / failed / DNS)."""

    def __init__(self, model: str, provider_message: str = "", runner: str = ""):
        f = _runner_facts(model)
        self.model = f["model"]
        self.runner = (runner or f["runner"] or "ollama").strip().lower()
        self.label = f["label"] if self.runner == f["runner"] else self.runner
        self.provider_message = (provider_message or "").strip()
        super().__init__(self.user_message())

    def user_message(self) -> str:
        hint = {"ollama": f"Is Ollama running at {settings.ollama_base_url}? (`ollama serve`)",
                "aistudio": "Check the internet connection / proxy (generativelanguage.googleapis.com) or AISTUDIO_BASE_URL.",
                "cloudflare": "Check the internet connection / proxy (api.cloudflare.com) or CF_AI_GATEWAY_URL.",
                "openai": "Check OPENAI_COMPAT_BASE_URL and that the server is running."}.get(self.runner, "Check the endpoint URL and your network.")
        return f"{self.label} could not be reached (model {self.model}): the connection failed. {hint}"


def as_timeout_error(exc: BaseException, model: str) -> Optional[ModelTimeoutError]:
    if isinstance(exc, ModelTimeoutError):
        return exc
    return ModelTimeoutError(model, provider_message=f"{type(exc).__name__}: {exc}"[:300]) if is_timeout(exc) else None


def as_unavailable_error(exc: BaseException, model: str) -> Optional[ModelUnavailableError]:
    if isinstance(exc, ModelUnavailableError):
        return exc
    return ModelUnavailableError(model, f"{type(exc).__name__}: {exc}"[:300]) if is_unreachable(exc) else None


def _report_failure(title: str, msg: str, model: str, phase: str) -> None:
    """Make the failure visible in the UI's telemetry list and Activity timeline (soft: reporting never changes the error)."""
    try:
        from services.notifier import notifier
        notifier.notify(title, msg, level="ERROR")
    except Exception:  # noqa: BLE001
        pass
    try:
        from services.action_log import log_action
        log_action("LLM", phase or "LLM", "llm_call", "fail", msg[:300], model=model)
    except Exception:  # noqa: BLE001
        pass


def fallback_for(model: str) -> str:
    """settings.llm_fallback_model when it is usable for `model` (set, valid, different), else ''.

    "Different" is judged after the runner's mapping: on Cloudflare an Ollama-style fallback name maps to the same model as the
    refused one, which would only repeat the refusal."""
    fb = (settings.llm_fallback_model or "").strip()
    if not fb or fb == (model or "").strip() or not _MODEL_NAME.fullmatch(fb):
        return ""
    try:
        from services.runners import get_runner
        r = get_runner()
        if r.map_model(fb) == r.map_model(model):
            return ""
    except Exception:  # noqa: BLE001 - an unusable runner config is reported by the call itself
        pass
    return fb


# ---------------------------------------------------------------- cooldown: model -> (until, status)
_blocked: Dict[str, Tuple[float, int]] = {}


def _block(model: str, status: int) -> None:
    cooldown = max(0.0, float(settings.llm_quota_cooldown_s))
    if cooldown:
        _blocked[model] = (time.monotonic() + cooldown, status)


def _blocked_status(model: str) -> Optional[int]:
    until, status = _blocked.get(model, (0.0, 0))
    return status if until > time.monotonic() else None


def reset_quota_state() -> None:
    """Forget every cooldown (tests; or after adding credits without restarting the server)."""
    _blocked.clear()


# ---------------------------------------------------------------- "a fallback answered" notes
_events: ContextVar[Optional[List[dict]]] = ContextVar("llm_fallback_events", default=None)


@contextmanager
def fallback_scope():
    """Collects the fallbacks that happen inside the block (also from child tasks: the list is shared by reference)."""
    events: List[dict] = []
    token = _events.set(events)
    try:
        yield events
    finally:
        _events.reset(token)


def _record(src: str, dst: str, status: int) -> None:
    events = _events.get()
    if events is not None and not any(e["from"] == src and e["to"] == dst for e in events):
        events.append({"from": src, "to": dst, "status": status})


def fallback_note(events: List[dict]) -> str:
    """Markdown line for the end of an answer that was produced by a fallback model ('' when none was)."""
    if not events:
        return ""
    parts = [f"`{e['from']}` was refused (HTTP {e['status']}, usage credits/limit used up), so `{e['to']}` answered" for e in events]
    return "\n\n---\n_Model note: " + "; ".join(parts) + ". Quality may differ from the usual model._"


def _announce(src: str, dst: str, status: int, phase: str) -> None:
    msg = f"{src} refused (HTTP {status}, credits/usage): continuing on {dst}" + (f" [{phase}]" if phase else "")
    try:
        from services.notifier import notifier
        notifier.notify("Model fallback", msg, level="WARNING")
    except Exception:  # noqa: BLE001 - reporting never changes the call
        pass
    try:
        from services.action_log import log_action
        log_action("LLM", "ModelFallback", "quota fallback", "info", msg[:300], model=dst)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- the helper the call sites use
async def with_model_fallback(call: Callable[[str], Awaitable[T]], model: str, *, phase: str = "") -> T:
    """`await call(model)`. When the provider refuses it for credits / usage (402; 429 with a quota text):

      * settings.llm_fallback_model set  -> `await call(fallback)` once (and for settings.llm_quota_cooldown_s every
        later call for `model` goes straight to the fallback),
      * not set                          -> raises ModelQuotaError (readable message, mapped to HTTP 402 by the API).

    A timeout (ModelTimeoutError, HTTP 504) or an unreachable endpoint (ModelUnavailableError, HTTP 503) is raised as a readable
    error too, without a fallback (the fallback would most likely wait just as long). Any other failure propagates unchanged; so
    does cancellation. `call` receives the model name to use and must build a fresh client for it."""
    fb = fallback_for(model)
    status = _blocked_status(model) if fb else None
    if status is None:
        try:
            return await call(model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            quota = as_quota_error(exc, model)
            if quota is None:
                auth = as_auth_error(exc, model)
                if auth is not None:        # wrong / missing credentials: a readable error, not a traceback (a fallback would fail the same way)
                    raise auth from exc
                slow = as_timeout_error(exc, model)
                if slow is not None:        # a hang: say so (and which setting to raise) instead of a 500 with a 200-line traceback
                    _report_failure("Model timeout", str(slow), slow.model, phase)
                    raise slow from exc
                down = as_unavailable_error(exc, model)
                if down is not None:
                    _report_failure("Model unreachable", str(down), down.model, phase)
                    raise down from exc
                raise
            if not fb:
                raise quota from exc
            status = quota.status
            _block(model, status)
            _announce(model, fb, status, phase)
    _record(model, fb, status)
    try:
        return await call(fb)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the fallback failed too: say both facts
        raise ModelQuotaError(model, status, fallback_tried=fb, fallback_error=f"{type(exc).__name__}: {exc}"[:200]) from exc


# ---------------------------------------------------------------- API mapping
async def auth_error_response(_request, exc: ModelAuthError):
    """FastAPI exception handler: HTTP 401 + {"detail": <readable message>}."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=401, content={"detail": exc.user_message(), "error": "model_auth", "model": exc.model,
                                                  "status": exc.status})


async def quota_error_response(_request, exc: ModelQuotaError):
    """FastAPI exception handler: HTTP 402 + {"detail": <readable message>}. The web UI shows `detail` in its error toast."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=402, content={
        "detail": exc.user_message(), "error": "model_quota", "model": exc.model,
        "status": exc.status, "fallback_tried": exc.fallback_tried or None})


def timeout_payload(exc: ModelTimeoutError) -> Tuple[int, dict]:
    return 504, {"detail": exc.user_message(), "kind": "timeout", "runner": exc.runner, "model": exc.model, "timeout_s": exc.timeout_s}


def unavailable_payload(exc: ModelUnavailableError) -> Tuple[int, dict]:
    return 503, {"detail": exc.user_message(), "kind": "unavailable", "runner": exc.runner, "model": exc.model}


def internal_payload(exc: BaseException) -> Tuple[int, dict]:
    """Last-resort 500 body: "<ExceptionType>: <message>" (short, no traceback) so the UI always has something to show."""
    text = " ".join(str(exc).split())[:300]
    return 500, {"detail": f"{type(exc).__name__}: {text}" if text else type(exc).__name__, "kind": "internal"}


async def timeout_error_response(_request, exc: ModelTimeoutError):
    """FastAPI exception handler: HTTP 504 + {"detail", "kind": "timeout", "runner", "model", "timeout_s"}."""
    from fastapi.responses import JSONResponse
    status, body = timeout_payload(exc)
    return JSONResponse(status_code=status, content=body)


async def unavailable_error_response(_request, exc: ModelUnavailableError):
    """FastAPI exception handler: HTTP 503 + {"detail", "kind": "unavailable", "runner", "model"}."""
    from fastapi.responses import JSONResponse
    status, body = unavailable_payload(exc)
    return JSONResponse(status_code=status, content=body)


class InternalErrorMiddleware:
    """Pure ASGI safety net: an exception that no handler took becomes HTTP 500 {"detail": "Type: message", "kind": "internal"} and its
    traceback is logged ONCE (the default path logs it and answers a text/plain "Internal Server Error" the UI cannot show). It sits inside
    the server-error layer and outside FastAPI's exception handling, so HTTPException, request validation errors and the registered
    handlers (402 / 401 / 504 / 503) never reach it. Not applied once a response has started (streams): that error is re-raised."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        started = False

        async def tracking_send(message):
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:  # noqa: BLE001 - CancelledError is a BaseException and passes through
            if started:
                raise
            log.error("Unhandled error in %s %s", scope.get("method", ""), scope.get("path", ""), exc_info=exc)
            _report_failure("Request failed", f"{scope.get('path', '')}: {type(exc).__name__}: {exc}"[:300], "", "API")
            status, body = internal_payload(exc)
            payload = json.dumps(body).encode("utf-8")
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})
