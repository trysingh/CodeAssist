"""Cached-token accounting across runners: each runner's usage shape -> telemetry -> ledger -> footer; every chat call counted."""
import io
import contextlib
from types import SimpleNamespace as NS

import pytest

from config.settings import settings
from services.telemetry import StdoutTelemetryCallback
from services.token_budget import footer_text, ledger_scope, token_mode_scope
from services.runners import make_chat
from services.runners.scope import runner_scope, thread_scope


def _openai(prompt, completion, extra=None):
    """Shape built by langchain_openai: usage_metadata (+cache_read when the server sent cached_tokens) and the raw usage."""
    raw = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, **(extra or {})}
    um = {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion}
    cached = ((extra or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        um["input_token_details"] = {"cache_read": cached}
    msg = NS(usage_metadata=um, response_metadata={"token_usage": raw})
    return NS(generations=[[NS(message=msg, generation_info={}, text="ok")]], llm_output={"token_usage": raw})


def _ollama(prompt_eval, eval_count):
    msg = NS(usage_metadata={"input_tokens": prompt_eval, "output_tokens": eval_count}, response_metadata={})
    return NS(generations=[[NS(message=msg, generation_info={"prompt_eval_count": prompt_eval, "eval_count": eval_count},
                              text="ok")]], llm_output=None)


def _run(runner, results, mode="auto"):
    with contextlib.redirect_stdout(io.StringIO()), runner_scope(runner), token_mode_scope(mode), \
            thread_scope("r", "t"), ledger_scope() as led:
        for r in results:
            cb = StdoutTelemetryCallback(model_name="m", repo_id="r", phase="ASK_AGENT")
            cb.on_chat_model_start({}, [[NS(content="hi")]])
            cb.on_llm_end(r)
        return led.totals(), footer_text(led.totals())


@pytest.mark.parametrize("runner,extra,cached", [
    ("cloudflare", {"prompt_tokens_details": {"cached_tokens": 5900}}, 5900),
    ("cloudflare", {"cached_tokens": 5900}, 5900),
    ("aistudio", {"prompt_tokens_details": {"cached_tokens": 4096}}, 4096),
    ("openai", {"prompt_tokens_details": {"cached_tokens": 2048}}, 2048),
])
def test_cached_tokens_reach_ledger_and_footer(runner, extra, cached):
    totals, foot = _run(runner, [_openai(9000, 250, extra)])
    assert totals["cached_input_tokens"] == cached and totals["input_tokens"] == 9000
    assert "cached)" in foot and f"runner {runner}" in foot


def test_no_cache_field_means_no_cached_claim():
    for runner, res in (("ollama", [_ollama(5000, 300)]), ("aistudio", [_openai(7000, 400)])):
        totals, foot = _run(runner, res)
        assert totals["cached_input_tokens"] == 0 and "cached" not in foot


def test_cached_never_exceeds_input():
    totals, _ = _run("cloudflare", [_openai(100, 10, {"prompt_tokens_details": {"cached_tokens": 500}})])
    assert totals["cached_input_tokens"] == 100


class _Fake:
    def __init__(self, **kw):
        self.kw = kw


def test_make_chat_adds_telemetry_and_labels_the_real_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "k", raising=False)
    with runner_scope("aistudio"):
        from services.runners import get_runner
        runner = get_runner()
        monkeypatch.setattr(type(runner), "chat", lambda self, model, **kw: _Fake(model=model, **kw))
        llm = make_chat("gemma4:31b-cloud", temperature=0, phase="WEB_REWRITE")
    cbs = llm.kw["callbacks"]
    assert cbs and isinstance(cbs[0], StdoutTelemetryCallback)
    assert cbs[0].phase == "WEB_REWRITE" and cbs[0].model_name == "gemma-4-31b-it"


def test_session_affinity_stable_per_thread(monkeypatch):
    monkeypatch.setattr(settings, "cf_account_id", "acc")
    monkeypatch.setattr(settings, "cf_api_token", "tok")
    from services.runners.cloudflare_runner import CloudflareRunner
    hdr = []
    for thr in (("r", "t1"), ("r", "t1"), ("r", "t2")):
        with runner_scope("cloudflare"), thread_scope(*thr):
            hdr.append(CloudflareRunner().chat("x:y", chat_cls=_Fake).kw["default_headers"]["x-session-affinity"])
    assert hdr[0] == hdr[1] != hdr[2]
