"""Regressions from a real run on Windows with LLM_RUNNER=aistudio in .env (31 failures, 8.8 min, live Google calls)."""
import os

from config.settings import settings
from services.code_agent import clean_reply
from services.llm_errors import ModelUnavailableError, as_unavailable_error, server_error_status


def test_tests_never_read_the_developers_env_or_keys():
    assert os.environ.get("ENGINE_IGNORE_DOTENV") == "1"
    assert type(settings).model_config.get("env_file") is None
    assert settings.llm_runner == "ollama" and settings.embedding_runner == "ollama" and settings.token_mode == "normal"
    assert not settings.gemini_api_key and not settings.cf_api_token


def test_gemma_thought_block_from_ai_studio_is_not_shown():
    assert clean_reply("<thought>*   User Role: Expert...\n</thought>\nThe port is 8080.") == "The port is 8080."
    assert clean_reply("<thought>planning, cut off by the output limit") == ""
    assert clean_reply("Gemma writes <thought> tags sometimes.") == "Gemma writes <thought> tags sometimes."


def test_untagged_default_embedding_name_maps_to_the_runner_model(monkeypatch):
    from services.runners import get_runner

    class Rec:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.setattr(settings, "embedding_model", "nomic-embed-text")          # untagged, as in a real .env
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "cf_account_id", "a")
    monkeypatch.setattr(settings, "cf_api_token", "t")
    assert get_runner("aistudio").embeddings(settings.embedding_model, embeddings_cls=Rec).kw["model"] == settings.aistudio_embedding_model
    assert get_runner("cloudflare").embeddings(settings.embedding_model, embeddings_cls=Rec).kw["model"] == settings.cf_embedding_model
    assert get_runner("aistudio").embeddings("text-embedding-004", embeddings_cls=Rec).kw["model"] == "text-embedding-004"


class InternalServerError(Exception):           # same name as openai.InternalServerError
    status_code = 500


class Wrapped(Exception):                       # langchain_openai.OpenAIAPIError wraps it ("from e")
    pass


def test_provider_500_becomes_a_readable_503(monkeypatch):
    from services.runners import runner_scope
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    with runner_scope("aistudio"):
        _check_500()


def _check_500():
    try:
        try:
            raise InternalServerError("Error code: 500 - Internal error encountered.")
        except InternalServerError as e:
            raise Wrapped("Error code: 500") from e
    except Wrapped as exc:
        assert server_error_status(exc) == 500
        err = as_unavailable_error(exc, "gemma4:31b-cloud")
    assert isinstance(err, ModelUnavailableError) and err.status == 500
    assert "Google AI Studio returned a server error (HTTP 500)" in str(err) and "provider's side" in str(err)


def test_non_server_errors_are_not_called_unavailable():
    class NotFound(Exception):
        status_code = 404
    assert as_unavailable_error(NotFound("model not found"), "m") is None
