import os
import sys
import tempfile
from pathlib import Path

# Isolate storage BEFORE config.settings is imported anywhere.
_tmp = tempfile.mkdtemp(prefix="coderag_test_")
os.environ["ENGINE_STORAGE_DIR"] = _tmp
os.environ["ENGINE_ROUTER_MODEL"] = ""
os.environ["ENGINE_ROUTER_AUTO"] = "true"                   # legacy router tests exercise the auto router; test_ask_agent.py covers the new default
os.environ["ENGINE_ASK_AGENT"] = "false"                   # legacy chat tests expect the single-shot answer; test_ask_agent.py opts in
os.environ["ENGINE_CHANGE_PLAN_FIRST"] = "false"       # scripted LLM tests expect one model call per patch; plan tests opt in
os.environ["ENGINE_CHANGE_REVIEW"] = "false"            # the reviewer is opt-in for tests, like the other stages
os.environ["ENGINE_CHANGE_TASKS"] = "false"             # the task loop is opt-in for tests, like the other stages
os.environ["ENGINE_CHANGE_CONTRACT"] = "false"          # scripted LLM tests expect no contract-designer call; those tests opt in
os.environ["ENGINE_CHANGE_AUTHOR_TESTS"] = "false"   # scripted LLM tests expect no extra test-author model call; those tests opt in
os.environ["ENGINE_CHANGE_GATE"] = "off"                 # gate tests opt in; the scripted pipeline tests run with the sandbox disabled
os.environ["ENGINE_CHANGE_PARTIAL_APPLY"] = "false"      # legacy scripted tests expect an all-or-nothing file; the repair tests opt in
os.environ["ENGINE_CHANGE_REPAIR_ATTEMPTS"] = "0"         # ... and expect every retry to use up an attempt
os.environ["ENGINE_CHANGE_DIAGNOSE"] = "false"            # ... and no extra diagnosis model call
os.environ["ENGINE_CHANGE_GAP_DECIDE"] = "false"          # ... and no targeted gap check
os.environ["ENGINE_WEB_VISUAL_FALLBACK"] = "off"         # legacy web tests never launch a browser; the quality/visual tests opt in
os.environ["ENGINE_WEB_OFFTOPIC_FILTER"] = "false"       # legacy web tests use filler page text; the quality tests opt in
os.environ["ENGINE_SANDBOX_ENABLED"] = "false"      # Phase 1 tests do not touch Docker; Phase 2 tests opt in explicitly
# ---- v2 defaults: legacy tests stay on the legacy (vector) path; v2 tests opt in with monkeypatch.setattr(settings, ...)
os.environ["ENGINE_REASONING_SUPERVISOR"] = "false"    # no intake/cross-check wrapper (and no delivery notes) around scripted answers
os.environ["ENGINE_ACTION_LOG_ENABLED"] = "true"       # logging itself is harmless; the ring buffer is what v2 tests read
os.environ["ENGINE_RETRIEVAL_MODE"] = "vector"         # legacy retrieval: vector search over code_chunks, no agentic switch
os.environ["ENGINE_INGEST_MODE"] = "vector"            # ingest_repository embeds before returning, as it always did
os.environ["ENGINE_VECTORIZE_FALLBACK"] = "off"        # never enqueue background jobs behind a legacy test's back
os.environ["ENGINE_REPEAT_REQUEST_GUARD"] = "false"    # v3.2: legacy tests may repeat a request on purpose; the v3.2 tests opt in
os.environ["ENGINE_WATCH_ON_INGEST"] = "false"         # no file-system observers started by ingest_repository in tests
# ---- runners: tests never read the developer's .env and never reach a real provider. ENGINE_* names win over the plain aliases
# (LLM_RUNNER, GEMINI_API_KEY, ...) a shell may export. Runner tests opt in with monkeypatch.setattr(settings, ...).
os.environ["ENGINE_IGNORE_DOTENV"] = "1"
os.environ["ENGINE_LLM_RUNNER"] = "ollama"
os.environ["ENGINE_EMBEDDING_RUNNER"] = "ollama"
os.environ["ENGINE_TOKEN_MODE"] = "normal"
os.environ["ENGINE_LLM_MODEL"] = "gemma4:31b-cloud"
os.environ["ENGINE_EMBEDDING_MODEL"] = "nomic-embed-text:latest"
for _k in ("GEMINI_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN", "CF_AI_GATEWAY_URL", "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY",
           "OPENAI_COMPAT_MODELS", "OPENAI_COMPAT_LLM_MODEL", "OPENAI_COMPAT_EMBEDDING_MODEL", "LLM_FALLBACK_MODEL",
           "AISTUDIO_LLM_MODEL", "AISTUDIO_VISION_MODEL", "AISTUDIO_MODELS", "CF_LLM_MODEL", "CF_VISION_MODEL", "CF_MODELS",
           "AISTUDIO_REASONING_EFFORT", "ASK_REASONING"):
    os.environ.pop(_k, None)                                # plain alias from the shell
    os.environ.pop("ENGINE_" + _k, None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))