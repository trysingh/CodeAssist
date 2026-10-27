import os
import sys
import tempfile
from pathlib import Path

# Isolate storage BEFORE config.settings is imported anywhere.
_tmp = tempfile.mkdtemp(prefix="coderag_test_")
os.environ["ENGINE_STORAGE_DIR"] = _tmp
os.environ["ENGINE_ROUTER_MODEL"] = ""
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
os.environ["ENGINE_SANDBOX_ENABLED"] = "false"      # Phase 1 tests do not touch Docker; Phase 2 tests opt in explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))