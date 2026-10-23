import os
import sys
import tempfile
from pathlib import Path

# Isolate storage BEFORE config.settings is imported anywhere.
_tmp = tempfile.mkdtemp(prefix="coderag_test_")
os.environ["ENGINE_STORAGE_DIR"] = _tmp
os.environ["ENGINE_ROUTER_MODEL"] = ""
os.environ["ENGINE_CHANGE_PLAN_FIRST"] = "false"       # scripted LLM tests expect one model call per patch; plan tests opt in
os.environ["ENGINE_CHANGE_AUTHOR_TESTS"] = "false"   # scripted LLM tests expect no extra test-author model call; those tests opt in
os.environ["ENGINE_SANDBOX_ENABLED"] = "false"      # Phase 1 tests do not touch Docker; Phase 2 tests opt in explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))