import os
import sys
import tempfile
from pathlib import Path

# Isolate storage BEFORE config.settings is imported anywhere.
_tmp = tempfile.mkdtemp(prefix="coderag_test_")
os.environ["ENGINE_STORAGE_DIR"] = _tmp
os.environ["ENGINE_ROUTER_MODEL"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
