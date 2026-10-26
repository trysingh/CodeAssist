# run.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # works from any working directory, on every OS

try:
    raise SystemExit(subprocess.run([sys.executable, str(ROOT / "engine.py"), "serve"], cwd=ROOT).returncode)
except KeyboardInterrupt:                   # Ctrl+C reaches both processes on macOS/Linux: exit quietly
    raise SystemExit(0)
