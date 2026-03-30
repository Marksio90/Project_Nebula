"""
tests/fixtures/conftest.py
Root pytest conftest — adds the repo root to sys.path so that
`from services.dsp_worker.audio.beat_aligner import ...` works
in unit tests without installing the packages.
"""
import sys
from pathlib import Path

# Repo root
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
