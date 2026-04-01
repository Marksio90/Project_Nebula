"""
tests/conftest.py
Root pytest conftest — adds the repo root to sys.path so that
`from services.dsp_worker.audio.beat_aligner import ...` works
in all unit and integration tests without installing packages.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent  # tests/ → Project_Nebula/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
