from pathlib import Path

_THIS_DIR = Path(__file__).parent


def pytest_ignore_collect(collection_path, config):
    """Skip this directory entirely when DSP deps (soundfile) are not installed."""
    try:
        import soundfile  # noqa: F401
    except ImportError:
        # Guard: only ignore paths inside this directory, not the entire suite
        try:
            Path(collection_path).relative_to(_THIS_DIR)
            return True
        except ValueError:
            return None
