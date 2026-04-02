def pytest_ignore_collect(collection_path, config):
    """Skip this directory entirely when DSP deps (soundfile) are not installed."""
    try:
        import soundfile  # noqa: F401
    except ImportError:
        return True
