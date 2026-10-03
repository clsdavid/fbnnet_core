import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Several tests write scratch files (example.bn, gene lists) to the working directory."""
    monkeypatch.chdir(tmp_path)
