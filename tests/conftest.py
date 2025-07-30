import asyncio, sqlite3, tempfile, os
import pytest
from unittest import mock
from qbittensor.validator.reward import ScoringManager


@pytest.fixture(scope="session")
def event_loop():
    """Pytest runs async tests on its own loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def gpu_patch():
    """Pretend there are 2 GPUs so code paths relying on _GPU_COUNT run."""
    with mock.patch("torch.cuda.device_count", return_value=2):
        yield


@pytest.fixture
def tmp_db_path(tmp_path):
    """Creates a temp SQLite file and passes its str path."""
    db_file = tmp_path / "validator_data.db"
    yield str(db_file)               # production code wants a str path
    sqlite3.connect(db_file).close() # cleanliness


@pytest.fixture
def scoring_mgr(tmp_db_path):
    return ScoringManager(database_path=tmp_db_path)
