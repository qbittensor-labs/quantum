from types import SimpleNamespace
from unittest import mock
import numpy as np

from qbittensor.validator.forward import forward


class _FakeMetagraph:
    """
    Minimal stand-in that satisfies every attribute / method forward() touches.
    """

    def __init__(self):
        # _bootstrap calls `.uids.tolist()`
        self.uids = np.array([0])

        # needs to index axons[uid].is_serving
        self.axons = [mock.Mock(is_serving=True)]

        self.hotkeys = ["hk"]

    # forward() calls .sync() periodically
    def sync(self):
        pass


def _make_dummy_validator(tmp_db_path):
    v = SimpleNamespace()
    v.wallet = mock.Mock()
    v.metagraph = _FakeMetagraph()
    v.database_path = tmp_db_path
    return v


def test_forward_runs_once(tmp_db_path):
    v = _make_dummy_validator(tmp_db_path)

    # first call bootstraps
    forward(v)

    # second call should run the weight push without raising
    forward(v)
