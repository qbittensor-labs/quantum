from types import SimpleNamespace
import time
from qbittensor.validator.services.weight_manager import WeightManager


def test_weight_manager_update_triggers(monkeypatch):
    """Intervals elapsed -> both private methods should run."""
    wm = WeightManager(SimpleNamespace())
    flags = {"score": 0, "weight": 0}

    monkeypatch.setattr(wm, "_update_scoring", lambda _ts: flags.__setitem__("score", 1))
    monkeypatch.setattr(wm, "_set_weights", lambda _ts: flags.__setitem__("weight", 1))

    # Force last run times far in the past
    wm.last_scoring_time = 0
    wm.last_weight_time = 0

    wm.update()
    assert flags == {"score": 1, "weight": 1}


def test_weight_manager_no_retrigger(monkeypatch):
    """Intervals NOT elapsed -> neither private method should run."""
    wm = WeightManager(SimpleNamespace())
    flags = {"score": 0, "weight": 0}

    monkeypatch.setattr(wm, "_update_scoring", lambda _ts: flags.__setitem__("score", 1))
    monkeypatch.setattr(wm, "_set_weights", lambda _ts: flags.__setitem__("weight", 1))

    # Pretend both ran just now
    current = time.time()
    wm.last_scoring_time = current
    wm.last_weight_time = current

    wm.update()
    assert flags == {"score": 0, "weight": 0}
