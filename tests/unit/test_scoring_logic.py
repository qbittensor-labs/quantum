from qbittensor.validator.reward import size_function, ScoringManager
import math


def test_size_function_regions():
    # floor
    assert size_function(8) == 0.1
    # linear: halfway from 12→32 should be halfway 0.1→0.4
    assert size_function(22) == 0.25
    # exponential kicks in
    knee_val = size_function(32)
    forty_val = size_function(40)
    assert forty_val > knee_val * math.pow(1.7, 8) * 0.99


def test_combined_score_calculation(tmp_path):
    mgr = ScoringManager(database_path=str(tmp_path / "dummy.db"))
    ee, size, combined = mgr.calculate_combined_score(1.5, 20)
    assert 0.0 <= ee <= 1.0
    assert 0.0 < size < 1.0
    assert combined == mgr.weight_ee * ee + (1 - mgr.weight_ee) * size
