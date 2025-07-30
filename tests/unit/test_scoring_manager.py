def test_normalize_ee_bounds(scoring_mgr):
    assert scoring_mgr.normalize_ee(0.0, 20) == 0.0
    assert scoring_mgr.normalize_ee(999, 2) == 1.0


def test_single_solution_score(scoring_mgr):
    score = scoring_mgr.calculate_single_solution_score(
        entropy=4.2, nqubits=24, is_correct=True
    )
    assert 0.0 < score < 1.0
