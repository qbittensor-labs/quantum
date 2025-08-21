def test_single_solution_score(scoring_mgr):
    score = scoring_mgr.calculate_single_solution_score(
        entropy=4.2, nqubits=24, is_correct=True
    )
    assert 0.0 < score < 1.0
