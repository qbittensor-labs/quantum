import json
from unittest import mock

import pytest

from qbittensor.validator.services import solution_extractor as se_mod
from qbittensor.validator.services.solution_extractor import SolutionExtractor


class DummySynapse:
    """
    Minimal stand-in for ChallengePeakedCircuit / ChallengeHStabCircuit.
    Only attributes that SolutionExtractor touches are implemented.
    """

    def __init__(self, challenge_id: str, bitstring: str, circuit_kind: str = "peaked"):
        self.challenge_id = challenge_id
        self.solution_bitstring = bitstring
        self.circuit_kind = circuit_kind
        self.validator_hotkey = "validator-hk"


@pytest.fixture(autouse=True)
def _patch_recognised_classes(monkeypatch):
    """
    Make DummySynapse satisfy the `isinstance` checks inside SolutionExtractor.
    """
    for cls_name in (
        "ChallengeCircuits",
        "ChallengePeakedCircuit",
        "ChallengeHStabCircuit",
    ):
        monkeypatch.setattr(se_mod, cls_name, DummySynapse, raising=False)


def test_extract_batch_json():
    payload = {
        "type": "batch",
        "solutions": [
            {"challenge_id": "c‑1", "solution_bitstring": "AAA"},
            {"challenge_id": "c‑2", "solution_bitstring": "BBB"},
        ],
    }

    syn = DummySynapse("batch‑cid", json.dumps(payload))
    sols = SolutionExtractor.extract(syn)

    assert {s.challenge_id for s in sols} == {"c‑1", "c‑2"}


def test_extract_single_solution():
    syn = DummySynapse("single‑cid", "XYZ")
    sols = SolutionExtractor.extract(syn)

    assert len(sols) == 1
    sol = sols[0]
    assert sol.challenge_id == "single‑cid"
    assert sol.solution_bitstring == "XYZ"
