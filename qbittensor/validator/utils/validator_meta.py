"""
Lightweight container for information about a circuit synapse.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChallengeMeta:
    challenge_id: str
    circuit_kind: str
    difficulty: float
    validator_hotkey: str
    entanglement_entropy: float
    nqubits: int
    rqc_depth: int
