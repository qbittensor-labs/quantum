"""
Utilities for generating circuits
"""

from __future__ import annotations

import gc
import random
import time
import hashlib
from typing import Iterable, Tuple

import bittensor as bt
import stim

from qbittensor.protocol import (
    ChallengePeakedCircuit,
    ChallengeHStabCircuit,
)
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams
from qbittensor.validator.peaked_circuit_creation.quimb_cache_utils import (
    clear_all_quimb_caches,
)
from qbittensor.validator.hidden_stabilizers_creation.lib import make_gen
from qbittensor.validator.hidden_stabilizers_creation.lib.circuit_gen import HStabCircuit

from qbittensor.validator.utils.entanglement_entropy import entanglement_entropy
from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.utils.crypto_utils import canonical_hash

__all__ = [
    "build_peaked_challenge",
    "build_hstab_challenge",
]

# Peaked circuits

def build_peaked_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengePeakedCircuit, ChallengeMeta, str]:
    """
    Build a peaked circuit challenge
    """
    seed = time.time_ns() % (2**32)
    params = CircuitParams.from_difficulty(difficulty)
    entropy = entanglement_entropy(params)

    nqubits: int = getattr(params, "nqubits", getattr(params, "num_qubits", 0))
    rqc_depth: int = getattr(params, "rqc_depth", getattr(params, "depth", 0))

    circuit = params.compute_circuit(seed)
    unsigned = {
        "seed": seed,
        "circuit_data": circuit.to_qasm(),
        "difficulty_level": difficulty,
        "validator_hotkey": wallet.hotkey.ss58_address,
        "nqubits": nqubits,
        "rqc_depth": rqc_depth,
    }
    unsigned["challenge_id"] = canonical_hash(unsigned)

    syn = ChallengePeakedCircuit(**unsigned, validator_signature=None)
    meta = ChallengeMeta(
        challenge_id=unsigned["challenge_id"],
        circuit_kind="peaked",
        difficulty=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        entanglement_entropy=entropy,
        nqubits=nqubits,
        rqc_depth=rqc_depth,
    )

    return syn, meta, circuit.target_state


# Hidden Stabiliser circuits

def build_hstab_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengeHStabCircuit, ChallengeMeta, str]:
    """
    Build an H-Stab circuit challenge.
    """
    nqubits: int = max(26, round(difficulty))
    seed = random.randrange(1 << 30)

    generator = make_gen(seed)
    circ: HStabCircuit = HStabCircuit.make_circuit(generator, nqubits)
    qasm = circ.to_qasm()
    flat_solution = _flatten_hstab_string(circ.stabilizers)
    cid = hashlib.sha256(qasm.encode()).hexdigest()

    syn = ChallengeHStabCircuit(
        circuit_data=qasm,
        challenge_id=cid,
        difficulty_level=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        validator_signature=None,
    )
    meta = ChallengeMeta(
        challenge_id=cid,
        circuit_kind="hstab",
        difficulty=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        entanglement_entropy=0.0,
        nqubits=nqubits,
        rqc_depth=0,
    )
    return syn, meta, flat_solution


# Helper
def _flatten_hstab_string(stabilizers: Iterable[stim.PauliString]) -> str:
    """flatten an iterable of `stim.PauliString` -> single str"""
    return "".join(map(str, stabilizers))

