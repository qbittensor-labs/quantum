"""
Utilities for generating circuits
"""

from __future__ import annotations

import random
import time
import hashlib
from typing import Iterable, Tuple
import gc
import sys
import numpy as np

import bittensor as bt
import stim

from qbittensor.protocol import (
    ChallengePeakedCircuit,
    ChallengeHStabCircuit,
)
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import (
    CircuitParams,
)
from qbittensor.validator.hidden_stabilizers_creation.lib import make_gen
from qbittensor.validator.peaked_circuit_creation.quimb_cache_utils import (
    clear_all_quimb_caches,
)
from qbittensor.validator.hidden_stabilizers_creation.lib.circuit_gen import HStabCircuit

from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.utils.crypto_utils import canonical_hash

__all__ = [
    "build_peaked_challenge",
    "build_hstab_challenge",
]
def _clear_memory() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass
    try:
        clear_all_quimb_caches()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _convert_peaked_difficulty_to_qubits(level: float) -> int:
    try:
        lvl = float(level)
    except Exception:
        lvl = 0.0
    if 0.0 <= lvl <= 10.0:
        nqubits = int(12 + 10 * np.log2(lvl + 3.9))
    else:
        nqubits = int(round(lvl))
    return max(16, min(nqubits, 100))

# Peaked circuits

def build_peaked_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengePeakedCircuit, ChallengeMeta, str]:
    """
    Build a peaked circuit challenge
    """
    seed = time.time_ns() % (2**32)

    nqubits = _convert_peaked_difficulty_to_qubits(difficulty)
    level = _convert_qubits_to_peaked_difficulty(nqubits)

    rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
    rqc_depth = int(round(rqc_mul * nqubits))
    pqc_depth = max(1, nqubits // 5)
    params = CircuitParams(level, nqubits, rqc_depth, pqc_depth)
    entropy = 0.0
    try:
        circuit = params.compute_circuit(seed)
    finally:
        _clear_memory()

    unsigned = {
        "seed": seed,
        "circuit_data": circuit.to_qasm(),
        "difficulty_level": float(nqubits),
        "validator_hotkey": wallet.hotkey.ss58_address,
        "nqubits": nqubits,
        "rqc_depth": rqc_depth,
    }
    unsigned["challenge_id"] = canonical_hash(unsigned)

    syn = ChallengePeakedCircuit(**unsigned, validator_signature=None)
    meta = ChallengeMeta(
        challenge_id=unsigned["challenge_id"],
        circuit_kind="peaked",
        difficulty=float(nqubits),
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

    try:
        generator = make_gen(seed)
        circ: HStabCircuit = HStabCircuit.make_circuit(generator, nqubits)
        qasm = circ.to_qasm()
    finally:
        _clear_memory()
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

def _params_from_difficulty(level: float) -> Tuple[int, int]:
    """
    Interpret `level` as desired number of qubits for peaked circuits.
    Returns (nqubits, rqc_depth).
    """
    nqubits = int(round(level))
    nqubits = max(16, min(nqubits, 100))
    rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
    rqc_depth = int(round(rqc_mul * nqubits))
    return nqubits, rqc_depth


def _convert_qubits_to_peaked_difficulty(nqubits: int) -> float:
    """
    Inverse mapping for legacy difficulty → nqubits relation used by CircuitParams.
    nqubits ≈ 12 + 10 * log2(level + 3.9) ⇒ level ≈ 2 ** ((nqubits - 12) / 10) - 3.9
    """
    from math import log2
    n = max(2, int(nqubits))
    return max(0.0, (2 ** ((n - 12) / 10.0)) - 3.9)
