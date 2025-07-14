"""
Helpers for creating ChallengeCircuits synapses and their metadata.
"""

from __future__ import annotations

import gc
import time
from typing import Tuple

import bittensor as bt
import torch

from qbittensor.protocol import ChallengeCircuits
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams
from qbittensor.validator.peaked_circuit_creation.quimb_cache_utils import clear_all_quimb_caches
from qbittensor.validator.utils.entanglement_entropy import entanglement_entropy

from .crypto_utils import canonical_hash
from .validator_meta import ChallengeMeta


# build one challenge
def build_challenge(*, wallet: bt.wallet, difficulty: float) -> Tuple[ChallengeCircuits, ChallengeMeta, str]:
    seed = time.time_ns() % (2**32)

    params = CircuitParams.from_difficulty(difficulty)
    entropy = entanglement_entropy(params)

    nqubits = int(getattr(params, "nqubits", getattr(params, "num_qubits", 0)))
    rqc_depth = int(getattr(params, "rqc_depth", getattr(params, "depth", 0)))

    circuit = params.compute_circuit(seed)

    unsigned = {
        "seed": seed,
        "circuit_data": circuit.to_qasm(),
        "difficulty_level": difficulty,
        "validator_hotkey": wallet.hotkey.ss58_address,
        "nqubits": nqubits,
        "rqc_depth": rqc_depth,
    }
    challenge_id = canonical_hash(unsigned)
    unsigned["challenge_id"] = challenge_id

    syn = ChallengeCircuits(**unsigned)
    syn.validator_signature = None
    meta = ChallengeMeta(
        challenge_id=challenge_id,
        difficulty=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        entanglement_entropy=entropy,
        nqubits=nqubits,  # <── NEW
        rqc_depth=rqc_depth,  # <── NEW
    )

    clear_all_quimb_caches()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        bt.logging.debug("Cleared CUDA cache")

    collected = 0
    for _ in range(3):
        collected += gc.collect()

    bt.logging.info(f"Memory cleanup complete: collected {collected} objects")

    return syn, meta, circuit.target_state
