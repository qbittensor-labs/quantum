import os
from pathlib import Path
import numpy as np

from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams
from qbittensor.validator.peaked_circuit_creation.lib.circuit import SU4
from qbittensor.validator.peaked_circuit_creation.lib.base_cache import (
    save_base_su4, load_base_su4, CACHE_DIR,
)


def _clean_cache_for(nq: int, rqc: int, pqc: int, seed: int):
    key = f"nq{nq}_r{rqc}_p{pqc}_seed{int(seed)}.npz"
    fp = CACHE_DIR / key
    if fp.exists():
        fp.unlink()


def _identity_brickwork(nqubits: int) -> list[SU4]:
    mats = []

    I4 = np.eye(4, dtype=np.complex128)
    unis: list[SU4] = []
    for q in range(0, nqubits - 1, 2):
        unis.append(SU4(q, q + 1, I4.copy()))
    for q in range(1, nqubits - 1, 2):
        unis.append(SU4(q, q + 1, I4.copy()))
    return unis


def test_compute_circuit_uses_cached_base_target():
    params = CircuitParams.from_difficulty(-3.5)
    nq = params.nqubits
    seed = 13579
    _clean_cache_for(nq, params.rqc_depth, params.pqc_depth, seed)

    base_target = ("10" * ((nq + 1) // 2))[:nq]
    unis = _identity_brickwork(nq)
    save_base_su4(
        nqubits=nq,
        rqc_depth=params.rqc_depth,
        pqc_depth=params.pqc_depth,
        seed=seed,
        target_state=base_target,
        unis=unis,
        peak_prob=1.0,
        ttl_hours=48,
    )

    circ = params.compute_circuit(seed)
    assert circ.num_qubits == nq
    assert circ.target_state == base_target


def test_compute_circuits_uses_cached_base_and_generates_variants():
    params = CircuitParams.from_difficulty(-3.5)
    nq = params.nqubits
    seed = 97531
    _clean_cache_for(nq, params.rqc_depth, params.pqc_depth, seed)

    base_target = ("01" * ((nq + 1) // 2))[:nq]
    unis = _identity_brickwork(nq)
    save_base_su4(
        nqubits=nq,
        rqc_depth=params.rqc_depth,
        pqc_depth=params.pqc_depth,
        seed=seed,
        target_state=base_target,
        unis=unis,
        peak_prob=1.0,
        ttl_hours=48,
    )

    circuits = params.compute_circuits(seed, n_variants=3)
    assert len(circuits) == 3
    assert circuits[0].target_state == base_target
    obf_targets = [c.target_state for c in circuits[1:]]
    assert any(t != base_target for t in obf_targets)


