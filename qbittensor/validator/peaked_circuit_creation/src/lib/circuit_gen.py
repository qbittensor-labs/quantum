from __future__ import annotations

from math import ceil, sqrt
from typing import Optional

import numpy as np

from .optim import DEVICE
from .circuit import PeakedCircuit
from .circuit_meta import CircuitShape


def generate_circuit_by_qubits(
    *,
    nqubits: int,
    seed: int,
    target_peaking: Optional[float] = None,
    pqc_prop: float = 1.0,
    maxiters: int = 5000,
    epsilon: float = 1e-6,
) -> PeakedCircuit:
    """
    Generate a peaked circuit given a number of qubits.
      - sample circuit gates with `CircuitShape.sample_gates(seed)`
      - optimize to a peaked circuit with `PeakedCircuit.from_circuit(...)`

    Args:
        nqubits: Number of qubits (>= 2).
        seed: RNG seed driving circuit sampling and optimization.
        target_peaking: Optional peak ratio vs. uniform.
        pqc_prop: Proportion of gates in a tile to optimize as peaking.
        maxiters: Max GD iterations per tile.
        epsilon: Early stop threshold per tile.

    Returns:
        PeakedCircuit
    """
    if nqubits <= 1:
        raise ValueError("nqubits must be >= 2")

    # target peaking heuristic (see src/gen_circuits.do_gens):
    if target_peaking is None:
        a = 0.22582936781580765
        b = 0.37283850723802836
        target_peaking = 10 ** (a * nqubits + b + 0.5)

    target_prob = float(min(1.0, target_peaking / (2 ** nqubits)))
    depth = nqubits // 2
    tile_width = ceil(sqrt(nqubits))
    circuit = CircuitShape(nqubits, depth, tile_width).sample_gates(seed)

    try:
        return PeakedCircuit.from_circuit(
            circuit,
            target_prob,
            pqc_prop=pqc_prop,
            maxiters=maxiters,
            epsilon=epsilon,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to generate peaked circuit: "
            f"nqubits={nqubits}, seed={seed}, "
            f"target_peaking={target_peaking}, target_prob={target_prob:.6e}, "
            f"pqc_prop={pqc_prop}, maxiters={maxiters}, epsilon={epsilon}"
        ) from e


__all__ = ["generate_circuit_by_qubits", "DEVICE"]

