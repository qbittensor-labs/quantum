from typing import Protocol

import numpy as np


class CircuitParams(Protocol):
    nqubits: int
    rqc_depth: float


def entanglement_entropy(circuit: CircuitParams) -> float:
    """
    Entanglement entropy of a random-quantum-circuit.

    Returns:
        S ∈ [0, Smax]
    """
    ln2 = np.log(2)
    nqubits = circuit.nqubits
    rqc = circuit.rqc_depth

    Smax = (nqubits / 2) * ln2 - ln2**2
    return Smax * (1 - np.exp(-rqc / (nqubits * ln2)))
