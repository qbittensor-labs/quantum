from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from stim import PauliString, Tableau

from .base import TaskProcessor


class QState:
    def __init__(self, statevector: np.ndarray) -> None:
        self.nqubits = int(np.log2(len(statevector)))
        self.array = statevector

    def num_qubits(self) -> int:
        return self.nqubits

    def get_expectation_val(self, npauli: PauliString) -> float:
        op = npauli.to_unitary_matrix(endian="little")
        return np.real(np.conjugate(self.array) @ op @ self.array)


def find_stabilizers(state: QState, epsilon: float = 1e-6) -> Optional[List[PauliString]]:
    """
    search through N-qubit Pauli operators to find stabilizers.
    """
    nqubits = state.num_qubits()
    if nqubits < 1:
        raise ValueError(f"expected at least 1 qubit, got {nqubits}")

    stabs: List[PauliString] = []
    pauli_iter = product(*(nqubits * ["IXYZ"]))
    _ = next(pauli_iter)
    stab_count = 0

    for ops in pauli_iter:
        if stab_count >= nqubits:
            tab = Tableau.from_stabilizers(stabs)
            return tab.to_stabilizers(canonicalize=True)

        npauli = PauliString("+" + "".join(ops))

        try:
            Tableau.from_stabilizers(stabs + [npauli], allow_underconstrained=True)
        except ValueError:
            continue

        expval = state.get_expectation_val(npauli)
        if abs(expval + 1) <= epsilon:
            npauli = -npauli
        elif abs(expval - 1) > epsilon:
            continue

        stabs.append(npauli)
        stab_count += 1

    return None


class HStabCircuitProcessor(TaskProcessor):
    def process(self, statevector: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Process statevector to extract stabilizers.

        Args:
            statevector: Final quantum state as numpy array
            **kwargs: Additional parameters (unused for this processor)

        Returns:
            Dictionary containing:
            - 'stabilizers': List of stabilizers
            - 'num_qubits': Number of qubits in the circuit
            - 'success': Boolean indicating if processing was successful
        """
        try:
            if not isinstance(statevector, np.ndarray):
                raise TypeError("Expected numpy array for statevector")

            tableau = Tableau.from_state_vector(statevector, endian="little")
            stabilizers = tableau.to_stabilizers(canonicalize=True)
            num_qubits = int(np.log2(len(statevector)))

            return {
                "stabilizers": stabilizers,
                "num_qubits": num_qubits,
                "success": True,
            }

        except Exception as e:
            return {
                "stabilizers": [],
                "num_qubits": 0,
                "success": False,
                "error": str(e),
            }

    def process_alt_method(self, statevector: np.ndarray, epsilon: float = 1e-6) -> Dict[str, Any]:
        """
        Process statevector using alternate method to extract stabilizers.
        """
        try:
            state = QState(statevector)
            stabilizers = find_stabilizers(state, epsilon)
            num_qubits = state.num_qubits()

            if stabilizers is None:
                return {"stabilizers": [], "num_qubits": num_qubits, "success": False, "error": "Search failed to find all stabilizers"}

            return {
                "stabilizers": stabilizers,
                "num_qubits": num_qubits,
                "success": True,
            }

        except Exception as e:
            return {
                "stabilizers": [],
                "num_qubits": 0,
                "success": False,
                "error": str(e),
            }

    def validate_result(self, result: Dict[str, Any]) -> bool:
        if not result.get("success", False):
            return False

        stabilizers = result.get("stabilizers", [])
        num_qubits = result.get("num_qubits", 0)

        if len(stabilizers) != num_qubits:
            return False
        try:
            for stab in stabilizers:
                str(stab)

        except Exception:
            return False

        return True
