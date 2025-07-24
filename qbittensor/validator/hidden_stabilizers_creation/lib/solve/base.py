from abc import abstractmethod
from itertools import product
from typing import Any, Callable, List, Optional, Self

import numpy as np
from stim import PauliString, Tableau

class QState:
    """
    Base class for a data type supporting evaluation of N-qubit Pauli
    expectation values on a quantum state, for use with `find_stabilizers`.
    """

    @abstractmethod
    def num_qubits(self) -> int:
        """
        Return the number of qubits in the state.
        """

    @abstractmethod
    def get_expectation_val(self, npauli: PauliString) -> float:
        """
        Compute the expectation value of an N-qubit Pauli operator.

        Args:
            npauli (stim.PauliString):
                The N-qubit Pauli operator to evaluate.

        Returns:
            expval (float):
                The expectation value of `npauli`.
        """

class HStabSolver:
    """
    Base class for a hidden stabilizers solver.
    """

    @abstractmethod
    def run_statevector(self, qasm: str) -> Optional[np.ndarray[complex, 1]]:
        """
        Run a QASM circuit and return the output state. Should return `None` if
        state vector simulation is not supported.

        Args:
            qasm (str):
                QASM circuit string.

        Returns:
            statevector (Optional[numpy.ndarray[complex, 1]]):
                State vector as a complex-valued 1D array, or `None` if state
                vector simulation is not supported for the circuit.
        """

    @abstractmethod
    def run_other(self, qasm: str) -> QState:
        """
        Run a QASM circuit with an arbitrary method, returning the quantum state
        in a structure that implements `QState`.

        Args:
            qasm (str):
                QASM circuit string.

        Returns:
            state (QState):
                The quantum state.
        """

def find_stabilizers(
    state: QState,
    epsilon: float = 1e-6,
) -> Optional[List[PauliString]]:
    """
    Mostly brute-force search through the set of all N-qubit Pauli operators to
    find a commuting set that stabilize `state`. Stabilizers are returned in
    canonical form.

    Args:
        state (QState):
            Quantum state object.
        epsilon (float):
            Tolerance for detecting specific floating-point values close to +1
            or -1.

    Returns:
        stabilizers (Optional[List[PauliString]]):
            Commuting set of stabilizer generators for `state`, in canonical
            form. `None` is returned if such a set could not be found.

    Raises:
        ValueError:
            - `state.num_qubits()` returns a number less than 1
    """
    nqubits = state.num_qubits()
    if nqubits < 1:
        raise ValueError(f"expected at least 1 qubit, got {nqubits}")
    stabs = list()
    pauli_iter = product(*(nqubits * ["IXYZ"]))
    _ = next(pauli_iter) # discard the identity
    stab_count = 0
    for ops in pauli_iter:
        if stab_count >= nqubits:
            tab = Tableau.from_stabilizers(stabs)
            return tab.to_stabilizers(canonicalize=True)

        npauli = PauliString("+" + "".join(ops))
        # check first for commutativity and independence by trying to construct
        # a stim tableau -- this is probably faster than any checks we can do
        # natively in Python; critically, this is definitely faster than
        # computing an expectation value
        try:
            Tableau.from_stabilizers(
                stabs + [npauli], allow_underconstrained=True)
        except ValueError:
            continue

        expval = state.get_expectation_val(npauli)
        if abs(expval + 1) <= epsilon:
            npauli = -npauli # if expval == -1 then -npauli is a stabilizer
        elif abs(expval - 1) > epsilon:
            continue

        stabs.append(npauli)
        stab_count += 1
    return None

