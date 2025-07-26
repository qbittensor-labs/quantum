from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
from stim import PauliString

from .base import HStabSolver, QState

def _is_power_of_2(n: int) -> Optional[int]:
    """
    If `n` is a power of 2, return the exponent; otherwise return `None`.
    """
    exp = int(np.log2(n))
    if 2 ** exp == n:
        return exp
    else:
        return None

class AerStateVector(QState):
    """
    Holds a state vector numpy array produced by a Qiskit Aer simulator (i.e. in
    *little-endian*).

    Fields:
        nqubits (int > 0):
            The number of qubits.
        array (numpy.ndarray[complex, 1]):
            The state vector.
    """
    nqubits: int
    array: np.ndarray[complex, 1]

    def __init__(self, state_vector: np.ndarray[complex, 1]):
        """
        Initialize with a full state vector array. The number of elements must
        be a power of 2.

        Args:
            state_vector (numpy.ndarray[complex, 1]):
                The state vector.

        Raises:
            ValueError:
                - `len(state_vector)` is not a power of 2
        """
        self.nqubits = _is_power_of_2(len(state_vector))
        if self.nqubits is None:
            raise ValueError("input state vector length is not a power of 2")
        self.array = state_vector

    def num_qubits(self) -> int:
        return self.nqubits

    def get_expectation_val(self, npauli: PauliString) -> float:
        op = npauli.to_unitary_matrix(endian="little")
        return np.conjugate(self.array) @ op @ self.array

class AerSolverStateVector(HStabSolver):
    """
    Simple solver using the Qiskit Aer state vector backend.

    Fields:
        device (str):
            Device to use ('CPU', 'GPU').
        backend (qiskit_aer.AerSimulator):
            Qiskit Aer backend.
    """
    device: str
    backend: AerSimulator

    def __init__(self, device: str = "CPU", **kwargs):
        """
        Initialize the Qiskit Aer state vector simulator. Simulator method is
        fixed to 'statevector'

        Args:
            device (str):
                Device to use ('CPU', 'GPU')
            **kwargs:
                Additional backend options to pass to `qiskit_aer.AerSimulator`.
        """
        self.device = device
        self.backend = AerSimulator(
            method="statevector", device=device, **kwargs)

    def run_statevector(self, qasm: str) -> Optional[np.ndarray[complex, 1]]:
        """
        Run a simple state vector simulation and return the output state vector.
        The state vector is returned in *little-endian* order.

        Args:
            qasm (str):
                QASM circuit string.

        Returns:
            statevector (Optional[numpy.ndarray[complex, 1]]):
                State vector as a complex-valued 1D array, in *little-endian*
                order.
        """
        try:
            circuit = QuantumCircuit.from_qasm_str(qasm)
            circuit.save_statevector()
            job = self.backend.run(circuit, shots=1)
            result = job.result()
            statevector = result.data(0)["statevector"]
            return np.array(statevector)
        except Exception as e:
            raise RuntimeError(f"failed to get state vector: {str(e)}")

    def run_other(self, qasm: str) -> AerStateVector:
        """
        Same as `self.run_statevector`, so that `AerStateVector` can be used
        with `.base.find_stabilizers`.

        Args:
            qasm (str):
                QASM circuit string.

        Returns:
            statevector (.base.StateVector):
                State vector as a complex-valued 1D array, in *little-endian*
                order.
        """
        statevector = self.run_statevector(qasm)
        return AerStateVector(statevector)

