from typing import Any, Dict
import numpy as np
from stim import Tableau
from .base import TaskProcessor


class HStabCircuitProcessor(TaskProcessor):
    def process(self, statevector: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Process statevector to extract stabilizers.

        Args:
            statevector: Final quantum state as numpy array
            **kwargs: Additional parameters (unused for this processor)

        Returns:
            Dictionary containing:
            - 'stabilizers': List of stablizers
            - 'num_qubits': Number of qubits in the circuit
            - 'success': Boolean indicating if processing was successful
        """
        try:
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
