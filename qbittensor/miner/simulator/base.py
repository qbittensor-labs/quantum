from typing import Any, Dict, Optional


class QuantumSimulator:
    """Base circuit simulator interface"""

    def run(self, qasm: str, shots: int = 1024) -> Dict[str, int]:
        """
        Run a QASM circuit and return measurement counts

        Args:
            qasm: QASM string
            shots: Number of measurement shots

        Returns:
            Dictionary mapping outputs to counts

        Example:
            >>> counts = simulator.run(qasm_string, shots=1024)
            >>> counts
            {'00': 512, '11': 512}
        """
        raise NotImplementedError("Subclasses must implement run()")

    def get_info(self) -> Dict[str, Any]:
        """
        Optional: Return backend capabilities/info

        Returns:
            Dictionary with backend information
        """
        return {}

    def get_statevector(self, qasm: str) -> Optional[Any]:
        """
        Optional: Get the statevector for backends that support it

        Args:
            qasm: QASM string representation of circuit

        Returns:
            Statevector or None if not supported
        """
        return None
