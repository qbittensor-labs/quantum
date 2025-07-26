from typing import Any, Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, U3Gate
from qiskit.qasm2 import CustomInstruction, loads
from qiskit_aer import AerSimulator

from .base import QuantumSimulator


def _detect_gpu_device() -> str:
    """Detect if GPU is available for qiskit-aer."""
    try:
        import torch

        if torch.cuda.is_available():
            return "GPU"
    except ImportError:
        pass

    return "CPU"


class DefaultSim(QuantumSimulator):
    def __init__(self, method: str = "automatic", device: str = None, **kwargs):
        """
        Inits qiskit aer sim

        Args:
            method: Simulation method
            device: Device to use ('CPU', 'GPU', None for auto-detect)
            **kwargs: Additional backend options
        """
        self.method = method
        self.device = device if device is not None else _detect_gpu_device()
        self.backend = AerSimulator(method=method, device=self.device, **kwargs)

    def run(self, qasm: str, shots: int = 1024) -> Dict[str, int]:
        """Run the circuit and return measurement counts."""
        try:
            circuit = self._parse_qasm(qasm)

            job = self.backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            if counts:
                max_len = max(len(bitstring) for bitstring in counts.keys())
                counts = {bitstring.zfill(max_len): count for bitstring, count in counts.items()}

            return counts

        except Exception as e:
            raise RuntimeError(f"Failed to run circuit: {str(e)}")

    def get_statevector(self, qasm: str) -> Optional[np.ndarray]:
        """Get the statevector of the circuit."""
        try:
            circuit = self._parse_qasm(qasm)
            circuit_no_meas = circuit.remove_final_measurements(inplace=False)

            backend_sv = AerSimulator(method="statevector", device=self.device)
            circuit_no_meas.save_statevector()  # type: ignore
            job = backend_sv.run(circuit_no_meas, shots=1)
            result = job.result()
            statevector = result.data(0)["statevector"]

            return np.array(statevector)

        except Exception as e:
            raise RuntimeError(f"Failed to get statevector: {str(e)}")

    def _parse_qasm(self, qasm: str) -> QuantumCircuit:
        """Parse QASM string to QuantumCircuit with fallback for custom instructions."""
        try:
            return QuantumCircuit.from_qasm_str(qasm)
        except Exception:
            custom_instructions = [
                CustomInstruction("rxx", 1, 2, RXXGate, builtin=True),
                CustomInstruction("ryy", 1, 2, RYYGate, builtin=True),
                CustomInstruction("rzz", 1, 2, RZZGate, builtin=True),
                CustomInstruction("u3", 3, 1, U3Gate, builtin=True),
            ]
            return loads(qasm, custom_instructions=custom_instructions)

    def get_info(self) -> Dict[str, Any]:
        """Get information about the simulator backend."""
        return {
            "backend": "qiskit_aer",
            "method": self.method,
            "device": self.device,
        }
