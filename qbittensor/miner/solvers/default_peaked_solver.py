import gc
import bittensor as bt
from ..simulator import create_simulator
from ..task_processors import PeakedCircuitProcessor


class DefaultPeakedSolver:
    def solve(self, qasm: str) -> str:
        """
        Solve a quantum circuit to find the peaked bitstring.

        Strategy:
        - ≤32 qubits: GPU-accelerated statevector (if available)
        - >32 qubits: Try MPS first, fallback to CPU statevector

        Args:
            qasm: QASM string of the circuit

        Returns:
            Most probable bitstring, or empty string if failed
        """
        try:
            num_qubits = self._count_qubits(qasm)

            bt.logging.info(f"Solving circuit with {num_qubits} qubits")

            if num_qubits > 32:
                bt.logging.info("Large circuit detected, trying MPS method")
                result = self._mps_run(qasm)
                if result:
                    return result

                bt.logging.info("MPS failed, falling back to standard simulation")

            return self._run(qasm)

        except Exception as e:
            bt.logging.error(f"Circuit solving failed: {e}")
            self._clear_memory()
            return ""

    def _mps_run(self, qasm: str) -> str:
        try:
            sim = create_simulator("qiskit", method="matrix_product_state", device="CPU")
            bt.logging.debug(f"Using MPS simulation on device: {sim.device}")

            counts = sim.run(qasm, shots=1)

            if counts:
                processor = PeakedCircuitProcessor(use_exact=False)
                result = processor.process(counts)
                peak_bitstring = result.get("peak_bitstring")

                if peak_bitstring:
                    bt.logging.info("MPS simulation successful")
                    return peak_bitstring

            return ""

        except Exception as e:
            bt.logging.debug(f"MPS simulation failed: {e}")
            return ""

    def _run(self, qasm: str) -> str:
        for device in ["GPU", "CPU"]:
            try:
                sim = create_simulator("qiskit", method="statevector", device=device)
                bt.logging.debug(f"Attempting statevector simulation on device: {sim.device}")

                statevector = sim.get_statevector(qasm)

                if statevector is not None:
                    processor = PeakedCircuitProcessor(use_exact=True)
                    result = processor.process(statevector)
                    peak_bitstring = result.get("peak_bitstring")

                    if peak_bitstring:
                        bt.logging.info(f"Statevector simulation successful on {device}")
                        return peak_bitstring

            except Exception as e:
                bt.logging.debug(f"Simulation failed on {device}: {e}")
                if device == "CPU":
                    bt.logging.error(f"Standard simulation failed on both GPU and CPU: {e}")
                    self._clear_memory()
                continue

        return ""

    def _count_qubits(self, qasm: str) -> int:
        import re

        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        raise ValueError("Could not determine number of qubits from QASM")

    def _clear_memory(self):
        try:
            gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass
