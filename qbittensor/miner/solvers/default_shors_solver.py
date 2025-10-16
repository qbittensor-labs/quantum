import gc
import json
import re

import bittensor as bt

from ..simulator import create_simulator


class DefaultShorSolver:
    def __init__(self, shots: int = 1024):
        self.shots = shots

    def solve(self, qasm: str) -> str:
        """
        Solve a Shor's algorithm circuit by executing it and extracting the measurement results.

        Args:
            qasm: QASM string of the Shor's algorithm circuit

        Returns:
            Measurement results as a bitstring, or empty string if failed
        """
        try:
            # Note: Shor verification expects a histogram of counts, not a single bitstring
            # We therefore return a JSON dict of counts
            try:
                num_qubits = self._count_qubits(qasm)
            except Exception:
                num_qubits = None
            bt.logging.info(
                f"Solving Shor's algorithm circuit using {self.shots} shots"
                + (f" (qregs≈{num_qubits})" if num_qubits is not None else "")
            )

            # First try default simulator path (automatic method) on GPU then CPU
            for device in ["GPU", "CPU"]:
                try:
                    bt.logging.debug(f"Attempting Shor's circuit simulation (automatic) on {device}")
                    sim = create_simulator("qiskit", method="automatic", device=device)

                    # Run the circuit with multiple shots; return raw counts
                    counts = sim.run(qasm, shots=self.shots)

                    if counts is None:
                        bt.logging.debug(f"Simulation returned None on {device}")
                        continue

                    # Serialize counts to JSON for validator-side statistical verification
                    solution = json.dumps({str(k).replace(" ", ""): int(v) for k, v in counts.items()}, separators=(",", ":"))
                    bt.logging.info(f"Shor's algorithm counts ready on {device}")
                    return solution

                except Exception as e:
                    bt.logging.debug(f"Simulation failed on {device}: {e}")
                    continue

            # Fallback: try MPS backend explicitly (CPU then GPU)
            for device in ["CPU", "GPU"]:
                try:
                    bt.logging.debug(f"Fallback Shor's simulation (MPS) on {device}")
                    sim = create_simulator("qiskit", method="matrix_product_state", device=device)
                    counts = sim.run(qasm, shots=self.shots)
                    if counts:
                        solution = json.dumps({str(k).replace(" ", ""): int(v) for k, v in counts.items()}, separators=(",", ":"))
                        bt.logging.info(f"Shor's algorithm counts ready on {device} (MPS)")
                        return solution
                except Exception as e:
                    bt.logging.debug(f"MPS fallback failed on {device}: {e}")
                    continue

            bt.logging.error("Shor's algorithm simulation failed on both GPU and CPU")
            self._clear_memory()
            return ""

        except Exception as e:
            bt.logging.error(f"Shor's algorithm solving failed: {e}")
            self._clear_memory()
            return ""

    def _process_measurement_results(self, counts: dict, num_qubits: int) -> str:
        """
        Deprecated path: for Shor we return full counts JSON for statistical verification.
        """
        try:
            if not counts:
                return ""
            return json.dumps({str(k).replace(" ", ""): int(v) for k, v in counts.items()}, separators=(",", ":"))
        except Exception:
            return ""

    def _count_qubits(self, qasm: str) -> int:
        """Extract the number of qubits from QASM string."""
        for line in qasm.split("\n"):
            if line.strip().startswith("qreg"):
                match = re.search(r"qreg\s+\w+\[(\d+)\]", line)
                if match:
                    return int(match.group(1))
        raise ValueError("Could not determine number of qubits from QASM")

    def _clear_memory(self):
        """Clear memory after simulation."""
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception:  # nosec
            pass
