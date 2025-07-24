import gc
import re
import bittensor as bt
from ..simulator import create_simulator
from ..task_processors import HStabCircuitProcessor


class DefaultHStabSolver:
    def solve(self, qasm: str) -> str:
        """
        Solve a hidden stabilizer circuit to find the stabilizer generators.

        Args:
            qasm: QASM string of the hidden stabilizer circuit

        Returns:
            Single string of concatenated stabilizers (length nxn), or empty string if failed
        """
        try:
            num_qubits = self._count_qubits(qasm)
            bt.logging.info(
                f"Solving hidden stabilizer circuit with {num_qubits} qubits"
            )
            return self._run(qasm)

        except Exception as e:
            bt.logging.error(f"Hidden stabilizer solving failed: {e}")
            self._clear_memory()
            return ""

    def _run(self, qasm: str) -> str:
        for device in ["GPU", "CPU"]:
            try:
                sim = create_simulator("qiskit", method="statevector", device=device)
                bt.logging.debug(f"Attempting statevector simulation on {device}")

                statevector = sim.get_statevector(qasm)
                if statevector is None:
                    continue

                processor = HStabCircuitProcessor()
                result = processor.process(statevector)

                if result.get("success", False):
                    stabilizers = result.get("stabilizers", [])
                    bt.logging.info(
                        f"Hidden stabilizer simulation successful on {device}"
                    )
                    return self._format_stabilizers(stabilizers)

            except Exception as e:
                bt.logging.debug(f"Simulation failed on {device}: {e}")
                continue

        bt.logging.error("Hidden stabilizer simulation failed on both GPU and CPU")
        self._clear_memory()
        return ""

    def _format_stabilizers(self, stabilizers) -> str:
        try:
            stabilizer_strings = []
            for stab in stabilizers:
                stab_str = str(stab)
                stabilizer_strings.append(stab_str)

            return "".join(stabilizer_strings)

        except Exception as e:
            bt.logging.error(f"Failed to format stabilizers: {e}")
            return ""

    def _count_qubits(self, qasm: str) -> int:
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
