import sys
from pathlib import Path

import bittensor as bt
from qbittensor.miner.solver_worker import SolverWorker
from qbittensor.miner.solvers.default_peaked_solver import DefaultPeakedSolver


class CircuitSolver:
    def __init__(self, base_dir):
        self._worker = SolverWorker(base_dir=base_dir, solver_fn=self._solve)
        self._worker.start()
        self._custom_solver = self._detect_custom()

        if self._custom_solver is None:
            self._default_solver = DefaultPeakedSolver()
        else:
            self._default_solver = None

    def _detect_custom(self):
        try:
            solver_path = (
                Path(__file__).parent.parent / "solvers" / "custom_peaked_solver.py"
            )

            if solver_path.exists():
                from qbittensor.miner.solvers.custom_peaked_solver import CustomSolver

                solver_instance = CustomSolver()
                bt.logging.info("Using custom solver")
                return solver_instance
            else:
                bt.logging.info("Using default solver")
                return None

        except ImportError:
            bt.logging.info(
                "Custom solver file exists but failed to import, using default solver"
            )
            return None
        except Exception as e:
            bt.logging.error(f"Failed to load custom solver: {e}")
            return None

    def _solve(self, qasm: str) -> str:
        """
        Solve a quantum circuit using custom or default solver pipeline.

        Returns:
            str: Peak bitstring, or empty string if solving failed
        """
        try:
            if self._custom_solver:
                return self._custom_solver.solve(qasm)
            else:
                return self._default_solver.solve(qasm)
        except Exception as e:
            bt.logging.error(f"Circuit solver failed: {e}")
            return ""

    def submit(self, syn):
        self._worker.submit_synapse(syn)

    def drain(self, n=10, validator_hotkey=None):
        return self._worker.drain_solutions(n=n, validator_hotkey=validator_hotkey)

    def handle_reward(self, cid, reward_dict):
        self._worker.handle_reward(cid, reward_dict)
