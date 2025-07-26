import sys
from pathlib import Path

import bittensor as bt
from qbittensor.miner.solver_worker import SolverWorker
from qbittensor.miner.solvers.default_peaked_solver import DefaultPeakedSolver
from qbittensor.miner.solvers.default_hstab_solver import DefaultHStabSolver


class CircuitSolver:
    def __init__(self, base_dir):
        self._worker = SolverWorker(base_dir=base_dir, solver_fn=self._solve)
        self._worker.start()
        self._custom_peaked_solver, self._custom_hstab_solver = self._detect_custom()
        self._solvers = {
            'peaked': self._custom_peaked_solver or DefaultPeakedSolver(),
            'hstab': self._custom_hstab_solver or DefaultHStabSolver()
        }

    def _detect_custom(self):
        """Detect both custom peaked and hstab solvers."""
        solvers_dir = Path(__file__).parent.parent / "solvers"
        custom_peaked_solver = None
        custom_hstab_solver = None
        
        peaked_path = solvers_dir / "custom_peaked_solver.py"
        if peaked_path.exists():
            try:
                from qbittensor.miner.solvers.custom_peaked_solver import CustomPeakedSolver
                custom_peaked_solver = CustomPeakedSolver()
                bt.logging.info("Using custom peaked solver")
            except ImportError:
                bt.logging.info("Custom peaked solver file exists but failed to import, using default peaked solver")
            except Exception as e:
                bt.logging.error(f"Failed to load custom peaked solver: {e}")
        else:
            bt.logging.info("Using default peaked solver")
        
        hstab_path = solvers_dir / "custom_hstab_solver.py"
        if hstab_path.exists():
            try:
                from qbittensor.miner.solvers.custom_hstab_solver import CustomHStabSolver
                custom_hstab_solver = CustomHStabSolver()
                bt.logging.info("Using custom hstab solver")
            except ImportError:
                bt.logging.info("Custom hstab solver file exists but failed to import, using default hstab solver")
            except Exception as e:
                bt.logging.error(f"Failed to load custom hstab solver: {e}")
        else:
            bt.logging.info("Using default hstab solver")
            
        return custom_peaked_solver, custom_hstab_solver

    def _solve(self, qasm: str, circuit_type: str = None) -> str:
        """
        Solve a quantum circuit using the appropriate solver based on circuit type.

        Args:
            qasm: QASM string of the circuit
            circuit_type: 'peaked' or 'hstab'

        Returns:
            str: Solution string, or empty string if solving failed
        """
        try:
            if circuit_type not in self._solvers:
                bt.logging.error(f"Unknown circuit type: {circuit_type}. Skipping circuit.")
                return ""
            
            bt.logging.info(f"Solving {circuit_type} circuit")
            result = self._solvers[circuit_type].solve(qasm)
            return self._check_solution(result, circuit_type)
                    
        except Exception as e:
            bt.logging.error(f"Circuit solver failed: {e}")
            return ""

    def _check_solution(self, solution, circuit_type: str) -> str:
        if not isinstance(solution, str):
            bt.logging.error(f"{circuit_type} solver returned {type(solution).__name__}, expected str")
            raise TypeError(f"Solver must return str, got {type(solution).__name__}")
        return solution

    def submit(self, syn):
        self._worker.submit_synapse(syn)

    def drain(self, n=10, validator_hotkey=None):
        return self._worker.drain_solutions(n=n, validator_hotkey=validator_hotkey)

    def handle_reward(self, cid, reward_dict):
        self._worker.handle_reward(cid, reward_dict)
