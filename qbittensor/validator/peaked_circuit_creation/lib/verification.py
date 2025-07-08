"""
Verification framework for quantum compute subnet to prevent miner exploitation.
This module provides methods for challenge generation and result verification.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CircuitChallenge:
    """Encapsulates a circuit challenge sent to miners"""

    challenge_id: str
    qasm_circuit: str
    measurement_basis: List[str]  # Random measurement bases for spot checks
    nonce: str  # Salt to prevent precomputation
    timestamp: float
    difficulty_level: float


@dataclass
class CircuitSolution:
    """Expected solution stored by validator"""

    challenge_id: str
    target_state: str
    peak_probability: float
    expected_measurements: Dict[str, float]  # Basis -> expected probability
    circuit_hash: str


@dataclass
class MinerResponse:
    """Response from a miner"""

    challenge_id: str
    state_vector: Optional[np.ndarray]  # Full state vector (for small circuits)
    measurements: Dict[str, float]  # Measurement results in requested bases
    execution_time: float
    miner_id: str


class VerificationFramework:
    """Main verification framework for the quantum compute subnet"""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.challenges: Dict[str, CircuitSolution] = {}

    def create_challenge(self, circuit_params, seed: int) -> Tuple[CircuitChallenge, CircuitSolution]:
        """
        Create a verifiable challenge from circuit parameters.

        Returns:
            - CircuitChallenge: Data to send to miner
            - CircuitSolution: Expected solution for verification
        """
        peaked_circuit = circuit_params.compute_circuit(seed)
        qasm = peaked_circuit.to_qasm()

        challenge_id = hashlib.sha256(f"{qasm}{time.time()}{seed}".encode()).hexdigest()[:16]

        nonce = hashlib.sha256(f"{challenge_id}{np.random.random()}".encode()).hexdigest()[:8]

        # Add nonce as a comment to QASM to make circuit unique
        qasm_with_nonce = f"// Challenge: {challenge_id}\n// Nonce: {nonce}\n{qasm}"

        # Generate random measurement bases for spot checks
        n_qubits = peaked_circuit.num_qubits
        n_checks = min(10, 2**n_qubits // 4)  # Reasonable number of checks
        measurement_bases = self._generate_measurement_bases(n_qubits, n_checks)

        # Calculate expected measurements for verification
        expected_measurements = self._calculate_expected_measurements(peaked_circuit, measurement_bases)

        # Create circuit hash for integrity
        circuit_hash = hashlib.sha256(qasm_with_nonce.encode()).hexdigest()

        # Create challenge and solution objects
        challenge = CircuitChallenge(
            challenge_id=challenge_id,
            qasm_circuit=qasm_with_nonce,
            measurement_basis=measurement_bases,
            nonce=nonce,
            timestamp=time.time(),
            difficulty_level=circuit_params.difficulty,
        )

        solution = CircuitSolution(
            challenge_id=challenge_id,
            target_state=peaked_circuit.target_state,
            peak_probability=peaked_circuit.peak_prob,
            expected_measurements=expected_measurements,
            circuit_hash=circuit_hash,
        )

        # Store solution for later verification
        self.challenges[challenge_id] = solution

        return challenge, solution

    def verify_response(self, response: MinerResponse) -> Dict[str, any]:
        """
        Verify a miner's response to a challenge.

        Returns dict with:
            - valid: bool
            - score: float (0-1)
            - reason: str (if invalid)
        """
        # Check if challenge exists
        if response.challenge_id not in self.challenges:
            return {"valid": False, "score": 0.0, "reason": "Unknown challenge ID"}

        solution = self.challenges[response.challenge_id]

        # Verify timing (prevent too-fast responses that indicate cheating)
        min_time = self._estimate_min_simulation_time(solution)
        if response.execution_time < min_time * 0.8:  # 20% margin
            return {"valid": False, "score": 0.0, "reason": f"Suspiciously fast execution: {response.execution_time}s"}

        # Verify measurements match expected values
        measurement_score = self._verify_measurements(response.measurements, solution.expected_measurements)

        if measurement_score < 0.9:  # 90% accuracy threshold
            return {
                "valid": False,
                "score": measurement_score,
                "reason": "Measurement results don't match expected values",
            }

        # For small circuits, verify full state vector
        if response.state_vector is not None:
            state_score = self._verify_state_vector(
                response.state_vector, solution.target_state, solution.peak_probability
            )

            if state_score < 0.95:
                return {"valid": False, "score": state_score, "reason": "State vector doesn't match expected peak"}

        # Calculate final score based on accuracy and efficiency
        final_score = self._calculate_final_score(measurement_score, response.execution_time, solution)

        return {"valid": True, "score": final_score, "reason": "Valid simulation"}

    def _generate_measurement_bases(self, n_qubits: int, n_checks: int) -> List[str]:
        """Generate random computational basis states for spot checking"""
        max_states = 2**n_qubits
        if n_checks >= max_states:
            # For small circuits, check all basis states
            return [format(i, f"0{n_qubits}b") for i in range(max_states)]

        # For larger circuits, randomly sample
        indices = np.random.choice(max_states, n_checks, replace=False)
        return [format(i, f"0{n_qubits}b") for i in indices]

    def _calculate_expected_measurements(self, circuit, measurement_bases: List[str]) -> Dict[str, float]:
        """
        Calculate expected measurement probabilities.
        This is a placeholder - in production, you'd run a trusted simulation.
        """
        # In practice, the validator would run its own trusted simulation
        # to get these values. For now, return placeholder.
        expected = {}
        for basis in measurement_bases:
            if basis == circuit.target_state:
                expected[basis] = circuit.peak_prob
            else:
                # Other states have low probability
                expected[basis] = (1 - circuit.peak_prob) / (2**circuit.num_qubits - 1)
        return expected

    def _verify_measurements(self, reported: Dict[str, float], expected: Dict[str, float]) -> float:
        """Compare reported measurements with expected values"""
        if set(reported.keys()) != set(expected.keys()):
            return 0.0

        total_error = 0.0
        for basis, exp_prob in expected.items():
            rep_prob = reported.get(basis, 0.0)
            # Relative error for non-zero probabilities
            if exp_prob > self.tolerance:
                error = abs(rep_prob - exp_prob) / exp_prob
            else:
                error = abs(rep_prob - exp_prob)
            total_error += error

        # Convert error to score
        avg_error = total_error / len(expected)
        score = max(0.0, 1.0 - avg_error)
        return score

    def _verify_state_vector(self, state_vector: np.ndarray, target_state: str, expected_peak: float) -> float:
        """Verify the full state vector for small circuits"""
        # Check normalization
        norm = np.sum(np.abs(state_vector) ** 2)
        if abs(norm - 1.0) > self.tolerance:
            return 0.0

        # Check peak probability
        target_idx = int(target_state, 2)
        actual_peak = np.abs(state_vector[target_idx]) ** 2

        if abs(actual_peak - expected_peak) > self.tolerance:
            peak_score = max(0.0, 1.0 - abs(actual_peak - expected_peak))
        else:
            peak_score = 1.0

        return peak_score

    def _estimate_min_simulation_time(self, solution: CircuitSolution) -> float:
        """
        Estimate minimum realistic simulation time based on circuit complexity.
        This prevents miners from claiming unrealistic performance.
        """
        # This is a placeholder - in production, you'd have empirical data
        n_qubits = len(solution.target_state)
        # Rough estimate: exponential in qubits
        base_time = 0.001 * (2 ** (n_qubits / 4))
        return base_time

    def _calculate_final_score(self, accuracy_score: float, execution_time: float, solution: CircuitSolution) -> float:
        """
        Calculate final score combining accuracy and performance.
        Faster accurate simulations get higher scores.
        """
        # Base score from accuracy
        score = accuracy_score

        # Bonus for efficiency (up to 20% bonus)
        expected_time = self._estimate_min_simulation_time(solution) * 2
        if execution_time < expected_time:
            efficiency_bonus = 0.2 * (expected_time - execution_time) / expected_time
            score = min(1.0, score + efficiency_bonus)

        return score
