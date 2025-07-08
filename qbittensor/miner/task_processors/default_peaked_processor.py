from typing import Any, Dict

import bittensor as bt
import numpy as np

from .base import TaskProcessor


class PeakedCircuitProcessor(TaskProcessor):
    def __init__(self, use_exact: bool = True):
        """
        Inits peaked circuit processor

        Args:
            use_exact: If True, use exact probability calculation from statevector.
                      If False, use sampling approach (find most frequent).
        """
        self.use_exact = use_exact

    def process(self, counts_or_statevector: Any, **kwargs) -> Dict[str, Any]:
        """
        Extracts the most probable bitstring

        Args:
            counts_or_statevector: Either:
                - Dict[str, int]: measurement counts (if use_exact=False)
                - np.ndarray: statevector (if use_exact=True)

        Returns:
            Dictionary containing:
                - peak_bitstring: The most probable bitstring
                - peak_probability: Its probability
        """
        if self.use_exact:
            if isinstance(counts_or_statevector, dict):
                raise ValueError("Expected statevector array but got counts dict. Set use_exact=False for sampling.")

            statevector = counts_or_statevector
            n_qubits = int(np.log2(len(statevector)))

            probabilities = np.abs(statevector) ** 2

            peak_idx = np.argmax(probabilities)
            peak_probability = probabilities[peak_idx]

            raw_bitstring = format(peak_idx, f"0{n_qubits}b")
            peak_bitstring = raw_bitstring[::-1]

            top_indices = np.argsort(probabilities)[-5:][::-1]
            bt.logging.info("Top 5 bitstrings by probability:")
            for idx in top_indices:
                raw_bitstr = format(idx, f"0{n_qubits}b")
                bitstr = raw_bitstr[::-1]
                prob = probabilities[idx]
                bt.logging.info(f"  {bitstr}: {prob:g}")
            peaking = probabilities[top_indices[0]] / probabilities[top_indices[1]]
            bt.logging.info(f"Actual peaking ratio: {peaking:g}")

            return {
                "peak_bitstring": peak_bitstring,
                "peak_probability": float(peak_probability),
                "peaking_ratio": peaking,
            }
        else:
            if not isinstance(counts_or_statevector, dict):
                raise ValueError("Expected counts dict but got array. Set use_exact=True for statevector.")

            counts = counts_or_statevector
            if not counts:
                return {"peak_bitstring": None, "peak_probability": 0.0}

            total_shots = sum(counts.values())

            peak_bitstring_qiskit = max(counts.keys(), key=lambda x: counts[x])
            peak_count = counts[peak_bitstring_qiskit]
            peak_probability = peak_count / total_shots
            peak_bitstring = peak_bitstring_qiskit[::-1]

            return {"peak_bitstring": peak_bitstring, "peak_probability": peak_probability}

    def validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate if we found a peak bitstring

        Args:
            result: The processed result dictionary

        Returns:
            True if a peak bitstring was found
        """
        return result.get("peak_bitstring") is not None
