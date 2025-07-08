"""
Normalizes any miner response into a list of Solution objects
"""
from __future__ import annotations

import json
from typing import List

import bittensor as bt
from qbittensor.protocol import ChallengeCircuits

from .solution_processor import Solution


class SolutionExtractor:
    """`extract(resp)`-> List[Solution]"""

    @staticmethod
    def _build(sol_like) -> Solution:
        """Safely pull every field we care about from an object or dict."""
        getter = (
            (lambda k, default=None: sol_like.get(k, default))
            if isinstance(sol_like, dict)
            else (lambda k, default=None: getattr(sol_like, k, default))
        )

        return Solution(
            challenge_id=getter("challenge_id", ""),
            solution_bitstring=getter("solution_bitstring", "")
            or getter("miner_solution", ""),
            difficulty_level=getter("difficulty_level"),
            entanglement_entropy=getter("entanglement_entropy"),
            nqubits=getter("nqubits"),
            rqc_depth=getter("rqc_depth"),
        )

    @staticmethod
    def extract(resp) -> List[Solution]:

        if isinstance(resp, (ChallengeCircuits)):
            raw = resp.solution_bitstring or getattr(resp, "miner_solution", "") or ""

            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and obj.get("type") == "batch":
                    solutions = [
                        SolutionExtractor._build(s)
                        for s in obj.get("solutions", [])
                        if s.get("solution_bitstring")
                    ]
                    if solutions:
                        return solutions
            except json.JSONDecodeError:
                pass

            if raw:
                return [SolutionExtractor._build(resp)]

        bt.logging.debug(
            f"[solution-extractor] nothing extracted from {type(resp).__name__}"
        )
        return []
