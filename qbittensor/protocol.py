# coding: utf-8
"""
Protocol for Quantum
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import bittensor as bt
from pydantic import Field
from typing import Any, Dict, List, Optional, Literal


class _CircuitSynapseBase(bt.Synapse):
    """Common metadata carried by every circuit-related synapse."""

    # signed by validator
    validator_signature: Optional[str] = Field(
        None, description="Ed25519 signature by the validator (hex)"
    )
    validator_hotkey: Optional[str] = Field(
        None, description="ss58 address of the validator that signed this challenge"
    )

    # deterministic challenge ID / difficulty
    challenge_id: Optional[str] = Field(
        None, description="SHA-256 hash of the unsigned challenge payload"
    )
    difficulty_level: Optional[float] = Field(
        None, description="Difficulty assigned by the validator"
    )

    # circuit payload & miner fields
    circuit_data: Optional[str] = Field(
        None, description="Serialized QASM of the circuit"
    )
    solution_bitstring: Optional[str] = Field(
        None, description="Miners bitstring solution"
    )

    # gossip: completion certificates & feedback
    certificates_json: Optional[str] = Field(
        default=None, description="Optional JSON batch of certificates"
    )
    desired_difficulty: Optional[float] = Field(
        None, description="Miner-suggested difficulty"
    )

    # populated during deserialization ↴
    certificates: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)

    # helpers for pushing / pulling certificates
    def attach_certificates(self, certs: List[Dict[str, Any]]) -> None:
        """Embed up to N certificates into this synapse."""
        if certs:
            self.certificates_json = json.dumps(
                {"type": "cert_batch", "certificates": certs},
                separators=(",", ":"),
            )

    def extract_certificates(self) -> List[Dict[str, Any]]:
        """
        Return a python list of certificate dicts.
        Works both before and after deserialization.
        """
        if self.certificates:  # populated by deserialize()
            return self.certificates
        if not self.certificates_json:  # raw string not yet parsed
            return []
        try:
            blob = json.loads(self.certificates_json)
            if isinstance(blob, dict) and blob.get("type") == "cert_batch":
                return blob.get("certificates", [])
            if isinstance(blob, dict) and "signature" in blob:
                return [blob]
            if isinstance(blob, list):
                return blob
        except Exception:
            pass
        return []

    # bittensor hook
    def deserialize(self):
        """
        Convert the JSON blob into a native Python list
        """
        self.certificates = self.extract_certificates()
        return self


class ChallengePeakedCircuit(_CircuitSynapseBase):
    """
    Peaked circuit challenge.
    """
    circuit_kind: Literal["peaked"] = "peaked"


class ChallengeHStabCircuit(_CircuitSynapseBase):
    """
    Hstabiliser circuit challenge.
    """
    circuit_kind: Literal["hstab"] = "hstab"


# legacy circuit
class ChallengeCircuits(_CircuitSynapseBase):
    pass
