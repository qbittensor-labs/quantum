"""
Pure function to verify a CompletedCircuits response.
"""

from __future__ import annotations


from qbittensor.protocol import CompletedCircuits

from .crypto_utils import sha256_hex, verify
from .validator_meta import ChallengeMeta


def verify_response(
    resp: CompletedCircuits,
    meta: ChallengeMeta,
    miner_hotkey: str,
) -> bool:
    """
    Returns True if *resp* passes every provenance & correctness check.
    """
    # validator provenance
    if (
        resp.validator_signature != meta.validator_signature
        or resp.validator_hotkey != meta.validator_hotkey
        or resp.challenge_id != meta.challenge_id
        or not verify(
            meta.validator_signature, meta.validator_hotkey, meta.challenge_id.encode()
        )
    ):
        return False

    # miner fields
    if resp.miner_solution is None or resp.miner_signature is None:
        return False

    payload = meta.challenge_id.encode() + bytes.fromhex(resp.miner_solution)
    if not verify(resp.miner_signature, miner_hotkey, payload):
        return False

    # correctness
    if sha256_hex(bytes.fromhex(resp.miner_solution)) != meta.solution_hash:
        return False

    return True
