"""
Utility helpers for signing and verifying `Rewards` payloads.
"""

from __future__ import annotations


import bittensor as bt


def _reward_message(challenge_id: str, entanglement_entropy: float) -> bytes:
    """
    Canonical, deterministic byte representation that is signed / verified.
    """
    return f"{challenge_id}|{entanglement_entropy:.12f}".encode("utf-8")


def sign_reward(
    hotkey: bt.Keypair, *, challenge_id: str, entanglement_entropy: float
) -> str:
    """
    Returns a hex-encoded Ed25519 signature created with `hotkey`.
    """
    msg = _reward_message(challenge_id, entanglement_entropy)
    return hotkey.sign(msg).hex()


def verify_reward_signature(
    *,
    signature_hex: str,
    validator_hotkey_ss58: str,
    challenge_id: str,
    entanglement_entropy: float,
) -> bool:
    """
    Anyone can call this to prove that `signature_hex` was produced by
    `validator_hotkey_ss58` over the given message.
    """
    try:
        kp = bt.Keypair(ss58_address=validator_hotkey_ss58)
        msg = _reward_message(challenge_id, entanglement_entropy)
        return kp.verify(signature=bytes.fromhex(signature_hex), message=msg)
    except Exception:
        return False
