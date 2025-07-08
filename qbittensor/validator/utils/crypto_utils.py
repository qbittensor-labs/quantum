"""
Hashing & Ed25519 helpers shared by validator / miner code.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

import bittensor as bt


# Hashing
def canonical_hash(payload: dict) -> str:
    """Deterministically hash a dict → SHA-256 hex digest."""
    serialised: Final = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialised).hexdigest()


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    """SHA-256 raw bytes."""
    return hashlib.sha256(data).digest()


# Ed25519 sign / verify
def sign(keypair: bt.Keypair, data: bytes) -> str:
    """Return *hex* signature of *data* with *keypair*."""
    return keypair.sign(data).hex()


def verify(sig_hex: str, ss58: str, data: bytes) -> bool:
    """True ⇔ *sig_hex* is a valid signature over *data* by *ss58*."""
    try:
        kp = bt.Keypair(ss58_address=ss58)
    except Exception:
        return False
    return kp.verify(data, bytes.fromhex(sig_hex))
