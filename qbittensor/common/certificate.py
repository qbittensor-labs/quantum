"""
Certificate helpers shared by both validator and miner.
"""

from __future__ import annotations
import json, datetime as dt
from typing import ClassVar, Dict
from pydantic import BaseModel, Field, validator
import bittensor as bt

_JSON_OPTS = dict(separators=(",", ":"), sort_keys=True)

# LEGACY CERTS CUTOFF
_CUTOFF = dt.datetime(2025, 8, 7, 12, 0, 0, tzinfo=dt.timezone.utc)


class Certificate(BaseModel):
    # payload
    challenge_id: str
    validator_hotkey: str
    miner_uid: int
    miner_hotkey: str | None = None  # allow legacy certs
    circuit_type: str = Field(
        default="peaked",
        pattern="^(peaked|shors)$",
        description="Quantum circuit type: 'peaked' or 'shors'"
    )
    entanglement_entropy: float
    nqubits: int
    rqc_depth: int
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.utcnow().isoformat(timespec="seconds"),
        description="UTC ISO-8601 creation time",
    )

    # cryptographic proof
    signature: str | None = Field(
        default=None, description="Ed25519 signature (hex) over canonical bytes"
    )

    # explicit tuple so both sides agree
    # extra safe !
    _ORDER: ClassVar[tuple[str, ...]] = (
        "challenge_id",
        "validator_hotkey",
        "miner_uid",
        "miner_hotkey",
        "circuit_type",
        "entanglement_entropy",
        "nqubits",
        "rqc_depth",
        "timestamp",
    )

    # Canonicalisation helpers
    def _canonical_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self._ORDER}

    def canonical_bytes(self) -> bytes:
        """Returns canonical UTF-8 bytes used for signing / verifying."""
        return json.dumps(self._canonical_dict(), **_JSON_OPTS).encode()

    # Sign / verify helpers
    def sign(self, wallet: bt.wallet):
        """Attach `signature` field (hex) using validator wallet’s hotkey keypair."""
        assert wallet.hotkey.ss58_address == self.validator_hotkey, "wallet mismatch"
        self.signature = wallet.hotkey.sign(self.canonical_bytes()).hex()

    def verify(self) -> bool:
        """
        True -> signature matches validator_hotkey
        False -> unsigned, malformed address, bad hex, or bad signature
        """
        if not self.signature:
            return False

        try:
            kp = bt.Keypair(ss58_address=self.validator_hotkey)
        except ValueError as e:
            bt.logging.debug(f"[cert] invalid hotkey {self.validator_hotkey[:8]}… ({e})")
            return False

        try:
            payload = self.canonical_bytes()
            sig     = bytes.fromhex(self.signature)
        except Exception as e:
            bt.logging.debug(f"[cert] signature decode failed ({e})")
            return False

        payload = self.canonical_bytes()
        if kp.verify(payload, sig):
            return True

        # accept legacy certs signed before the cutoff
        if (not getattr(self, "miner_hotkey", None)) and self._is_pre_cutoff():
            return kp.verify(self._legacy_bytes(), sig)

        return False

    # Pydantic extra validation
    @validator("timestamp")
    def _parse_ts(cls, v: str) -> str:
        # Raises if malformed
        dt.datetime.fromisoformat(v)
        return v

    # LEGACY CERTS HELPERS
    def _legacy_bytes(self) -> bytes:
       """Canonical bytes for certs issued before hotkey field existed."""
       legacy_order = tuple(k for k in self._ORDER if k != "miner_hotkey")
       payload = {k: getattr(self, k) for k in legacy_order}
       return json.dumps(payload, **_JSON_OPTS).encode()
    def _is_pre_cutoff(self) -> bool:
       try:
           ts = dt.datetime.fromisoformat(self.timestamp)
           if ts.tzinfo is None:                
                ts = ts.replace(tzinfo=dt.timezone.utc)
           return ts < _CUTOFF
       except Exception:
           return False
