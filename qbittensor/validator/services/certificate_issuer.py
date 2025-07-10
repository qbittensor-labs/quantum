"""
Validator-side certificate helper
"""

from __future__ import annotations
import json, pathlib, shutil, tempfile
from pathlib import Path
from typing import List

import bittensor as bt
from qbittensor.common.certificate import Certificate


#  Configuration
_BASE = (
    pathlib.Path(__file__).resolve().parents[2] / "certificates"
)  # points to .../validator/certificates
_OUTBOX = _BASE / "pending"
_SENT = _BASE / "sent"

_WHITELIST_JSON = (
    Path(__file__).resolve().parents[1] / "config" / "whitelist_validators.json"
)
_BASE.mkdir(exist_ok=True)
_OUTBOX.mkdir(exist_ok=True)
_SENT.mkdir(exist_ok=True)


class CertificateIssuer:
    """Create + store certificates, then pop them later for gossip."""

    def __init__(self, wallet: bt.wallet):
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address

        with open(_WHITELIST_JSON) as f:
            self._whitelist = set(json.load(f)["whitelist"])
        if self.hotkey not in self._whitelist:
            bt.logging.warning(
                f"⚠️  Hotkey {self.hotkey} missing from whitelist; miners will reject certs."
            )

    #  Public API
    def issue(
        self,
        *,
        challenge_id: str,
        miner_uid: int,
        entanglement_entropy: float,
        nqubits: int,
        rqc_depth: int,
    ) -> Certificate:
        """
        Build, sign, and persist a certificate in the miner’s outbox directory.
        Returns the Certificate object (in case caller needs it for logging).
        """
        cert = Certificate(
            challenge_id=challenge_id,
            validator_hotkey=self.hotkey,
            miner_uid=miner_uid,
            entanglement_entropy=entanglement_entropy,
            nqubits=nqubits,
            rqc_depth=rqc_depth,
        )
        cert.sign(self.wallet)

        # one-file-per-cert -> certificates/pending/<uid>/<challenge>.json
        miner_dir = _OUTBOX / str(miner_uid)
        miner_dir.mkdir(exist_ok=True)

        path = miner_dir / f"{cert.challenge_id}.json"
        path.write_text(cert.model_dump_json())

        bt.logging.debug(f"[cert] queued → {path}")
        return cert

    def pop_for(self, miner_uid: int, max_items: int | None = None) -> List[dict]:
        """
        Atomically move ALL (or first N) pending certificates for miner_uid
        """
        miner_dir = _OUTBOX / str(miner_uid)
        if not miner_dir.exists():
            return []

        cert_files = sorted(miner_dir.iterdir())  # deterministic order
        if max_items:
            cert_files = cert_files[:max_items]

        certs: List[dict] = []
        for f in cert_files:
            try:
                certs.append(json.loads(f.read_text()))
            except Exception as e:
                bt.logging.warning(f"[cert] could not read {f}: {e}")

        # atomic move to “sent” for audit/debug
        if cert_files:
            sent_dir = _SENT / str(miner_uid)
            sent_dir.mkdir(exist_ok=True)
            #for f in cert_files:
            #    shutil.move(str(f), sent_dir / f.name)

        return certs
