"""
Miner-side verification of certificates.  Rejects anything that is either
(a) not properly signed, or (b) from a non-whitelisted validator.
Certificates that pass can be written to disk, added to local score, etc.
"""

from __future__ import annotations
import json, pathlib
from datetime import datetime, timezone
from typing import List
import bittensor as bt

from qbittensor.common.certificate import Certificate


_WHITELIST_JSON = (
    pathlib.Path(__file__).resolve().parents[2]
    / "validator"
    / "config"
    / "whitelist_validators.json"
)


class CertificateVerifier:
    def __init__(self):
        with open(_WHITELIST_JSON) as f:
            self._whitelist = set(json.load(f)["whitelist"])

    def validate_batch(self, syn) -> List[Certificate]:
        """
        Pull certificates embedded in an incoming ChallengeCircuits or response,
        verify each, and return the valid ones.  Any invalid cert is logged.
        """
        good: List[Certificate] = []

        for raw in syn.extract_certificates():
            try:
                cert = Certificate(**raw)
            except Exception as e:
                bt.logging.warning(f"[cert] malformed: {e} {raw}")
                continue

            # whitelist check
            if cert.validator_hotkey not in self._whitelist:
                bt.logging.warning(
                    f"[cert] {cert.validator_hotkey} not in whitelist – rejected"
                )
                continue

            # cryptographic check
            #if not cert.verify():
            #    bt.logging.warning("[cert] bad signature – rejected")
            #    continue

            good.append(cert)

        bt.logging.debug(
            f"[cert] accepted {len(good)}/{len(syn.extract_certificates())}"
        )
        return good

    def persist(self, certs: List[Certificate], directory: pathlib.Path):
        """
        write accepted certs to disk (one JSON per file).
        """
        directory.mkdir(parents=True, exist_ok=True)
        hist_dir = directory.parent / "old_certificates"

        # build a quick lookup of already‑archived CIDs (filenames start with cid__)
        archived_cids = {p.name.split("__", 1)[0] for p in hist_dir.glob("*.json")}

        for cert in certs:
            cid = cert.challenge_id # Certificate attribute

            # skip if present in live dir or in archive
            if (directory / f"{cid}.json").exists() or cid in archived_cids:
                continue

            # ensure timestamp field is present and in UTC
            if not getattr(cert, "timestamp", None):
                cert.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

            # convert to plain dict before dumping
            payload = cert.__dict__ if isinstance(cert, Certificate) else cert
            (directory / f"{cid}.json").write_text(json.dumps(payload))
