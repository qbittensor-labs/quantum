"""
Miner-side verification of certificates.  Rejects anything that is either
(a) not properly signed, or (b) from a non-whitelisted validator.
Certificates that pass can be written to disk, added to local score, etc.
"""

from __future__ import annotations
import json, pathlib, datetime as dt
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
            if not cert.verify():
                bt.logging.warning("[cert] bad signature – rejected")
                continue

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
        for c in certs:
            fname = directory / f"{c.challenge_id}__{c.validator_hotkey}.json"
            fname.write_text(c.model_dump_json())
