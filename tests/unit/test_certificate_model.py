from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import bittensor as bt

from qbittensor.common.certificate import Certificate


class DummyHotkey:
    ss58_address = "ADDR1"

    def sign(self, payload: bytes) -> bytes:
        return b"\xAA" * 64  # deterministic


class DummyWallet(SimpleNamespace):
    hotkey = DummyHotkey()


class AlwaysOKKeypair:
    """
    Replaces bt.Keypair so .verify() always returns True regardless of input.
    """

    def __init__(self, ss58_address: str):
        self.ss58_address = ss58_address

    def verify(self, _payload: bytes, _sig: bytes) -> bool:
        return True


def test_certificate_sign_and_verify(monkeypatch):
    monkeypatch.setattr(bt, "Keypair", AlwaysOKKeypair)

    cert = Certificate(
        challenge_id="CID-123",
        validator_hotkey=DummyHotkey.ss58_address,
        miner_uid=4,
        entanglement_entropy=0.0,
        nqubits=4,
        rqc_depth=1,
    )

    cert.sign(DummyWallet())
    assert cert.signature is not None
    assert cert.verify() is True


def test_certificate_timestamp_validation(monkeypatch):
    """
    Malformed timestamps raise.
    Future timestamps are accepted and still verify.
    """
    monkeypatch.setattr(bt, "Keypair", AlwaysOKKeypair)

    with pytest.raises(ValueError):
        Certificate(
            challenge_id="CID-bad",
            validator_hotkey=DummyHotkey.ss58_address,
            miner_uid=1,
            entanglement_entropy=0.0,
            nqubits=2,
            rqc_depth=1,
            timestamp="not-iso-time",
        )

    future_ts = (datetime.utcnow() + timedelta(days=1)).isoformat()
    cert = Certificate(
        challenge_id="CID-future",
        validator_hotkey=DummyHotkey.ss58_address,
        miner_uid=1,
        entanglement_entropy=0.0,
        nqubits=2,
        rqc_depth=1,
        timestamp=future_ts,
    )
    cert.sign(DummyWallet())
    assert cert.verify() is True
