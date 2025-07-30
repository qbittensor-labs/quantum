from types import SimpleNamespace
import hashlib
import json
import pytest
import bittensor as bt

from qbittensor.validator.utils import crypto_utils as cu


#  Dummy Keypair that always verifies as True
class DummyKeypair:
    def __init__(self, ss58_address: str):
        self.ss58_address = ss58_address

    def sign(self, payload: bytes) -> bytes:
        return b"\xAA" * 64

    def verify(self, _payload: bytes, _sig: bytes) -> bool:
        return True


@pytest.fixture(autouse=True)
def _patch_keypair(monkeypatch):
    monkeypatch.setattr(bt, "Keypair", DummyKeypair)


#  Tests
def test_canonical_hash_determinism():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}  # same content, different order

    h1 = cu.canonical_hash(a)
    h2 = cu.canonical_hash(b)

    assert h1 == h2
    # Control: change value -> hash changes
    assert cu.canonical_hash({"x": 1, "y": 3}) != h1


def test_sha256_helpers():
    data = b"hello"
    assert cu.sha256_hex(data) == hashlib.sha256(data).hexdigest()
    assert cu.sha256_bytes(data) == hashlib.sha256(data).digest()


def test_sign_and_verify_roundtrip():
    kp = DummyKeypair("XPLSK1")
    msg = b"payload-123"

    sig = cu.sign(kp, msg)
    assert isinstance(sig, str) and len(sig) == 128  # 64 bytes -> 128 hex chars

    ok = cu.verify(sig, "XPLSK1", msg)
    assert ok is True
