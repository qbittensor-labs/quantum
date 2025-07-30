from types import ModuleType, SimpleNamespace
from uuid import uuid4
import sys
import pytest

# Stub qbittensor.protocol.CompletedCircuits so import succeeds
proto = sys.modules.setdefault("qbittensor.protocol", ModuleType("qbittensor.protocol"))


class CompletedCircuits: # minimal placeholder
    pass


proto.CompletedCircuits = CompletedCircuits # type: ignore[attr-defined]

# 2.  Import target module after stub is in place
from qbittensor.validator.utils.response_verifier import verify_response
from qbittensor.validator.utils import crypto_utils as cu

# Patch both crypto_utils.verifyand response_verifier.verify
@pytest.fixture(autouse=True)
def _stub_verify(monkeypatch):
    monkeypatch.setattr(cu, "verify", lambda *_: True)
    # response_verifier.py imported 'verify' directly; patch that symbol too
    import qbittensor.validator.utils.response_verifier as rv
    monkeypatch.setattr(rv, "verify", lambda *_: True)


# Helper builders
def _meta(ch_id: str, val_sig: str, val_hk: str, sol_bytes: bytes):
    """Return an object with attrs accessed by verify_response()."""
    return SimpleNamespace(
        challenge_id=ch_id,
        validator_signature=val_sig,
        validator_hotkey=val_hk,
        solution_hash=cu.sha256_hex(sol_bytes),
        circuit_kind="peaked",
        difficulty=0.0,
        entanglement_entropy=0.0,
        nqubits=0,
        rqc_depth=0,
    )


def _resp(meta, sol_bytes: bytes):
    r = CompletedCircuits()
    r.validator_signature = meta.validator_signature
    r.validator_hotkey = meta.validator_hotkey
    r.challenge_id = meta.challenge_id
    r.miner_solution = sol_bytes.hex()
    r.miner_signature = "corn" * 16 # any 64‑byte hex
    return r


# Tests
def test_verify_response_success():
    sol = b"\x01\x02"
    cid = uuid4().hex
    meta = _meta(cid, "aa" * 32, "VAL-HK", sol)

    assert verify_response(_resp(meta, sol), meta, "MINER-HK")


def test_verify_response_hash_mismatch():
    good = b"\x07"
    bad = b"\x08"
    cid = uuid4().hex
    meta = _meta(cid, "bb" * 32, "VAL-HK", good)

    assert not verify_response(_resp(meta, bad), meta, "MINER-HK")


def test_verify_response_missing_miner_signature():
    sol = b"\x05"
    cid = uuid4().hex
    meta = _meta(cid, "cc" * 32, "VAL-HK", sol)

    resp = _resp(meta, sol)
    resp.miner_signature = None # omit mandatory field

    assert not verify_response(resp, meta, "MINER-HK")
