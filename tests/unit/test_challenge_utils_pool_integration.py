import re
import types

import pytest

from qbittensor.validator.utils.challenge_utils import build_peaked_challenge
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams


class _DummyCirc:
	def __init__(self, n, seed):
		self.num_qubits = n
		self.peak_prob = 0.5
		self.target_state = "0" * n
		self._seed = seed

	def to_qasm(self):
		return (
			"OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n"
			f"qreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\nmeasure q -> c;\n"
		)


class _DummyWallet:
	class _HK:
		ss58_address = "test_hotkey"
	def __init__(self):
		self.hotkey = self._HK()

@pytest.fixture(autouse=True)
def _fast_compute_circuits(monkeypatch):
	def _fake_compute(self, seed: int, n_variants: int = 10):
		n = int(self.nqubits)
		return [_DummyCirc(n, seed ^ (k * 0x9E3779B1)) for k in range(n_variants)]

	monkeypatch.setattr(CircuitParams, "compute_circuits", _fake_compute, raising=True)
	yield

def test_build_peaked_challenge_uses_pool_and_returns_target():
	wallet = _DummyWallet()
	syn, meta, target = build_peaked_challenge(wallet=wallet, difficulty=27.0) # this is qubits
	assert hasattr(syn, "circuit_data") and isinstance(syn.circuit_data, str)
	assert re.search(r"OPENQASM 2.0;", syn.circuit_data)
	assert isinstance(target, str) and set(target).issubset({"0", "1"})
	assert meta.nqubits >= 10
	assert meta.circuit_kind == "peaked"
