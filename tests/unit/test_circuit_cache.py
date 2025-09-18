import time
import re

import numpy as np
import pytest

from qbittensor.validator.services.circuit_cache import CircuitPool
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams


class _DummyCirc:
	def __init__(self, n: int, seed: int):
		self.num_qubits = n
		self._seed = seed
		self.peak_prob = 0.5
		self.target_state = "".join("1" if (seed >> i) & 1 else "0" for i in range(n))

	def to_qasm(self) -> str:
		return (
			"OPENQASM 2.0;\n"
			"include \"qelib1.inc\";\n\n"
			f"qreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\n"
			"measure q -> c;\n"
		)


@pytest.fixture(autouse=True)
def _fast_compute_circuits(monkeypatch):
	def _fake_compute(self, seed: int, n_variants: int = 10):
		n = int(self.nqubits)
		return [_DummyCirc(n, int(seed) ^ (k * 0x9E3779B1)) for k in range(n_variants)]

	monkeypatch.setattr(CircuitParams, "compute_circuits", _fake_compute, raising=True)
	yield


def test_circuit_pool_basic_unique_and_refill():
	pool = CircuitPool(batch_size=4, low_watermark=1, ttl_seconds=2.0)
	art1 = pool.get(16)
	assert art1.num_qubits == 16
	assert isinstance(art1.qasm, str) and len(art1.qasm) > 0
	assert re.search(r"OPENQASM 2.0;", art1.qasm)
	art2 = pool.get(16)
	art3 = pool.get(16)
	art4 = pool.get(16)
	seeds = {art1.seed, art2.seed, art3.seed, art4.seed}
	assert len(seeds) == 4, "expected all artifacts to be unique by seed"
	time.sleep(0.5)
	art5 = pool.get(16)
	assert art5.num_qubits == 16


def test_circuit_pool_per_qubits_keys():
	pool = CircuitPool(batch_size=3, low_watermark=0, ttl_seconds=60.0)
	art_a = pool.get(12)
	art_b = pool.get(14)
	assert art_a.num_qubits == 12
	assert art_b.num_qubits == 14
	assert art_a.qasm != art_b.qasm
