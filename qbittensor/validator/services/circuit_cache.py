from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

import bittensor as bt
from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import (
    CircuitParams,
)


@dataclass(slots=True)
class CircuitArtifact:
    qasm: str
    target_state: str
    peak_prob: float
    num_qubits: int
    seed: int
    rqc_depth: int
    created_at: float


class CircuitPool:
    """
    cache of peaked circuits keyed by num_qubits.
    (1 base + N-1 obfuscated variants)
    """

    def __init__(
        self,
        *,
        batch_size: int = 10,
        low_watermark: int = 3,
        ttl_seconds: float = 15 * 60.0,
    ) -> None:
        self._batch_size = max(1, int(batch_size))
        self._low_water = max(0, int(low_watermark))
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._qmap: Dict[int, Deque[CircuitArtifact]] = {}
        self._last_fill: Dict[int, float] = {}
        self._fills_inflight: Dict[int, bool] = {}

    def get(self, nqubits: int) -> CircuitArtifact:
        """
        Return a single circuit.
        """
        n = int(nqubits)
        with self._lock:
            dq = self._qmap.setdefault(n, deque())
            if dq:
                art = dq.popleft()
            else:
                art = None
        if art is None:
            batch = self._build_batch(n)
            art = batch[0]
            with self._lock:
                for a in batch[1:]:
                    self._qmap[n].append(a)
                self._last_fill[n] = time.time()
                self._fills_inflight[n] = False
            return art

        with self._lock:
            need_refill = (len(self._qmap[n]) <= self._low_water) or self._is_stale(n)
            inflight = self._fills_inflight.get(n, False)
            if need_refill and not inflight:
                self._fills_inflight[n] = True
                threading.Thread(target=self._refill, args=(n,), daemon=True).start()
        return art

    def _is_stale(self, nqubits: int) -> bool:
        ts = self._last_fill.get(nqubits, 0.0)
        return (time.time() - ts) > self._ttl

    def _refill(self, nqubits: int) -> None:
        try:
            batch = self._build_batch(nqubits)
            with self._lock:
                for a in batch:
                    self._qmap.setdefault(nqubits, deque()).append(a)
                self._last_fill[nqubits] = time.time()
        except Exception:
            bt.logging.error("[circuit-pool] background refill failed", exc_info=True)
        finally:
            with self._lock:
                self._fills_inflight[nqubits] = False

    def _build_batch(self, nqubits: int) -> List[CircuitArtifact]:
        def _convert_qubits_to_peaked_difficulty(n: int) -> float:
            m = max(2, int(n))
            return max(0.0, (2 ** ((m - 12) / 10.0)) - 3.9)

        difficulty = _convert_qubits_to_peaked_difficulty(nqubits)
        rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
        rqc_depth = int(round(rqc_mul * nqubits))
        pqc_depth = max(1, nqubits // 5)
        params = CircuitParams(difficulty, nqubits, rqc_depth, pqc_depth)
        seed_base = (np.uint64(time.time_ns()) ^ np.uint64(nqubits * 0x9E3779B1)).item()
        circuits = params.compute_circuits(int(seed_base), n_variants=self._batch_size)
        artifacts: List[CircuitArtifact] = []
        created = time.time()
        for idx, c in enumerate(circuits):
            artifacts.append(
                CircuitArtifact(
                    qasm=c.to_qasm(),
                    target_state=c.target_state,
                    peak_prob=float(c.peak_prob),
                    num_qubits=c.num_qubits,
                    seed=int(seed_base) ^ (idx * 0x9E3779B1),
                    rqc_depth=params.rqc_depth,
                    created_at=created,
                )
            )
        return artifacts


GLOBAL_CIRCUIT_POOL = CircuitPool()


