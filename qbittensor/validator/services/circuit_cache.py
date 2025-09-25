from __future__ import annotations

import threading
import time
import json
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
        self._cache_dir = Path("peaked_cache/obf_circuits")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._cache_dir / "served_manifest.json"
        self._load_manifest()
        self._cleanup_old_entries()

    def get(self, nqubits: int) -> CircuitArtifact:
        """
        Return a single circuit.
        """
        n = int(nqubits)
        with self._lock:
            dq = self._qmap.setdefault(n, deque())
            if dq:
                art = dq.popleft()
                return art
        
        disk_result = self._get_unused_from_disk(n)
        if disk_result:
            qasm, target_state = disk_result
            return CircuitArtifact(
                qasm=qasm,
                target_state=target_state,
                peak_prob=0.0,
                num_qubits=n,
                seed=0,
                rqc_depth=0,
                created_at=time.time()
            )
        
        batch = self._build_batch(n)
        art = batch[0]
        
        for idx, circuit in enumerate(batch):
            self._save_to_disk(circuit, idx)
        
        if len(self._manifest.get("entries", [])) % 100 == 0:
            self._cleanup_old_entries()
        
        with self._lock:
            for a in batch[1:]:
                self._qmap[n].append(a)
            self._last_fill[n] = time.time()
            self._fills_inflight[n] = False
        
        return art

    def _is_stale(self, nqubits: int) -> bool:
        ts = self._last_fill.get(nqubits, 0.0)
        return (time.time() - ts) > self._ttl

    def _refill(self, nqubits: int) -> None:
        try:
            batch = self._build_batch(nqubits)
            for idx, circuit in enumerate(batch):
                self._save_to_disk(circuit, idx)
            
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
            try:
                qasm_str = c.to_qasm()
                
                artifacts.append(
                    CircuitArtifact(
                        qasm=qasm_str,
                        target_state=c.target_state,
                        peak_prob=float(c.peak_prob),
                        num_qubits=c.num_qubits,
                        seed=int(seed_base) ^ (idx * 0x9E3779B1),
                        rqc_depth=params.rqc_depth,
                        created_at=created,
                    )
                )
            except Exception as e:
                bt.logging.error(f"[circuit-cache] Failed for circuit {idx}: {e}", exc_info=True)
                raise
        
        return artifacts
    
    def _load_manifest(self) -> None:
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, 'r') as f:
                    self._manifest = json.load(f)
            except Exception:
                self._manifest = {"entries": []}
        else:
            self._manifest = {"entries": []}
    
    def _save_manifest(self) -> None:
        with open(self._manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2)
    
    def _cleanup_old_entries(self, max_age_seconds: float = 3600, max_entries: int = 1000) -> None:
        try:
            now = time.time()
            entries = self._manifest.get("entries", [])
            
            kept_entries = []
            removed_count = 0
            for entry in entries:
                age = now - entry.get("created_ts", now)
                if entry.get("used", False) and age > max_age_seconds:
                    filepath = self._cache_dir / entry.get("filename", "")
                    if filepath.exists():
                        try:
                            filepath.unlink()
                            removed_count += 1
                        except Exception:
                            pass
                else:
                    kept_entries.append(entry)
            
            if len(kept_entries) > max_entries:
                kept_entries.sort(key=lambda x: x.get("created_ts", 0), reverse=True)
                for entry in kept_entries[max_entries:]:
                    filepath = self._cache_dir / entry.get("filename", "")
                    if filepath.exists():
                        try:
                            filepath.unlink()
                        except Exception:
                            pass
                kept_entries = kept_entries[:max_entries]
            
            if self._cache_dir.exists():
                manifest_files = {entry.get("filename", "") for entry in kept_entries}
                for filepath in self._cache_dir.glob("*.qasm"):
                    if filepath.name not in manifest_files:
                        try:
                            filepath.unlink()
                        except Exception:
                            pass
            
            if removed_count > 0 or len(entries) != len(kept_entries):
                self._manifest["entries"] = kept_entries
                self._save_manifest()
        except Exception:
            pass
    
    def _save_to_disk(self, artifact: CircuitArtifact, variant_idx: int) -> None:
        timestamp = int(time.time())
        filename = f"gen_{timestamp}_var{variant_idx:02d}_qubits_{artifact.num_qubits}.qasm"
        filepath = self._cache_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                f.write(artifact.qasm)
        except Exception as e:
            bt.logging.error(f"[circuit-cache] Failed to save circuit: {e}")
            return
        
        self._manifest["entries"].append({
            "filename": filename,
            "hash": hashlib.sha256(artifact.qasm.encode()).hexdigest(),
            "target": artifact.target_state,
            "used": False,
            "created_ts": time.time()
        })
        self._save_manifest()
    
    def _get_unused_from_disk(self, nqubits: int) -> Optional[tuple[str, str]]:
        for entry in self._manifest["entries"]:
            if not entry.get("used", True) and f"qubits_{nqubits}.qasm" in entry.get("filename", ""):
                filepath = self._cache_dir / entry["filename"]
                if filepath.exists():
                    try:
                        with open(filepath, 'r') as f:
                            qasm = f.read()
                        entry["used"] = True
                        self._save_manifest()
                        return qasm, entry.get("target", "")
                    except Exception as e:
                        bt.logging.error(f"[circuit-cache] Failed to load circuit from disk: {e}")
        return None


GLOBAL_CIRCUIT_POOL = CircuitPool()


