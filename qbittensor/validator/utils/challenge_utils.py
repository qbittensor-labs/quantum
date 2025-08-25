"""
Utilities for generating circuits
"""

from __future__ import annotations

import random
import time
import hashlib
from typing import Iterable, Tuple
import subprocess
import tempfile
import json
import os
from pathlib import Path
import gc
import sys
import numpy as np

import bittensor as bt
import stim

from qbittensor.protocol import (
    ChallengePeakedCircuit,
    ChallengeHStabCircuit,
)
from qbittensor.validator.hidden_stabilizers_creation.lib import make_gen
from qbittensor.validator.peaked_circuit_creation.quimb_cache_utils import (
    clear_all_quimb_caches,
)
from qbittensor.validator.hidden_stabilizers_creation.lib.circuit_gen import HStabCircuit

from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.utils.crypto_utils import canonical_hash

__all__ = [
    "build_peaked_challenge",
    "build_hstab_challenge",
]
def _clear_memory() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass
    try:
        clear_all_quimb_caches()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _convert_peaked_difficulty_to_qubits(level: float) -> int:
    try:
        lvl = float(level)
    except Exception:
        lvl = 0.0
    if 0.0 <= lvl <= 10.0:
        nqubits = int(12 + 10 * np.log2(lvl + 3.9))
    else:
        nqubits = int(round(lvl))
    return max(16, min(nqubits, 39))

# Peaked circuits

def build_peaked_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengePeakedCircuit, ChallengeMeta, str]:
    """
    Build a peaked circuit challenge
    """
    seed = time.time_ns() % (2**32)

    nqubits = _convert_peaked_difficulty_to_qubits(difficulty)
    level = _convert_qubits_to_peaked_difficulty(nqubits)
    try:
        bt.logging.info(
            f"[peaked] requested diff={float(difficulty):.2f} -> nqubits={int(nqubits)} (level={float(level):.3f})"
        )
    except Exception:
        pass

    rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
    rqc_depth = int(round(rqc_mul * nqubits))

    gpu_id = int(os.getenv("VALIDATOR_GPU_ID", "0"))
    timeout_s = float(os.getenv("VALIDATOR_GEN_TIMEOUT_S", "1000"))

    with tempfile.TemporaryDirectory(prefix="peaked_gen_") as tmpdir:
        outdir = Path(tmpdir)
        worker_path = (Path(__file__).resolve().parents[1] / "services" / "gen_worker.py").resolve()

        used_seed = None
        metadata = None
        qasm = None

        for attempt in range(3):
            attempt_seed = int(seed) if attempt == 0 else random.randrange(1 << 32)
            cmd = [
                sys.executable,
                str(worker_path),
                "--difficulty",
                str(level),
                "--seed",
                str(attempt_seed),
                "--gpu-id",
                str(gpu_id),
                "--output-dir",
                str(outdir),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 timed out (seed={attempt_seed})")
                time.sleep(2.0)
                continue
            except Exception as e:
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 failed to launch (seed={attempt_seed}): {e}")
                time.sleep(1.0)
                continue

            if proc.returncode != 0:
                tail = ""
                try:
                    logs = sorted(outdir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if logs:
                        data = logs[0].read_text()
                        tail = data[-2000:]
                except Exception:
                    pass
                bt.logging.warning(
                    f"[peaked] attempt {attempt+1}/3 failed rc={proc.returncode} (seed={attempt_seed}); stderr=\n{(proc.stderr or '')[:800]}\nlog_tail=\n{tail}"
                )
                time.sleep(1.0)
                continue

            if proc.stdout:
                try:
                    _ = json.loads(proc.stdout.strip().splitlines()[-1])
                except Exception:
                    pass

            meta_fp = None
            for fp in outdir.glob(f"seed_{attempt_seed}_*_metadata.json"):
                meta_fp = fp
                break
            if not meta_fp:
                metas = list(outdir.glob("*_metadata.json"))
                meta_fp = metas[0] if metas else None
            if not meta_fp:
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 produced no metadata (seed={attempt_seed})")
                time.sleep(1.0)
                continue

            try:
                metadata = json.loads(Path(meta_fp).read_text())
            except Exception:
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 could not parse metadata (seed={attempt_seed})")
                time.sleep(0.5)
                continue
            qasm_file = outdir / metadata.get("qasm_filename", "")
            if not qasm_file.exists():
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 missing qasm file (seed={attempt_seed})")
                time.sleep(0.5)
                continue
            try:
                qasm = qasm_file.read_text()
            except Exception:
                bt.logging.warning(f"[peaked] attempt {attempt+1}/3 failed to read qasm (seed={attempt_seed})")
                time.sleep(0.5)
                continue

            used_seed = attempt_seed
            break

        if metadata is None or qasm is None or used_seed is None:
            bt.logging.error(f"[peaked] all attempts failed for nqubits={nqubits}")
            raise RuntimeError("generation worker failed")

        unsigned = {
            "seed": int(metadata.get("seed", used_seed)),
            "circuit_data": qasm,
            "difficulty_level": float(nqubits),
            "validator_hotkey": wallet.hotkey.ss58_address,
            "nqubits": nqubits,
            "rqc_depth": rqc_depth,
        }
        unsigned["challenge_id"] = canonical_hash(unsigned)

        syn = ChallengePeakedCircuit(**unsigned, validator_signature=None)
        meta = ChallengeMeta(
            challenge_id=unsigned["challenge_id"],
            circuit_kind="peaked",
            difficulty=float(nqubits),
            validator_hotkey=wallet.hotkey.ss58_address,
            entanglement_entropy=0.0,
            nqubits=nqubits,
            rqc_depth=rqc_depth,
        )

        target_state = metadata.get("target_state", "")
        return syn, meta, target_state


# Hidden Stabiliser circuits

def build_hstab_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengeHStabCircuit, ChallengeMeta, str]:
    """
    Build an H-Stab circuit challenge.
    """
    nqubits: int = max(26, round(difficulty))
    seed = random.randrange(1 << 30)

    try:
        generator = make_gen(seed)
        circ: HStabCircuit = HStabCircuit.make_circuit(generator, nqubits)
        qasm = circ.to_qasm()
    finally:
        _clear_memory()
    flat_solution = _flatten_hstab_string(circ.stabilizers)
    cid = hashlib.sha256(qasm.encode()).hexdigest()

    syn = ChallengeHStabCircuit(
        circuit_data=qasm,
        challenge_id=cid,
        difficulty_level=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        validator_signature=None,
    )
    meta = ChallengeMeta(
        challenge_id=cid,
        circuit_kind="hstab",
        difficulty=difficulty,
        validator_hotkey=wallet.hotkey.ss58_address,
        entanglement_entropy=0.0,
        nqubits=nqubits,
        rqc_depth=0,
    )
    return syn, meta, flat_solution


# Helper
def _flatten_hstab_string(stabilizers: Iterable[stim.PauliString]) -> str:
    """flatten an iterable of `stim.PauliString` -> single str"""
    return "".join(map(str, stabilizers))

def _params_from_difficulty(level: float) -> Tuple[int, int]:
    """
    Interpret `level` as desired number of qubits for peaked circuits.
    Returns (nqubits, rqc_depth).
    """
    nqubits = int(round(level))
    # Cap at 39 qubits
    nqubits = max(16, min(nqubits, 39))
    rqc_mul = 150 * np.exp(-nqubits / 4) + 0.5
    rqc_depth = int(round(rqc_mul * nqubits))
    return nqubits, rqc_depth


def _convert_qubits_to_peaked_difficulty(nqubits: int) -> float:
    """
    Inverse mapping for legacy difficulty → nqubits relation used by CircuitParams.
    nqubits ≈ 12 + 10 * log2(level + 3.9) ⇒ level ≈ 2 ** ((nqubits - 12) / 10) - 3.9
    """
    n = max(2, int(nqubits))
    return max(0.0, (2 ** ((n - 12) / 10.0)) - 3.9)
