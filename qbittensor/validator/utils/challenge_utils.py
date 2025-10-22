"""
Utilities for generating circuits
"""

from __future__ import annotations

import random
import secrets
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
from qbittensor.validator.services.circuit_cache import GLOBAL_CIRCUIT_POOL
from qbittensor.validator.utils.crypto_utils import canonical_hash

__all__ = [
    "build_peaked_challenge",
    "build_hstab_challenge",
]

# clear error type surfaced to the validator loop
class ValidatorOOMError(RuntimeError):
    pass

# dynamic cap state
_PEAKED_CAP_DEFAULT = 39
_PEAKED_CAP_MIN = 16
_RUN_DIR = Path.cwd()
_PEAKED_CAP_FILE = _RUN_DIR / "validator_peaked_max_cap"
_PEAKED_OOM_COUNT_FILE = _RUN_DIR / "validator_peaked_oom_count"
_PEAKED_OOMS_PER_STEP = 4

# local runtime knobs
_GPU_RESET_ON_OOM = True
_GPU_ID = 0
_GEN_TIMEOUT_S = 1200.0

def _read_int_file(path: str | Path, default: int) -> int:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return default

def _write_int_file(path: str | Path, value: int) -> None:
    try:
        with open(path, "w") as f:
            f.write(str(int(value)))
    except Exception:
        pass

def _current_peaked_max_cap() -> int:
    cap = _read_int_file(_PEAKED_CAP_FILE, _PEAKED_CAP_DEFAULT)
    cap = max(_PEAKED_CAP_MIN, min(int(cap), _PEAKED_CAP_DEFAULT))
    return cap

def _register_peaked_oom_and_maybe_lower_cap() -> None:
    # increment OOM counter
    cnt = _read_int_file(_PEAKED_OOM_COUNT_FILE, 0) + 1
    _write_int_file(_PEAKED_OOM_COUNT_FILE, cnt)
    # every N OOMs, reduce cap by 1 (not below minimum), reset counter
    if cnt >= _PEAKED_OOMS_PER_STEP:
        cap = _current_peaked_max_cap()
        new_cap = max(_PEAKED_CAP_MIN, cap - 1)
        if new_cap != cap:
            bt.logging.warning(f"[peaked] lowering dynamic max cap from {cap} to {new_cap} after {cnt} OOMs")
            _write_int_file(_PEAKED_CAP_FILE, new_cap)
        _write_int_file(_PEAKED_OOM_COUNT_FILE, 0)

def _clear_memory() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
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

def _run_cmd(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout: {e}"
    except Exception as e:
        return 125, "", f"error: {e}"

def _kill_gen_worker() -> None:
    try:
        rc, out, err = _run_cmd(["pkill", "-f", "gen_worker.py"], timeout=10)
        if rc == 0:
            bt.logging.info("[peaked][oom] pkill gen_worker.py -> rc=0")
        elif rc in (1,):
            bt.logging.info("[peaked][oom] pkill gen_worker.py -> no processes")
        else:
            bt.logging.warning(f"[peaked][oom] pkill gen_worker.py rc={rc} stderr={err}")
    except Exception:
        pass

def _maybe_gpu_device_reset(gpu_id: int) -> None:
    try:
        if not _GPU_RESET_ON_OOM:
            return
        cmds: list[list[str]] = []
        # try non-sudo first
        cmds.append(["nvidia-smi", "--gpu-reset", "-i", str(gpu_id)])
        # then sudo -n (non-interactive fail-fast)
        cmds.append(["sudo", "-n", "nvidia-smi", "--gpu-reset", "-i", str(gpu_id)])
        # reload driver
        cmds.append(["sudo", "-n", "rmmod", "nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"])
        cmds.append(["sudo", "-n", "modprobe", "nvidia"])
        # MIG toggle attempts (non-sudo, then sudo -n)
        for base in (["nvidia-smi"], ["sudo", "-n", "nvidia-smi"]):
            cmds.append(base + ["-i", str(gpu_id), "-mig", "0"])
            cmds.append(base + ["-i", str(gpu_id), "-mig", "1"])
            cmds.append(base + ["-i", str(gpu_id), "-mig", "0"])

        for c in cmds:
            rc, out, err = _run_cmd(c, timeout=60)
            bt.logging.info(f"[peaked][oom] {' '.join(c)} -> rc={rc}")
            if err:
                bt.logging.warning(f"[peaked][oom] stderr: {err}")
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
    # changed hard cap 39 -> dynamic cap
    return max(16, min(nqubits, _current_peaked_max_cap()))

# Peaked circuits

def build_peaked_challenge(
    *, wallet: bt.wallet, difficulty: float
) -> Tuple[ChallengePeakedCircuit, ChallengeMeta, str]:
    """
    Build a peaked circuit challenge
    """
    seed = secrets.randbits(128)

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

    gpu_id = _GPU_ID
    timeout_s = _GEN_TIMEOUT_S

    try:
        art = GLOBAL_CIRCUIT_POOL.get(nqubits)
        qasm = art.qasm
        used_seed = int(art.seed)
        rqc_depth = int(art.rqc_depth)
        metadata = {
            "seed": used_seed,
            "qasm_filename": "",
            "target_state": art.target_state,
        }
    except Exception:
        with tempfile.TemporaryDirectory(prefix="peaked_gen_") as tmpdir:
            outdir = Path(tmpdir)
            worker_path = (Path(__file__).resolve().parents[1] / "services" / "gen_worker.py").resolve()

            used_seed = None
            metadata = None
            qasm = None

            for attempt in range(2):
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
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 timed out (seed={attempt_seed})")
                    time.sleep(2.0)
                    continue
                except Exception as e:
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 failed to launch (seed={attempt_seed}): {e}")
                    time.sleep(1.0)
                    continue

                if proc.returncode == 99:
                    try: _kill_gen_worker()
                    except Exception: pass
                    try: _clear_memory()
                    except Exception: pass
                    time.sleep(1.0)
                    bt.logging.error(f"[peaked] OOM in worker (seed={attempt_seed}); stderr=\n{(proc.stderr or '')[:800]}")
                    _register_peaked_oom_and_maybe_lower_cap()
                    raise ValidatorOOMError("GPU OOM in peaked circuit worker")

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
                        f"[peaked] attempt {attempt+1}/2 failed rc={proc.returncode} (seed={attempt_seed}); stderr=\n{(proc.stderr or '')[:800]}\nlog_tail=\n{tail}"
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
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 produced no metadata (seed={attempt_seed})")
                    time.sleep(2.0)
                    continue

                try:
                    metadata = json.loads(Path(meta_fp).read_text())
                except Exception:
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 could not parse metadata (seed={attempt_seed})")
                    time.sleep(0.5)
                    continue
                qasm_file = outdir / metadata.get("qasm_filename", "")
                if not qasm_file.exists():
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 missing qasm file (seed={attempt_seed})")
                    time.sleep(0.5)
                    continue
                try:
                    qasm = qasm_file.read_text()
                except Exception:
                    bt.logging.warning(f"[peaked] attempt {attempt+1}/2 failed to read qasm (seed={attempt_seed})")
                    time.sleep(0.5)
                    continue

                used_seed = attempt_seed
                break

            if metadata is None or qasm is None or used_seed is None:
                bt.logging.error(f"[peaked] all attempts failed for nqubits={nqubits}")
                raise RuntimeError("generation worker failed")

    unsigned = {
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
    nqubits = max(16, min(nqubits, _current_peaked_max_cap()))
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
