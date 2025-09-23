import os
import time
import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from .circuit import SU4


CACHE_DIR = Path(os.getenv("QBT_BASE_SU4_CACHE_DIR", "peaked_cache/base_su4"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(nqubits: int, rqc_depth: int, pqc_depth: int, seed: int) -> str:
    return f"nq{nqubits}_r{rqc_depth}_p{pqc_depth}_seed{int(seed)}"


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.npz"


def save_base_su4(
    *,
    nqubits: int,
    rqc_depth: int,
    pqc_depth: int,
    seed: int,
    target_state: str,
    unis: List[SU4],
    peak_prob: float,
    ttl_hours: Optional[float] = None,
) -> Path:
    mats = np.stack([u.mat for u in unis]).astype(np.complex128)
    targets = np.array([[u.target0, u.target1] for u in unis], dtype=np.int16)
    now = time.time()
    ttl = float(os.getenv("QBT_BASE_CACHE_TTL_H", str(ttl_hours if ttl_hours is not None else 48)))
    meta = {
        "nqubits": int(nqubits),
        "rqc_depth": int(rqc_depth),
        "pqc_depth": int(pqc_depth),
        "seed": int(seed),
        "created_ts": now,
        "expires_ts": now + ttl * 3600.0,
        "peak_prob": float(peak_prob),
    }
    key = _cache_key(nqubits, rqc_depth, pqc_depth, seed)
    path = _cache_path(key)
    np.savez_compressed(
        path,
        unis_mat=mats,
        targets=targets,
        target_state=np.array(target_state),
        meta=np.array(json.dumps(meta)),
    )
    return path


def load_base_su4(
    *,
    nqubits: int,
    rqc_depth: int,
    pqc_depth: int,
    seed: int,
) -> Optional[Tuple[str, List[SU4], float]]:
    try:
        prune_expired()
    except Exception:
        pass
    key = _cache_key(nqubits, rqc_depth, pqc_depth, seed)
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        if time.time() > float(meta.get("expires_ts", 0)):
            return None
        target_state = str(data["target_state"])  # np.str_ -> str
        mats = data["unis_mat"]
        targets = data["targets"]
        unis: List[SU4] = [
            SU4(int(t[0]), int(t[1]), mats[i]) for i, t in enumerate(targets)
        ]
        peak_prob = float(meta.get("peak_prob", 0.0))
        return target_state, unis, peak_prob
    except Exception:
        return None


def prune_expired() -> int:
    removed = 0
    now = time.time()
    for p in CACHE_DIR.glob("*.npz"):
        try:
            d = np.load(p, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            if now > float(meta.get("expires_ts", 0)):
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed

try:
    prune_expired()
except Exception:
    pass


