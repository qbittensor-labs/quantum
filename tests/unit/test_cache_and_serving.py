import os
import json
import time
from pathlib import Path
import numpy as np

from qbittensor.validator.peaked_circuit_creation.lib.circuit_gen import CircuitParams
from qbittensor.validator.peaked_circuit_creation.lib.circuit import SU4
from qbittensor.validator.peaked_circuit_creation.lib.base_cache import save_base_su4, load_base_su4, prune_expired


ART_OBF = Path("peaked_cache/obf_circuits")
MANIFEST = ART_OBF / "served_manifest.json"


def _reset_paths():
    ART_OBF.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        MANIFEST.unlink()
    for p in ART_OBF.glob("*.qasm"):
        p.unlink()


def test_base_su4_cache_roundtrip():
    from qbittensor.validator.peaked_circuit_creation.lib.base_cache import CACHE_DIR
    for p in CACHE_DIR.glob("*.npz"):
        p.unlink()

    params = CircuitParams.from_difficulty(-2.0)
    seed = 12345
    mats = np.stack([np.eye(4, dtype=np.complex128) for _ in range(10)])
    unis = [SU4(i, i+1, mats[i]) for i in range(0, 10)]
    target_state = "0" * params.nqubits

    path = save_base_su4(
        nqubits=params.nqubits,
        rqc_depth=params.rqc_depth,
        pqc_depth=params.pqc_depth,
        seed=seed,
        target_state=target_state,
        unis=unis,
        peak_prob=1.0,
        ttl_hours=0.0002,  # 0.72 TTL for test
    )
    assert path.exists()

    loaded = load_base_su4(
        nqubits=params.nqubits,
        rqc_depth=params.rqc_depth,
        pqc_depth=params.pqc_depth,
        seed=seed,
    )
    assert loaded is not None
    t_loaded, unis_loaded, pprob = loaded
    assert t_loaded == target_state
    assert len(unis_loaded) == len(unis)
    assert np.allclose(unis_loaded[0].mat, np.eye(4))

    time.sleep(1.0)
    removed = prune_expired()
    assert removed >= 1


def test_obf_serving_unique_and_tracking():
    from obf_cache_tests import serve_for_difficulty, _load_manifest
    _reset_paths()
    os.environ.setdefault("QBT_DECOMP_PER_SU4", "1")
    os.environ.setdefault("QBT_OBF_BOUNDARY_RATE", "0.0")
    os.environ.setdefault("QBT_OBF_SWAP_RATE", "0.0")

    diff = -2.0
    seed = 24680
    served = []
    served.append(serve_for_difficulty(diff, variants=3, seed=seed))
    # should not reuse the same file
    served.append(serve_for_difficulty(diff, variants=3, seed=seed))
    served.append(serve_for_difficulty(diff, variants=3, seed=seed))

    paths = [s["path"] for s in served]
    assert len(paths) == len(set(paths)), "served paths must be unique (no reuse)"

    m = _load_manifest()
    files = {e["filename"]: e for e in m["entries"]}
    for p in paths:
        fn = Path(p).name
        assert fn in files
        assert files[fn].get("used", False) is True
        assert isinstance(files[fn].get("target", ""), str)


