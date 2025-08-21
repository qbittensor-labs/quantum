from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import bittensor as bt

def _metrics_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "services" / "metrics"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _append_csv(file_path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    try:
        is_new = not file_path.exists()
        with file_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        bt.logging.debug(f"[metrics] CSV write failed for {file_path.name}: {exc}")


def log_generation_metric(
    *,
    kind: str,
    nqubits: int,
    rqc_depth: int,
    seed: int,
    duration_s: float,
    device: str,
    target_peaking: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    try:
        ts = datetime.now(timezone.utc).isoformat()
        path = _metrics_dir() / "generation_metrics.csv"
        fields = [
            "timestamp_utc",
            "kind",
            "nqubits",
            "rqc_depth",
            "seed",
            "duration_s",
            "device",
            "target_peaking",
            "success",
            "error",
        ]
        row = {
            "timestamp_utc": ts,
            "kind": kind,
            "nqubits": int(nqubits),
            "rqc_depth": int(rqc_depth),
            "seed": int(seed),
            "duration_s": float(duration_s),
            "device": device,
            "target_peaking": (None if target_peaking is None else float(target_peaking)),
            "success": bool(success),
            "error": (str(error)[:256] if error else None),
        }
        _append_csv(path, fields, row)
    except Exception as exc:
        bt.logging.debug(f"[metrics] generation log failed: {exc}")


def log_miner_roundtrip(
    *,
    uid: int,
    miner_hotkey: str,
    kind: str,
    nqubits: int,
    difficulty: float,
    duration_s: float,
    num_solutions: int,
    correct_solutions: int,
) -> None:
    try:
        ts = datetime.now(timezone.utc).isoformat()
        path = _metrics_dir() / "miner_roundtrip_metrics.csv"
        fields = [
            "timestamp_utc",
            "uid",
            "miner_hotkey",
            "kind",
            "nqubits",
            "difficulty",
            "duration_s",
            "num_solutions",
            "correct_solutions",
        ]
        row = {
            "timestamp_utc": ts,
            "uid": int(uid),
            "miner_hotkey": miner_hotkey,
            "kind": kind,
            "nqubits": int(nqubits),
            "difficulty": float(difficulty),
            "duration_s": float(duration_s),
            "num_solutions": int(num_solutions),
            "correct_solutions": int(correct_solutions),
        }
        _append_csv(path, fields, row)
    except Exception as exc:
        bt.logging.debug(f"[metrics] miner roundtrip log failed: {exc}")

