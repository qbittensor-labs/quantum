# coding: utf-8
"""
Persistent storage for the miner.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Tuple

import bittensor as bt

from .config import Paths

__all__ = ["Storage"]


class Storage:  # pylint: disable=too-few-public-methods
    """Handle *solutions*"""

    # Construction / bootstrap
    def __init__(self, paths: Paths) -> None:
        self.p = paths

        # quick look‑ups in memory (populated by _bootstrap())
        self._solved: Dict[str, str] = {}
        self._challenge_validators: Dict[str, str] = {}
        self._sent: set[str] = set()
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Scan existing files so we survive restarts gracefully."""
        for folder in (self.p.solved_root, self.p.solved_peaked, self.p.solved_shors):
            for fp in folder.glob("*.json"):
                try:
                    meta = json.loads(fp.read_text())
                    cid = meta["challenge_id"]
                    bits = meta.get("solution_bitstring") or meta.get(
                        "peak_bitstring", ""
                    )
                    self._solved[cid] = bits

                    if "validator_hotkey" in meta:
                        self._challenge_validators[cid] = meta["validator_hotkey"]
                except Exception:
                    bt.logging.debug(f"Storage bootstrap.bad solution file {fp.name}")
                    continue

    # Public API – solutions
    def save_solution(
        self,
        cid: str,
        bitstring: str,
        circuit_type: str,
        validator_hotkey: str | None = None,
    ) -> None:
        """Persist computed solution (peak bitstring or stabilizer string) to disk."""

        if circuit_type == "peaked":
            target_dir = self.p.solved_peaked
        elif circuit_type == "shors":
            target_dir = self.p.solved_shors
        else:
            target_dir = self.p.solved_root

        target_dir.mkdir(parents=True, exist_ok=True)

        self._solved[cid] = bitstring
        if validator_hotkey:
            self._challenge_validators[cid] = validator_hotkey

        payload = {
            "challenge_id": cid,
            "solution_bitstring": bitstring,
            "timestamp": time.time(),
        }
        if validator_hotkey:
            payload["validator_hotkey"] = validator_hotkey

        (target_dir / f"{cid}.json").write_text(json.dumps(payload))

        # House‑keeping – remove unsolved scraps
        for fp in self.p.unsolved.glob(f"{cid}*"):
            fp.unlink(missing_ok=True)

    def drain_unsent(
        self, max_count: int = 10, validator_hotkey: str | None = None
    ) -> List[Tuple[str, str]]:
        """Return ≤ max_count unsent solutions for the specified validator."""
        output: List[Tuple[str, str]] = []
        bt.logging.debug(
            f"[storage] drain_unsent: validator_hotkey={validator_hotkey}, solved={len(self._solved)}, sent={len(self._sent)}"
        )

        for cid, bitstring in self._solved.items():
            if len(output) >= max_count:
                break

            # skip if caller asked for a specific validator and this CID belongs to another
            if (
                validator_hotkey
                and self._challenge_validators.get(cid) != validator_hotkey
            ):
                continue

            output.append((cid, bitstring))
            self._sent.add(cid)

        validator_display = validator_hotkey[:10] if validator_hotkey and isinstance(validator_hotkey, str) else 'any'
        bt.logging.info(
            f"[storage] drain_unsent: returning {len(output)} solutions for validator {validator_display}"
        )
        return output

    # Convenience helpers
    def is_solved(self, cid: str) -> bool:
        return cid in self._solved
