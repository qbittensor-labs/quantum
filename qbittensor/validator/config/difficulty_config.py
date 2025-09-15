from __future__ import annotations

import json
import threading
import math
from pathlib import Path
from typing import Dict, Sequence, Callable
from qbittensor.validator.config.sql_utils import max_solved_difficulty

import bittensor as bt

HotkeyLookup = Callable[[int], str]

class DifficultyConfig:
    """Thread-safe wrapper around <uid → difficulty> map stored as JSON.
    """

    UNRESTRICTED_CEILING = 0.7 # can increase to 0.7 at any time
    MAX_ABS_INCREASE   = 0.4 # max increase is 0.4 after 0.7
    MIN_DIFFICULTY     = 0.0

    def __init__(
        self,
        path: Path,
        uids: Sequence[int],
        default: float = 0.0,
        *,
        db_path: Path | None = None,
        hotkey_lookup: HotkeyLookup | None = None,
        clamp: bool = True,
    ):
 
        self._path = path
        self._lock = threading.Lock()
        self._default = default
        self._uids = list(uids)
        self._table: Dict[int, float] = self._load()
        self._db_path       = db_path
        self._hotkey_lookup = hotkey_lookup
        self._clamp         = clamp

    def get(self, uid: int) -> float:
        """Retrieve the difficulty for a given UID (falls back to default if missing)."""
        with self._lock:
            val = self._table.get(uid, self._default)
            try:
                f = float(val)
            except Exception:
                return self._default
            return f if math.isfinite(f) and f >= self.MIN_DIFFICULTY else self._default

    def set(self, uid: int, value: float) -> bool:
        """
        Update a miners difficulty.
        """
        # reject non-finite or negative values
        try:
            value = float(value)
        except Exception:
            return False
        if not math.isfinite(value) or value < self.MIN_DIFFICULTY:
            return False # Can't set a negative difficulty

        with self._lock:
            cfg_current = self._table.get(uid, self._default)

            db_current = 0.0
            if self._db_path and self._hotkey_lookup:
                miner_hotkey = self._hotkey_lookup(uid)
                db_current   = max_solved_difficulty(self._db_path, miner_hotkey)
                # sanitize DB read
                try:
                    db_current = float(db_current)
                except Exception:
                    db_current = 0.0
                if not math.isfinite(db_current) or db_current < self.MIN_DIFFICULTY:
                    db_current = 0.0

            # whichever is higher becomes the baseline
            current = max(cfg_current, db_current) # db_current empty for Hstab

            # downward moves are free
            if value <= current:
                new_val = value

            elif not self._clamp:
                new_val = value
            else:
                # Anything ≤ 0.7 is always allowed
                if value <= self.UNRESTRICTED_CEILING:
                    new_val = value
                else:
                     # We’re already past the ceiling
                    if current < self.UNRESTRICTED_CEILING:
                        # Jump only as far as the ceiling
                        new_val = self.UNRESTRICTED_CEILING
                    else:
                        # Clamp to current + 0.4
                        allowed_max = current + self.MAX_ABS_INCREASE
                        new_val = min(value, allowed_max)

            if new_val != current:
                # final guard before storing
                if not math.isfinite(new_val) or new_val < self.MIN_DIFFICULTY:
                    return False
                self._table[uid] = float(new_val)
                self._dump()
                bt.logging.trace(f"uid {uid} difficulty now {new_val}")
                return True

            return False

    def update_uid_list(self, new_uids):
        """Replace the live UID roster and back-fill brand-new miners."""
        with self._lock:
            self._uids = list(new_uids)
            changed = False
            for uid in new_uids:
                if uid not in self._table:
                    self._table[uid] = self._default
                    changed = True
            if changed:
                self._dump()


    def _load(self) -> Dict[int, float]:
        """Load the table from disk, backfilling any missing UIDs."""
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            table: Dict[int, float] = {}
            for k, v in raw.items():
                try:
                    uid_int = int(k)
                except ValueError:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    fv = self._default
                if not math.isfinite(fv) or fv < self.MIN_DIFFICULTY:
                    fv = self._default
                table[uid_int] = fv
            updated = False
            for uid in self._uids:
                if uid not in table:
                    table[uid] = self._default
                    updated = True
            if updated:
                self._dump(table)
            return table

        table = {uid: self._default for uid in self._uids}
        self._dump(table)
        return table

    def _dump(self, table: Dict[int, float] | None = None) -> None:
        """Atomically write the current table to disk as JSON with string keys."""
        data = table if table is not None else self._table
        serializable = {}
        for k, v in data.items():
            try:
                fv = float(v)
                if not math.isfinite(fv) or fv < self.MIN_DIFFICULTY:
                    fv = self._default
            except Exception:
                fv = self._default
            serializable[str(k)] = fv
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(serializable, indent=2, allow_nan=False))
        tmp.replace(self._path)
