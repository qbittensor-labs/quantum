from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Sequence

import bittensor as bt


class DifficultyConfig:
    """Thread-safe wrapper around <uid → difficulty> map stored as JSON.
    """

    def __init__(
        self,
        path: Path,
        uids: Sequence[int],
        default: float = 0.0,
    ):
 
        self._path = path
        self._lock = threading.Lock()
        self._default = default
        self._uids = list(uids)
        self._table: Dict[int, float] = self._load()

    def get(self, uid: int) -> float:
        """Retrieve the difficulty for a given UID (falls back to default if missing)."""
        with self._lock:
            return self._table.get(uid, self._default)

    def set(self, uid: int, value: float) -> None:
        with self._lock:
            self._table[uid] = float(value)
            self._dump()
            bt.logging.info(f"uid {uid} with diff {value}")

    def _load(self) -> Dict[int, float]:
        """Load the table from disk, backfilling any missing UIDs."""
        # If file exists, load and convert keys to ints
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            table: Dict[int, float] = {}
            for k, v in raw.items():
                try:
                    uid_int = int(k)
                except ValueError:
                    continue
                table[uid_int] = float(v)
            # Backfill any missing UIDs
            updated = False
            for uid in self._uids:
                if uid not in table:
                    table[uid] = self._default
                    updated = True
            # Persist if we added any defaults
            if updated:
                self._dump(table)
            return table

        # First-time initialization: seed with all UIDs at default
        table = {uid: self._default for uid in self._uids}
        self._dump(table)
        return table

    def _dump(self, table: Dict[int, float] | None = None) -> None:
        """Atomically write the current table to disk as JSON with string keys."""
        data = table if table is not None else self._table
        # Convert keys to strings for JSON
        serializable = {str(k): v for k, v in data.items()}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(serializable, indent=2))
        tmp.replace(self._path)
