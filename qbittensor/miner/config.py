from __future__ import annotations

"""
typed helpers to keep paths & numbers in
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Paths", "DEFAULT_SCAN_INTERVAL", "DEFAULT_QUEUE_SIZE"]

DEFAULT_SCAN_INTERVAL: float = 30.0  # seconds between idle directory polls
DEFAULT_QUEUE_SIZE: int = 1_000  # in‑memory backlog before we start dropping


@dataclass(frozen=True, slots=True)
class Paths:
    """
    Computed directory layout rooted at base.
    """

    base: Path
    unsolved: Path
    solved_root: Path
    solved_peaked: Path
    solved_hstab: Path
    solved_shors: Path

    @classmethod
    def from_base(cls, base: Path) -> "Paths":
        base = base.expanduser().resolve()
        return cls(
            base=base,
            unsolved=base / "unsolved_circuits",
            solved_root=base / "solved_circuits",
            solved_peaked=base / "peaked_circuits" / "solved_circuits",
            solved_hstab=base / "hstab_circuits" / "solved_circuits",
            solved_shors=base / "shors_circuits" / "solved_circuits",
        )
