"""
ChallengeProducer - queues hstab and peaked challenges for miners.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import bittensor as bt
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.utils.challenge_utils import (
    build_peaked_challenge,
    build_hstab_challenge,
)
from qbittensor.validator.utils.validator_meta import ChallengeMeta

__all__ = ["ChallengeProducer", "QItem"]


class QItem(NamedTuple):
    uid: int
    syn: Any  # general synapse type for new circuits
    meta: ChallengeMeta
    target: str
    file_path: Path | None


# define dataclass class for challenges
@dataclass(slots=True)
class _KindStrategy:
    weight: float
    builder: callable

class ChallengeProducer:
    """
    Fills a queue with challenge. Backround thread
    """

    _SLEEP_S: float = 0.1
    _CLEAN_INTERVAL_S: int = 300

    def __init__(
        self,
        wallet,
        *,
        directory: Path | None = None,
        queue_size: int = 64,
        default_difficulty: float = 0.0,
        diff_cfg: Dict[str, DifficultyConfig],
        uid_list: List[int] | None = None,
        validator=None,
    ) -> None:
        self._wallet = wallet
        self._directory = (
            directory
            or (Path(__file__).resolve().parent / ".." / "pending_challenges").resolve()
        )
        self._directory.mkdir(parents=True, exist_ok=True)

        self.queue: "queue.Queue[QItem]" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._stash: Dict[int, List[Tuple[Any, ...]]] = {}
        self._uid_list: List[int] = uid_list or []
        self._uid_index: int = 0
        self._uid_lock = threading.Lock()

        self._validator = validator
        self._diff_cfg = diff_cfg
        self._default_difficulty = default_difficulty

        # strategy
        self._strategies: Dict[str, _KindStrategy] = {
            "peaked": _KindStrategy(0.5, build_peaked_challenge),
            "hstab": _KindStrategy(0.5, build_hstab_challenge),
        }
        self._weights_cache = [s.weight for s in self._strategies.values()]

    # Start producer
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.cleanup_stash()

        self._thread = threading.Thread(
            target=self._loop, name="ChallengeProducer", daemon=True
        )
        self._thread.start()
        bt.logging.info("[challenge-producer] thread started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.cleanup_stash()

    # Public
    def update_uid_list(self, new_uids: List[int]) -> None:
        """
        Replace the UID ring while preserving position so we dont
        miss a subset of miners when the set changes.
        """
        with self._uid_lock:
            current_uid = (
                self._uid_list[self._uid_index]
                if self._uid_list and self._uid_index < len(self._uid_list)
                else None
            )

            self._uid_list = new_uids
            if current_uid in new_uids:
                self._uid_index = (new_uids.index(current_uid) + 1) % len(new_uids)
            else:
                self._uid_index = 0

            # prune stash for removed miners
            self._stash = {u: self._stash.get(u, []) for u in new_uids}

            bt.logging.info(
                f"[challenge-producer] UID list updated - {len(new_uids)} miners"
            )

    # Thread body
    def _loop(self) -> None:
        last_cleanup = time.time()

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_cleanup > self._CLEAN_INTERVAL_S:
                self._cleanup_old_files()
                last_cleanup = now

            if self.queue.full() or getattr(self._validator, "_in_flight", set()):
                time.sleep(self._SLEEP_S)
                continue

            try:
                uid = self._next_uid()
                syn, meta, target = self._build_challenge_for_uid(uid)
                self.queue.put_nowait(QItem(uid, syn, meta, target, file_path=None))
                bt.logging.trace(
                    f"[challenge-producer] queued {meta.circuit_kind} for UID {uid}"
                )
            except Exception:
                bt.logging.error("[challenge-producer] error", exc_info=True)

            time.sleep(self._SLEEP_S)

    # Challenge creation
    def _build_challenge_for_uid(self, uid: int) -> Tuple[Any, ChallengeMeta, str]:
        kind = random.choices(
            population=list(self._strategies.keys()),
            weights=self._weights_cache,
            k=1,
        )[0]

        strat = self._strategies[kind]
        difficulty = self._diff_cfg[kind].get(uid)
        syn, meta, target = strat.builder(
            wallet=self._wallet,
            difficulty=difficulty,
        )
        return syn, meta, target

    # UID handler
    def _next_uid(self) -> int:
        with self._uid_lock:
            if not self._uid_list:
                raise RuntimeError("No UIDs available")
            uid = self._uid_list[self._uid_index]
            self._uid_index = (self._uid_index + 1) % len(self._uid_list)
            return uid

    def cleanup_stash(self) -> None:
        for uid, items in self._stash.items():
            for *_, fp in items:
                self._cleanup_file(fp)
            bt.logging.debug(
                f"[cleanup] removed {len(items)} stashed challenges for UID {uid}"
            )
        self._stash.clear()

    def _cleanup_file(self, fp: Path | None) -> None:
        pass

    def _cleanup_old_files(self) -> None:
        pass
