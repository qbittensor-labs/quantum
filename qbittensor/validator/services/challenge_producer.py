"""
Challenge Producer for Quantum (no longer async)
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Any, Tuple, List
import queue as q
import bittensor as bt
from qbittensor.protocol import ChallengeCircuits
from qbittensor.validator.utils.challenge_utils import build_challenge
from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.config.difficulty_config import DifficultyConfig


DEFAULT_DIR = (Path(__file__).resolve().parent / ".." / "pending_challenges").resolve()
DEFAULT_DIR.mkdir(parents=True, exist_ok=True)


class QItem(NamedTuple):
    uid: int
    syn: ChallengeCircuits
    meta: ChallengeMeta
    target: str
    file_path: Path | None


class ChallengeProducer:
    """
    Spawn with .start(); consume with .queue.get().
    Internally runs a daemon thread that fills the queue.
    """

    _SLEEP          = 0.1
    _CLEAN_INTERVAL = 300

    def __init__(
        self,
        wallet,
        *,
        directory: Path = DEFAULT_DIR,
        queue_size: int = 64,
        difficulty: float = 0.0,
        diff_cfg: DifficultyConfig | None = None,
        batch_size: int = 1,
        uid_list: List[int] | None = None,
        validator=None,
    ):
        self._wallet      = wallet
        self._directory   = directory
        self._difficulty  = difficulty
        self._diff_cfg    = diff_cfg
        self._stash: dict[int, list[Tuple[Any, ...]]] = {}
        self._uid_list: List[int] = uid_list or []
        self._uid_index: int = 0
        self._uid_stats: dict[int, int] = {}
        self._total_processed = 0

        self.queue: q.Queue[QItem] = q.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._validator = validator

    @property
    def _uid_cycle(self):
        with self._uid_lock:
            if not self._uid_list:
                raise StopIteration("No UIDs available")
            
            current_index = self._uid_index
            uid = self._uid_list[current_index]
            self._uid_index = (self._uid_index + 1) % len(self._uid_list)
            self._uid_stats[uid] = self._uid_stats.get(uid, 0) + 1
            self._total_processed += 1
            
            return uid

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.cleanup_stash()
        self._thread = threading.Thread(
            target=self._loop,
            name="ChallengeProducer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.cleanup_stash()

    def _loop(self) -> None:
        bt.logging.info("[challenge-producer] thread started")
        last_cleanup = time.time()

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_cleanup > self._CLEAN_INTERVAL:
                self._cleanup_old_files()
                last_cleanup = now

            # If the validator is still waiting on a reply, pause.
            if getattr(self._validator, "_in_flight", set()):
                time.sleep(self._SLEEP)
                continue

            if self.queue.full():
                time.sleep(self._SLEEP)
                continue

            try:
                uid = self._uid_cycle
                syn, meta, target, fp = self._make_payload(uid)
                self.queue.put_nowait(QItem(uid, syn, meta, target, fp))
                bt.logging.debug(f"[challenge-producer] queued challenge for UID {uid}")
            except Exception:
                bt.logging.error("[challenge-producer] error", exc_info=True)

            time.sleep(self._SLEEP)

    def _make_payload(self, uid: int) -> Tuple[Any, ...]:
        """Return a single (syn, meta, target, file_path) tuple."""
        stash = self._stash.get(uid)
        if stash:
            return stash.pop(0)

        difficulty = (
            float(self._diff_cfg.get(uid)) if self._diff_cfg else self._difficulty
        )
        syn, meta, target = build_challenge(
            wallet=self._wallet,
            difficulty=difficulty,
        )
        syn.difficulty_level = difficulty
        fp = self._write_to_disk(meta, syn, target)
        return syn, meta, target, fp

    def cleanup_stash(self) -> None:
        for uid, items in self._stash.items():
            for _, _, _, fp in items:
                self._cleanup_file(fp)
            bt.logging.info(f"[cleanup] removed {len(items)} stashed challenges for UID {uid}")
        self._stash.clear()

    _uid_lock = threading.Lock()

    def update_uid_list(self, new_uids):
        with self._uid_lock:
            current_uid = self._uid_list[self._uid_index] if self._uid_list and self._uid_index < len(self._uid_list) else None
            last_processed_uid = None
            if self._uid_list:
                if self._uid_index > 0:
                    last_processed_uid = self._uid_list[self._uid_index - 1]
                elif self._total_processed > 0:
                    last_processed_uid = self._uid_list[-1]
            
            self._uid_list = new_uids
            
            if current_uid and current_uid not in new_uids:
                self._uid_index = 0
                bt.logging.debug(f"[challenge-producer] reset to start (current UID {current_uid} removed)")
            
            elif last_processed_uid and last_processed_uid in new_uids:
                last_index = new_uids.index(last_processed_uid)
                self._uid_index = (last_index + 1) % len(new_uids)
                bt.logging.debug(f"[challenge-producer] preserved position after UID {last_processed_uid}")
            else:
                self._uid_index = 0
                bt.logging.debug(f"[challenge-producer] reset to start")
            
            self._stash = {uid: self._stash.get(uid, []) for uid in new_uids}
            self._uid_stats = {uid: count for uid, count in self._uid_stats.items() if uid in new_uids}
            bt.logging.info(f"[challenge-producer] UID list updated: {len(new_uids)} miners")

    # file management
    def _cleanup_old_files(self, max_age_seconds: int = 86_400) -> None:
        try:
            removed = 0
            for fp in self._directory.glob("*.json"):
                if time.time() - fp.stat().st_mtime > max_age_seconds:
                    fp.unlink(missing_ok=True)
                    removed += 1
            if removed:
                bt.logging.info(f"[challenge-producer] cleaned {removed} old files")
        except Exception as e:
            bt.logging.warning(f"[challenge-producer] cleanup error: {e}")

    # File management helpers
    def _write_to_disk(
        self,
        meta: ChallengeMeta,
        syn: ChallengeCircuits,
        target: str
    ) -> Path:
        return # removing the writes entirely; currently unused
        fp = self._directory / f"{meta.challenge_id}.json"
        with fp.open("w") as f:
            json.dump(
                {
                    "circuit_data": syn.circuit_data,
                    "meta"        : asdict(meta),
                    "target"      : target,
                },
                f,
            )
        return fp

    def _cleanup_file(self, fp: Path | None) -> None:
        if fp and fp.exists():
            try:
                fp.unlink(missing_ok=True)
            except Exception:
                bt.logging.warning(f"[cleanup] could not delete {fp}")