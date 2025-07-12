"""
Challenge Producer for Quantum (no longer async)
"""

from __future__ import annotations

import json
import time
import itertools
import threading
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Any, Tuple, Iterator, List
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
        self._uid_cycle: Iterator[int] = itertools.cycle(uid_list or [])

        self.queue: q.Queue[QItem] = q.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._validator = validator

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
                uid = next(self._uid_cycle)
                syn, meta, target, fp = self._make_payload(uid)
                self.queue.put_nowait(QItem(uid, syn, meta, target, fp))
                bt.logging.debug("[challenge-producer] queued challenge")
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