"""
ChallengeProducer - queues hstab and peaked challenges for miners.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple
import math

import bittensor as bt
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.utils.challenge_utils import (
    build_peaked_challenge,
    build_hstab_challenge,
)
from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.utils.challenge_utils import ValidatorOOMError

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
        self._MAX_DIFFICULTY = {"hstab": 50.0}

        # strategy
        self._strategies: Dict[str, _KindStrategy] = {
            "peaked": _KindStrategy(0.5, build_peaked_challenge),
            "hstab": _KindStrategy(0.5, build_hstab_challenge),
        }
        self._weights_cache = [s.weight for s in self._strategies.values()]

        # one peaked one hstab per UID
        self._pending_uid: int | None = None
        self._pending_kind: str | None = None

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
        bt.logging.info("[challenge-producer] thread stopped")

    # Public
    def update_uid_list(self, new_uids: List[int]) -> None:
        """
        Replace the UID ring while preserving position:
        - keep the same UID if it still exists,
        - otherwise advance through the old list (wrap-around) until a match is found.
        """
        with self._uid_lock:
            old_list = self._uid_list or []
            has_valid_idx = old_list and 0 <= self._uid_index < len(old_list)
            current_uid = old_list[self._uid_index] if has_valid_idx else None

            new_set = set(new_uids)

            # Decide which UID we want to land on
            target_uid = None
            if current_uid in new_set:
                # keep the same UID
                target_uid = current_uid
            else:
                # move to the next from the old list, wrapping, until a match is found
                if has_valid_idx and old_list:
                    for step in range(1, len(old_list) + 1):
                        cand = old_list[(self._uid_index + step) % len(old_list)]
                        if cand in new_set:
                            target_uid = cand
                            break

            # Apply the new list and set the index
            self._uid_list = new_uids
            if new_uids and target_uid is not None:
                self._uid_index = new_uids.index(target_uid)
            else:
                # fallback if no matches or empty list
                self._uid_index = 0

            # prune stash for removed miners
            self._stash = {u: self._stash.get(u, []) for u in new_uids}

            bt.logging.info(f"[challenge-producer] UID list updated - {len(new_uids)} miners")

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
                if self._pending_uid is not None and self._pending_kind is not None:
                    uid = self._pending_uid
                    syn, meta, target = self._build_challenge_of_kind(uid, self._pending_kind)
                    self.queue.put_nowait(QItem(uid, syn, meta, target, file_path=None))
                    bt.logging.trace(
                        f"[challenge-producer] queued {meta.circuit_kind} for UID {uid} (pending)"
                    )
                    self._pending_uid = None
                    self._pending_kind = None
                else:
                    uid = self._next_uid()

                    room_for_two = (
                        (self.queue.maxsize == 0) or
                        (self.queue.qsize() <= max(0, self.queue.maxsize - 2))
                    )

                    # peaked
                    try:
                        syn1, meta1, target1 = self._build_challenge_of_kind(uid, "peaked")
                        self.queue.put_nowait(QItem(uid, syn1, meta1, target1, file_path=None))
                        bt.logging.trace(
                            f"[challenge-producer] queued {meta1.circuit_kind} for UID {uid}"
                        )
                    except ValidatorOOMError:
                        self._pending_uid = uid
                        self._pending_kind = "peaked"
                        bt.logging.error("[challenge-producer] OOM during peaked build; attempting GPU reset script if provided")
                        try:
                            import os, sys
                            gpu_id = int(os.getenv("VALIDATOR_GPU_ID", "0"))
                            script = os.getenv("GPU_RESET_SCRIPT", "/root/quantum/gpu-reset-and-exec.sh")
                            cmd = [script, sys.executable, *sys.argv]
                            env = os.environ.copy()
                            env["GPU_ID"] = str(gpu_id)
                            if script and os.path.exists(script):
                                bt.logging.warning(f"[challenge-producer] executing GPU reset script: {script}")
                                os.execvpe(script, cmd, env)
                            else:
                                bt.logging.warning(f"[challenge-producer] GPU reset script not found, skipping: {script}")
                        except Exception as e:
                            bt.logging.error(f"[challenge-producer] GPU reset script execution failed: {e}")

                    if room_for_two:
                        # hstab
                        syn2, meta2, target2 = self._build_challenge_of_kind(uid, "hstab")
                        self.queue.put_nowait(QItem(uid, syn2, meta2, target2, file_path=None))
                        bt.logging.trace(
                            f"[challenge-producer] queued {meta2.circuit_kind} for UID {uid}"
                        )
                    else:
                        self._pending_uid = uid
                        self._pending_kind = "hstab"
            except Exception:
                bt.logging.error("[challenge-producer] error", exc_info=True)

            time.sleep(self._SLEEP_S)
        bt.logging.info("[challenge-producer] stop_event set; exiting loop")

    # Challenge creation
    def _build_challenge_for_uid(self, uid: int) -> Tuple[Any, ChallengeMeta, str]:
        return self._build_challenge_of_kind(uid, "peaked")

    def _build_challenge_of_kind(self, uid: int, kind: str) -> Tuple[Any, ChallengeMeta, str]:
        kind = (kind or "").lower()
        if kind not in self._strategies:
            raise ValueError(f"Unknown challenge kind: {kind!r}")
        strat = self._strategies[kind]

        difficulty_cfg = self._diff_cfg.get(kind)
        if difficulty_cfg is None:
            raise KeyError(f"Missing difficulty config for kind: {kind}")

        difficulty = difficulty_cfg.get(uid)
        if difficulty is None:
            difficulty = self._default_difficulty
        # sanitize (reject NaN/Inf/negatives)
        try:
            dv = float(difficulty)
        except Exception:
            dv = self._default_difficulty
        if not math.isfinite(dv) or dv < 0.0:
            dv = self._default_difficulty
        difficulty = dv

        cap = self._MAX_DIFFICULTY.get(kind)
        if cap is not None and difficulty > cap: # caps hstab at 50
            difficulty = cap

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

