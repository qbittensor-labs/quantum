# coding: utf-8
"""
ChallengeProducer. background task that keeps an asyncio.Queue
filled with fresh `ChallengeCircuits` objects with difficulty awareness.
"""

from __future__ import annotations

import asyncio
import importlib.resources as pkg_res
import json
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Any, Tuple, Iterator, List
import itertools

import bittensor as bt
from qbittensor.protocol import ChallengeCircuits
from qbittensor.validator.utils.challenge_utils import build_challenge
from qbittensor.validator.utils.validator_meta import ChallengeMeta
from qbittensor.validator.config.difficulty_config import DifficultyConfig

PKG_ROOT = Path(pkg_res.files("qbittensor.validator"))
DEFAULT_DIR = PKG_ROOT / "pending_challenges"
DEFAULT_DIR.mkdir(exist_ok=True)


class QItem(NamedTuple):
    uid: int
    syn: ChallengeCircuits
    meta: ChallengeMeta
    target: str
    file_path: Path | None


class ChallengeProducer:
    """Run `start()` once; consume items from `.queue`."""

    _SLEEP = 0.1

    def __init__(
        self,
        wallet,
        *,
        directory: Path = DEFAULT_DIR,
        queue_size: int = 64,
        difficulty: float = 0.0,
        diff_cfg: DifficultyConfig = None,
        batch_size: int = 32,
        uid_list: List[int] = None,
        validator=None,
    ):
        self._wallet = wallet
        self._directory = directory
        self._difficulty = difficulty
        self._diff_cfg = diff_cfg
        self._stash: dict[int, list[tuple[Any, ...]]] = {}
        self._BATCH_SIZE = batch_size
        self._uid_cycle: Iterator[int] = itertools.cycle(uid_list or [])

        self.queue: asyncio.Queue[QItem] = asyncio.Queue(maxsize=queue_size)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._validator = validator

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        # Clear stash when starting to ensure we use current batch size
        self.cleanup_stash()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._stop.set()
        if self._diff_cfg:
            self.cleanup_stash()

    async def _loop(self) -> None:
        bt.logging.info("[challenge-producer] started")
        last_cleanup = asyncio.get_event_loop().time()
        cleanup_interval = 300

        while not self._stop.is_set():
            try:
                current_time = asyncio.get_event_loop().time()
                if current_time - last_cleanup > cleanup_interval:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._cleanup_old_files)
                    last_cleanup = current_time

                if self._diff_cfg:
                    # pull the *next* miner UID
                    uid = next(self._uid_cycle)
                    if uid in self._validator._in_flight:  # ➍ back-pressure
                        await asyncio.sleep(self._SLEEP)
                        continue
                    # build one circuit at their configured difficulty
                    syn, meta, target, fp = await self._make_payload(uid)
                    # stash the uid on the QItem so forward() knows who it's for
                    await self.queue.put(QItem(uid, syn, meta, target, fp))
                else:
                    syn, meta, target = build_challenge(
                        wallet=self._wallet, difficulty=self._difficulty
                    )
                    fp = self._write_to_disk(meta, syn, target)
                    await self.queue.put(QItem(syn, meta, target, fp))

                bt.logging.debug(f"[challenge-producer] queued challenge")
            except Exception as e:
                bt.logging.error("[challenge-producer] error", exc_info=True)
            await asyncio.sleep(self._SLEEP)

    async def _make_payload(self, uid: int) -> Tuple[Any, ...]:
        """Build one circuit, patch difficulty, return the usual 4-tuple."""
        stash = self._stash.get(uid)
        if stash:
            return stash.pop(0)

        difficulty = (
            float(self._diff_cfg.get(uid)) if self._diff_cfg else self._difficulty
        )
        bt.logging.info(f"uid: {uid}, diff: {difficulty}")

        # Create just ONE circuit, not BATCH_SIZE
        syn, meta, target = build_challenge(wallet=self._wallet, difficulty=difficulty)
        syn.difficulty_level = difficulty
        fp = self._write_to_disk(meta, syn, target)
        return (syn, meta, target, fp)

    def cleanup_stash(self) -> None:
        """Clean up any remaining stashed challenge files."""
        for uid, items in self._stash.items():
            for syn, meta, target_state, fp in items:
                self._cleanup_file(fp)
            bt.logging.info(
                f"[cleanup] Removed {len(items)} stashed challenges for UID {uid}"
            )
        self._stash.clear()

    def _cleanup_old_files(self, max_age_seconds: int = 86400) -> None:
        """Remove challenge files older than max_age_seconds (default 24 hours)."""
        try:
            import time

            if not self._directory.exists():
                return

            now = time.time()
            removed_count = 0
            checked_count = 0
            errors_count = 0
            max_files_per_run = 1000

            for fp in self._directory.glob("*.json"):
                checked_count += 1
                if checked_count > max_files_per_run:
                    bt.logging.debug(
                        f"[challenge-producer] Cleanup batch limit reached ({max_files_per_run} files)"
                    )
                    break

                try:
                    file_age = now - fp.stat().st_mtime
                    if file_age > max_age_seconds:
                        fp.unlink(missing_ok=True)
                        removed_count += 1
                except PermissionError:
                    errors_count += 1
                    bt.logging.debug(
                        f"[challenge-producer] Permission denied for {fp.name}"
                    )
                except Exception:
                    errors_count += 1
                    pass

            if removed_count > 0 or errors_count > 0:
                bt.logging.info(
                    f"[challenge-producer] Cleanup: checked={checked_count}, "
                    f"removed={removed_count}, errors={errors_count}"
                )

        except Exception as e:
            bt.logging.warning(f"[challenge-producer] Error during cleanup: {e}")

    def _write_to_disk(
        self, meta: ChallengeMeta, syn: ChallengeCircuits, target: str
    ) -> Path:
        fp = self._directory / f"{meta.challenge_id}.json"
        with fp.open("w") as f:
            json.dump(
                {
                    "circuit_data": syn.circuit_data,
                    "meta": asdict(meta),
                    "target": target,
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
