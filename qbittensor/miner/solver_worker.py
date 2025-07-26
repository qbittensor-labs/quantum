from __future__ import annotations

"""
Worker for the miner
"""

import asyncio
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Tuple

import bittensor as bt

from qbittensor.protocol import ChallengeCircuits

from .config import DEFAULT_QUEUE_SIZE, DEFAULT_SCAN_INTERVAL, Paths
from .extract import cid_from_filename, qasm_from_file, qasm_from_synapse
from .storage import Storage

__all__ = ["SolverWorker"]


class SolverWorker:
    """Pulls work from two sources → solves → hands results to *Storage*."""

    def __init__(
        self,
        base_dir: Path,
        *,
        solver_fn: Callable[[str, str], str],
        queue_size: int = DEFAULT_QUEUE_SIZE,
        scan_interval: float = DEFAULT_SCAN_INTERVAL,
        thread_name: str = "CircuitSolver",
    ) -> None:
        self.paths = Paths.from_base(base_dir)
        self.storage = Storage(self.paths)
        self._solve = solver_fn
        self._scan_interval = scan_interval
        self._queue: "queue.Queue[Tuple[str, str, str, str]]" = queue.Queue(maxsize=queue_size)
        self._thread_name = thread_name
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._start_thread()

    def submit_synapse(self, syn: ChallengeCircuits) -> None:
        """Extract QASM, circuit type & CID then push to queue (idempotent)."""
        cid = syn.challenge_id
        if self.storage.is_solved(cid):
            return
        if qasm := qasm_from_synapse(syn):
            validator_hotkey = getattr(syn, "validator_hotkey", None) or ""
            circuit_type = getattr(syn, "circuit_kind", "peaked")
            self._enqueue(cid, qasm, circuit_type, validator_hotkey)

    def submit_file(self, fp: Path) -> None:
        cid = cid_from_filename(fp)
        if self.storage.is_solved(cid):
            return
        if qasm := qasm_from_file(fp):
            self._enqueue(cid, qasm, "peaked", "")

    def drain_solutions(self, *, n: int = 10, validator_hotkey: str = None):
        return self.storage.drain_unsent(max_count=n, validator_hotkey=validator_hotkey)

    def record_reward(self, cid: str, certificate: dict) -> None:
        self.storage.save_certificate(cid, certificate)

    # Internal helpers
    # Queue mgmt

    def _enqueue(self, cid: str, qasm: str, circuit_type: str, validator_hotkey: str = ""):
        try:
            self._queue.put_nowait((cid, qasm, circuit_type, validator_hotkey))
            bt.logging.debug(
                f" queued {circuit_type} circuit {cid[:10]} from validator {validator_hotkey[:10] if validator_hotkey else 'unknown'}"
            )
        except queue.Full:
            bt.logging.warning(" solver queue full. dropping challenge")

    # Thread bootstrap

    def _start_thread(self):
        def runner():
            self._running = True
            try:
                asyncio.run(self._main_loop())
            finally:
                self._running = False

        threading.Thread(target=runner, name=self._thread_name, daemon=True).start()
        bt.logging.info(f" Solver thread '{self._thread_name}' started")

    # Main loop

    async def _main_loop(self):
        self._scan_unsolved_dir()  # once on start‑up
        while True:
            try:
                # Use thread-safe queue with timeout
                cid, qasm, circuit_type, validator_hotkey = self._queue.get(timeout=self._scan_interval)
            except queue.Empty:
                self._scan_unsolved_dir()
                continue

            bt.logging.debug(
                f" solving {circuit_type} circuit {cid[:10]} from validator {validator_hotkey[:10] if validator_hotkey else 'unknown'}"
            )
            bits = await asyncio.to_thread(self._solve, qasm, circuit_type)
            bt.logging.debug(f" {cid[:10]} → {bits}")
            self.storage.save_solution(cid, bits, validator_hotkey or None)
            self._queue.task_done()

    # File‑system polling

    def _scan_unsolved_dir(self):
        for fp in self.paths.unsolved.glob("*"):
            self.submit_file(fp)
