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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import bittensor as bt


from .config import DEFAULT_QUEUE_SIZE, DEFAULT_SCAN_INTERVAL, Paths
from .extract import cid_from_filename, qasm_from_file, qasm_from_synapse
from .storage import Storage
from qbittensor.protocol import _CircuitSynapseBase

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
        self._queue: "asyncio.Queue[Tuple[str, str, str, str]]" = asyncio.Queue(
            maxsize=queue_size
        )  # (cid, qasm, circuit_type, validator_hotkey)
        self._thread_name = thread_name
        self._running = False
        self.executor = ProcessPoolExecutor(mp_context=mp.get_context("spawn"))

    def __del__(self):
        self.executor.shutdown(wait=True)

    def start(self) -> None:
        if self._running:
            return
        self._start_thread()

    def submit_synapse(self, syn: _CircuitSynapseBase) -> None:
        """Extract QASM, circuit type & CID then push to queue (idempotent)."""
        cid = syn.challenge_id
        if self.storage.is_solved(cid):
            return
        qasm = qasm_from_synapse(syn)
        # Handle None/empty QASM as special case for difficulty queries
        is_empty_qasm = syn.circuit_data is None or syn.circuit_data == "" or qasm is None or qasm.strip() == ""

        if qasm or is_empty_qasm:
            validator_hotkey = getattr(syn, "validator_hotkey", None) or ""
            circuit_type = getattr(syn, "circuit_kind", "peaked")
            # For empty QASM, use a dummy QASM string for processing
            qasm_to_process = qasm if qasm else ""
            self._enqueue(cid, qasm_to_process, circuit_type, validator_hotkey)

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
            cid_short = cid[:10] if cid and isinstance(cid, str) else str(cid or 'unknown')[:10]
            validator_short = validator_hotkey[:10] if validator_hotkey and isinstance(validator_hotkey, str) else 'unknown'
            bt.logging.debug(
                f" queued {circuit_type} circuit {cid_short} from validator {validator_short}"
            )
        except asyncio.QueueFull:
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
                cid, qasm, circuit_type, validator_hotkey = await asyncio.wait_for(
                    self._queue.get(), timeout=self._scan_interval
                )
            except asyncio.TimeoutError:
                self._scan_unsolved_dir()
                continue

            cid_display = cid[:10] if cid and isinstance(cid, str) else str(cid or 'unknown')[:10]
            validator_display = validator_hotkey[:10] if validator_hotkey and isinstance(validator_hotkey, str) else 'unknown'

            # Handle None/empty QASM (difficulty queries) gracefully
            if qasm is None or not qasm or qasm.strip() == "":
                bt.logging.debug(
                    f" skipping empty/None QASM circuit {cid_display} (difficulty query)"
                )
                # Log empty qasm
                bits = ""
                bt.logging.debug(f" {cid_display} → {bits} (None/empty QASM for difficulty query)")
            else:
                bt.logging.debug(
                    f" solving {circuit_type} circuit {cid_display} from validator {validator_display}"
                )
                loop = asyncio.get_running_loop()
                bits = await loop.run_in_executor(self.executor, self._solve, qasm, circuit_type)
                bt.logging.debug(f" {cid_display} → {bits}")
                self.storage.save_solution(cid, bits, circuit_type, validator_hotkey or None)
            self._queue.task_done()

    # File‑system polling

    def _scan_unsolved_dir(self):
        for fp in self.paths.unsolved.glob("*"):
            self.submit_file(fp)
