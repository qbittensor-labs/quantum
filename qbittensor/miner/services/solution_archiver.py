from __future__ import annotations

import json, time
from pathlib import Path
import bittensor as bt


class SolutionArchiver:
    """
    Move solved circuit files whose challenge_id already appears in any
    certificate file.
    """

    def __init__(
        self,
        solved_dirs: list[Path],
        cert_dir: Path,
        archive_dir: Path,
        cleanup_interval_minutes: int = 60,
    ):
        self.solved_dirs = solved_dirs
        self.cert_dir = cert_dir
        self.archive_dir = archive_dir
        self.interval_s = cleanup_interval_minutes * 60
        self._last_cleanup = 0.0
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def should_run(self) -> bool:
        return time.time() - self._last_cleanup > self.interval_s

    def run_if_needed(self) -> None:
        if self.should_run():
            self.run()

    def run(self) -> None:
        moved = errs = 0
        cids_with_cert = self._collect_cert_cids()

        for solved_dir in self.solved_dirs:
            for fp in solved_dir.glob("*.json"):
                try:
                    cid = self._extract_cid(fp)
                    if cid in cids_with_cert:
                        fp.rename(self.archive_dir / fp.name)
                        moved += 1
                except Exception as e:
                    errs += 1
                    bt.logging.debug(f"[sol archive] Could not move {fp.name}: {e}")

        if moved or errs:
            bt.logging.info(
                f"[sol archive] Archived {moved} solved circuits " f"(errors={errs})"
            )
        self._last_cleanup = time.time()

    def _collect_cert_cids(self) -> set[str]:
        cids: set[str] = set()
        for fp in self.cert_dir.glob("*.json"):
            try:
                cids.add(json.loads(fp.read_text())["challenge_id"])
            except Exception:
                continue
        return cids

    @staticmethod
    def _extract_cid(fp: Path) -> str:
        # fast path: try to parse JSON
        try:
            return json.loads(fp.read_text())["challenge_id"]
        except Exception:
            # fall back to file‑name convention
            return fp.stem.split("_")[0]
