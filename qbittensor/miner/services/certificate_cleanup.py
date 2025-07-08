from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import bittensor as bt


class CertificateCleanup:
    def __init__(
        self,
        cert_dir: Path,
        historical_dir: Path,
        archive_after_hours: int = 24,
        delete_after_days: int = 7,
        cleanup_interval_minutes: int = 5,
    ):
        self.cert_dir = cert_dir
        self.historical_dir = historical_dir
        self.archive_after_seconds = archive_after_hours * 3600
        self.delete_after_seconds = delete_after_days * 24 * 3600
        self.cleanup_interval_seconds = cleanup_interval_minutes * 60
        self._last_cleanup = 0.0
        self.historical_dir.mkdir(parents=True, exist_ok=True)

    def should_run_cleanup(self) -> bool:
        return time.time() - self._last_cleanup > self.cleanup_interval_seconds

    def run_cleanup_if_needed(self) -> None:
        if self.should_run_cleanup():
            self.run_full_cleanup()

    def run_full_cleanup(self) -> None:
        self._archive_old_certificates()
        self._delete_very_old_certificates()
        self._last_cleanup = time.time()

    def _archive_old_certificates(self, max_files_per_run: int = 1000) -> None:
        if not self.cert_dir.exists():
            return

        now = time.time()
        moved_count = 0
        checked_count = 0
        errors_count = 0

        try:
            for cert_file in self.cert_dir.glob("*.json"):
                if checked_count >= max_files_per_run:
                    break

                checked_count += 1

                try:
                    file_age = now - cert_file.stat().st_mtime
                    if file_age > self.archive_after_seconds:
                        historical_path = self.historical_dir / cert_file.name
                        cert_file.rename(historical_path)
                        moved_count += 1
                except (PermissionError, OSError) as e:
                    errors_count += 1
                    bt.logging.debug(
                        f"[cert-cleanup] Error moving {cert_file.name}: {e}"
                    )

            if moved_count > 0 or errors_count > 0:
                bt.logging.info(
                    f"[cert-cleanup] Archived {moved_count} certificates "
                    f"(checked={checked_count}, errors={errors_count})"
                )

        except Exception as e:
            bt.logging.warning(f"[cert-cleanup] Archive operation failed: {e}")

    def _delete_very_old_certificates(self) -> None:
        if not self.historical_dir.exists():
            return

        now = time.time()
        deleted_count = 0

        try:
            for cert_file in self.historical_dir.glob("*.json"):
                try:
                    file_age = now - cert_file.stat().st_mtime
                    if file_age > self.delete_after_seconds:
                        cert_file.unlink(missing_ok=True)
                        deleted_count += 1
                except (PermissionError, OSError) as e:
                    bt.logging.debug(
                        f"[cert-cleanup] Error deleting {cert_file.name}: {e}"
                    )

            if deleted_count > 0:
                bt.logging.info(
                    f"[cert-cleanup] Deleted {deleted_count} old historical certificates"
                )

        except Exception as e:
            bt.logging.warning(f"[cert-cleanup] Historical cleanup failed: {e}")
