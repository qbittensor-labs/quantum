import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import bittensor as bt
from git import Repo

CLEANUP_FLAG = Path("/tmp/validator_cleanup_done")
_stop_event = threading.Event()


def restart_pm2_self(grace_period: int = 10) -> bool:
    """
    1) tell PM2 to SIGTERM us
    2) wait up to `grace_period` seconds for the cleanup flag
    3) call `pm2 restart`
    """
    try:
        mypid = os.getpid()
        raw = subprocess.check_output(["pm2", "jlist"], text=True)
        procs = json.loads(raw)

        for p in procs:
            if p.get("pid") == mypid:
                pm_id = p["pm_id"]

                bt.logging.info(f"[auto-updater] Sending SIGTERM via PM2 to id={pm_id}")
                subprocess.check_call(["pm2", "sendSignal", "SIGTERM", str(pm_id)])

                bt.logging.info(f"[auto-updater] Waiting up to {grace_period}s for cleanup flag")
                for _ in range(grace_period):
                    if CLEANUP_FLAG.exists():
                        bt.logging.info("[auto-updater] Detected cleanup flag, proceeding to restart")
                        try:
                            CLEANUP_FLAG.unlink()
                        except Exception:
                            pass
                        break
                    time.sleep(1)
                else:
                    bt.logging.warning("[auto-updater] Cleanup flag not seen, proceeding anyway")

                bt.logging.info(f"[auto-updater] Restarting validator via PM2 (id={pm_id})")
                subprocess.check_call(["pm2", "restart", str(pm_id), "--update-env"])
                return True

        bt.logging.info("[auto-updater] No PM2 entry for our PID; skipping PM2 restart")
    except Exception as e:
        bt.logging.error(f"[auto-updater] PM2 self-restart failed: {e}")
    return False


def auto_update(
    repo_path: str = ".",
    branch: str = "main",
    interval_seconds: int = 300,
    stop_event: threading.Event = _stop_event,
) -> None:
    """Daemon thread: poll the repo, pull on changes, then restart via PM2 or execv."""
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        origin = repo.remotes.origin
        bt.logging.info(f"[auto-updater] polling every {interval_seconds}s on '{branch}'")

        while not stop_event.is_set():
            try:
                current = repo.active_branch.name
                if current != branch:
                    bt.logging.debug(f"[auto-updater] on '{current}' not '{branch}'; skipping")
                else:
                    origin.fetch()
                    if repo.commit("HEAD") != repo.commit(f"origin/{branch}"):
                        before = repo.head.commit.hexsha[:7]
                        after = repo.commit(f"origin/{branch}").hexsha[:7]
                        bt.logging.info(f"[auto-updater] update {before} → {after}")

                        if repo.is_dirty(untracked_files=True):
                            bt.logging.info("[auto-updater] stashing local changes")
                            repo.git.stash("push", "--include-untracked", "-m", "auto-update")

                        origin.pull(branch)

                        bt.logging.info("[auto-updater] Installing package in editable mode")
                        ret = subprocess.call([sys.executable, "-m", "pip", "install", "-e", "."])
                        if ret != 0:
                            bt.logging.warning("[auto-updater] pip install -e . exit code %d", ret)
                        else:
                            bt.logging.info("[auto-updater] Package installed successfully")

                        if not restart_pm2_self(grace_period=10):
                            bt.logging.info("[auto-updater] Falling back to execv restart")
                            os.execv(sys.executable, [sys.executable] + sys.argv)

                        return

                for _ in range(int(interval_seconds)):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                bt.logging.error(f"[auto-updater] error in loop: {e}")
                time.sleep(min(interval_seconds, 60))

    except Exception as e:
        bt.logging.error(f"[auto-updater] initialization failed: {e}")
    finally:
        bt.logging.info("[auto-updater] updater thread exiting")


def start_updater(check_interval_minutes: int = 5) -> None:
    interval = check_interval_minutes * 60
    updater = threading.Thread(
        target=auto_update,
        args=(".", "main", interval, _stop_event),
        daemon=True,
        name="auto-updater",
    )
    updater.start()
    bt.logging.info(f"[auto-updater] thread started (interval: {check_interval_minutes}m)")


def stop_updater() -> None:
    _stop_event.set()
    bt.logging.info("[auto-updater] stop signal sent")
