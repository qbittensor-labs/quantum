"""
Forward method for a Validator in Subnet 63 - Quantum
"""
from __future__ import annotations

import concurrent.futures as futures
import asyncio
import os
import queue
import threading
import time
from pathlib import Path
from typing import Protocol

import bittensor as bt
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.reward import ScoringManager
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.services.solution_processor import SolutionProcessor
from qbittensor.validator.services.weight_manager import WeightManager
from qbittensor.validator.services.certificate_manager import CertificateManager
from qbittensor.validator.utils.whitelist import load_whitelist
from qbittensor.validator.database.fixups import apply_fixups 

try:
    import torch
    _GPU_COUNT = torch.cuda.device_count()
except Exception:
    _GPU_COUNT = 0

_CPU_COUNT = max(1, os.cpu_count() or 1)

_DISPATCH_WORKERS = max(4, min(32, _CPU_COUNT * 2))
_QUEUE_SIZE = 128

CFG_DIR = Path(__file__).resolve().parent / "config"
CFG_DIR.mkdir(exist_ok=True)
_REFRESH_SECONDS = 450
_DISPATCH_SLEEP  = 0.01
_inflight_lock   = threading.Lock()

# guarantee an asyncio loop in each worker thread
def _ensure_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# bootstrap & runtime
class _ValidatorLike(Protocol):
    metagraph: any
    wallet: any
    is_running: bool


def _bootstrap(v: _ValidatorLike) -> None:
    if getattr(v, "_bootstrapped", False):
        return
    v._bootstrapped = True

    # Create config files if first startup
    import shutil
    template = CFG_DIR / "difficulty_hstab.sample.json"
    runtime  = CFG_DIR / "difficulty_hstab.json"

    if not runtime.exists() and template.exists():
        shutil.copy2(template, runtime)
        bt.logging.info(f"[bootstrap] seeded {runtime.name} from template")

    v._in_flight   = set()
    if not hasattr(v, "_shutdown_event"):
        v._shutdown_event = threading.Event()
    v._hotkey_cache: dict[int, str] = {}

    raw = v.metagraph.uids.tolist()
    uid_list = [u for u in raw if v.metagraph.axons[u].is_serving]

    from qbittensor.validator.services.cursor_store import CursorStore
    v._cursor = CursorStore(Path(__file__).parent / "last_uid.txt")

    start_uid = v._cursor.load()
    if start_uid in uid_list:
        idx = uid_list.index(start_uid)
        uid_list = uid_list[idx:] + uid_list[:idx]

    v._whitelist = load_whitelist(CFG_DIR / "whitelist_validators.json")

    db_dir = Path(__file__).parent / "database"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "validator_data.db"

    apply_fixups(db_path)

    v._diff_cfg = {

        "peaked": DifficultyConfig(
            CFG_DIR / "difficulty_peaked.json", uid_list, 0.0, db_path=db_path,
            hotkey_lookup=lambda u: v.metagraph.hotkeys[u], clamp=False
        ),

        "hstab": DifficultyConfig(
            CFG_DIR / "difficulty_hstab.json", uid_list, 26.0, db_path=None,  # Skips max lookup
            hotkey_lookup=lambda u: v.metagraph.hotkeys[u], clamp=False
        ),
    }

    v.certificate_issuer = CertificateIssuer(wallet=v.wallet)
    v._cert_mgr = CertificateManager(v.certificate_issuer)
    v._sol_proc  = SolutionProcessor(cert_issuer=v.certificate_issuer)
    v._scoring_mgr = ScoringManager(str(db_path))
    v._weight_mgr  = WeightManager(v)

    from qbittensor.validator.services.challenge_producer import ChallengeProducer
    v._producer = ChallengeProducer(
        wallet=v.wallet,
        diff_cfg={
            "peaked": v._diff_cfg["peaked"],
            "hstab": v._diff_cfg["hstab"],
        },
        uid_list=uid_list,
        validator=v,
        queue_size=_QUEUE_SIZE,
    )
    v._producer._weights_cache = [s.weight for s in v._producer._strategies.values()]
    v._producer.start()

    from qbittensor.validator.services.response_processor import ResponseProcessor
    v._resp_proc = ResponseProcessor(v)

    v._queue = v._producer.queue

    if not hasattr(v, "_dispatcher_thread"):
        v._dispatcher_thread = threading.Thread(target=_dispatcher_loop, args=(v,), name="Dispatcher", daemon=True)
        v._dispatcher_thread.start()

def _dispatcher_loop(v: _ValidatorLike) -> None:
    while not getattr(v, "_shutdown_event", threading.Event()).is_set():
        try:
            item = v._queue.get_nowait()
        except queue.Empty:
            time.sleep(_DISPATCH_SLEEP)
            continue
        try:
            uid, *_ = item
        except Exception:
            time.sleep(_DISPATCH_SLEEP)
            continue
        with _inflight_lock:
            if uid in v._in_flight:
                v._queue.put(item)
                time.sleep(_DISPATCH_SLEEP)
                continue
        try:
            _handle_item(v, item)
        except Exception as e:
            bt.logging.error(f"[dispatcher] unhandled error: {e}", exc_info=True)
    bt.logging.info("[dispatcher] shutdown signal received; exiting dispatcher loop")

_dendrite_lock = threading.Lock()

def _handle_item(v: _ValidatorLike, item) -> None:
    _ensure_event_loop()
    uid, *_ = item
    if getattr(v, "_shutdown_event", threading.Event()).is_set():
        return
    with _inflight_lock:
        if uid in v._in_flight:
            return
        v._in_flight.add(uid)
    try:
        miner_hotkey = v.metagraph.hotkeys[uid]
        with _dendrite_lock:
            if not getattr(v, "_shutdown_event", threading.Event()).is_set():
                v._resp_proc.process(item, miner_hotkey=miner_hotkey)
    finally:
        with _inflight_lock:
            v._in_flight.discard(uid)
        v._cursor.save(uid)


def _refresh_uid_deps(v: _ValidatorLike) -> None:
    bt.logging.info("refreshing metagraph…")
    v.metagraph.sync()

    live = {u for u in v.metagraph.uids if v.metagraph.axons[u].is_serving}
    for uid, hk in enumerate(v.metagraph.hotkeys):
        if uid not in live:
            continue
        prev = v._hotkey_cache.get(uid)
        if prev and prev != hk:
            bt.logging.info(f"UID {uid} reassigned {prev} - {hk}; resetting difficulty")
            for cfg in v._diff_cfg.values():
               cfg.set(uid, 0.0)
    v._hotkey_cache = {uid: hk for uid, hk in enumerate(v.metagraph.hotkeys)}

    final_uids = sorted(live)
    if final_uids != getattr(v, "_uid_cache", None):
        v._uid_cache = final_uids
        for cfg in v._diff_cfg.values():
            cfg.update_uid_list(final_uids)
        v._producer.update_uid_list(final_uids)
        bt.logging.info(f"metagraph changed: {len(final_uids)} live miners")


def forward(self: _ValidatorLike) -> None:
    """Entry point"""
    if not getattr(self, "_bootstrapped", False):
        _bootstrap(self)

    if time.time() - getattr(self, "_last_uid_refresh", 0) >= _REFRESH_SECONDS:
        _refresh_uid_deps(self)
        self._last_uid_refresh = time.time()

    # weight push
    self._weight_mgr.update()
    self._cert_mgr.update()


def shutdown(v: _ValidatorLike, timeout_s: float | None = None) -> None:
    if timeout_s is None:
        try:
            timeout_s = float(os.getenv("VALIDATOR_SHUTDOWN_TIMEOUT_S", "300"))
        except Exception:
            timeout_s = 300.0
    try:
        bt.logging.info("[shutdown] initiating validator shutdown…")
    except Exception:
        pass

    try:
        if hasattr(v, "_shutdown_event"):
            v._shutdown_event.set()
    except Exception:
        pass

    try:
        if hasattr(v, "_producer") and v._producer:
            v._producer.stop()
            bt.logging.info("[shutdown] challenge producer stopped")
    except Exception:
        bt.logging.warning("[shutdown] error stopping producer", exc_info=True)

    try:
        th = getattr(v, "_dispatcher_thread", None)
        if th and th.is_alive():
            th.join(timeout=5.0)
    except Exception:
        pass

    t0 = time.time()
    try:
        while True:
            with _inflight_lock:
                remaining = len(getattr(v, "_in_flight", set()))
            if remaining == 0:
                break
            elapsed = time.time() - t0
            if elapsed > timeout_s:
                bt.logging.error(f"[shutdown] hard timeout after {elapsed:.1f}s with {remaining} in‑flight; forcing exit")
                import os as _os
                _os._exit(1)
            time.sleep(0.25)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

    try:
        bt.logging.info("[shutdown] validator shutdown complete")
    except Exception:
        pass


__all__ = ["forward", "shutdown"]
