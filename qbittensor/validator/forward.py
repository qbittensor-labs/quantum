"""
Forward method for a Validator in Subnet 63 - Quantum
"""
from __future__ import annotations

import concurrent.futures as futures
import queue, threading, time, os, asyncio
from pathlib import Path
from typing import Any, List

import bittensor as bt
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.reward import ScoringManager
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.services.solution_processor import SolutionProcessor
from qbittensor.validator.services.weight_manager import WeightManager
from qbittensor.validator.utils.whitelist import load_whitelist
from qbittensor.validator.database.fixups import apply_fixups 

# AUTO‑SCALE: detect hardware once at import time
try:
    import torch
    _GPU_COUNT = torch.cuda.device_count()
except Exception:
    _GPU_COUNT = 0

_CPU_COUNT = max(1, os.cpu_count() or 1)

_PEAKED_WORKERS = max(1, min(32, (_GPU_COUNT or 1) * 2)) # GPU‑bound
_HSTAB_WORKERS  = max(4, min(32, _CPU_COUNT * 2)) # CPU‑bound

_PEAKED_QSIZE = _PEAKED_WORKERS * 4
_HSTAB_QSIZE  = _HSTAB_WORKERS * 4

RPC_DEADLINE = 10
CFG_DIR = Path(__file__).resolve().parent / "config"
CFG_DIR.mkdir(exist_ok=True)
_REFRESH_SECONDS = 300
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
def _bootstrap(v: "Validator") -> None:
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

    db_dir = Path(__file__).parent / "database"; db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "validator_data.db"

    apply_fixups(db_path)

    v._diff_cfg = {
        "peaked": DifficultyConfig(
            CFG_DIR / "difficulty_peaked.json", uid_list, 0.0, db_path=db_path,
            hotkey_lookup=lambda u: v.metagraph.hotkeys[u], clamp=True
        ),
        "hstab": DifficultyConfig(
            CFG_DIR / "difficulty_hstab.json", uid_list, 26.0, db_path=None, # Skips max lookup
            hotkey_lookup=lambda u: v.metagraph.hotkeys[u], clamp=False
        ),
    }

    v.certificate_issuer = CertificateIssuer(wallet=v.wallet)
    v._sol_proc  = SolutionProcessor(cert_issuer=v.certificate_issuer)
    v._scoring_mgr = ScoringManager(str(db_path))
    v._weight_mgr  = WeightManager(v)

    from qbittensor.validator.services.challenge_producer import ChallengeProducer
    v._producers = {
        "peaked": ChallengeProducer(
            wallet=v.wallet, diff_cfg={"peaked": v._diff_cfg["peaked"]},
            uid_list=uid_list, validator=v, queue_size=_PEAKED_QSIZE
        ),
        "hstab": ChallengeProducer(
            wallet=v.wallet, diff_cfg={"hstab": v._diff_cfg["hstab"]},
            uid_list=uid_list, validator=v, queue_size=_HSTAB_QSIZE
        ),
    }
    v._producers["peaked"]._strategies["hstab"].weight = 0.0
    v._producers["hstab"]._strategies["peaked"].weight = 0.0
    for p in v._producers.values():
        p._weights_cache = [s.weight for s in p._strategies.values()]
        p.start()

    from qbittensor.validator.services.response_processor import ResponseProcessor
    v._resp_proc = ResponseProcessor(v)

    v._queues = {k: p.queue for k, p in v._producers.items()}
    v._executors = {
        "peaked": futures.ThreadPoolExecutor(max_workers=_PEAKED_WORKERS, thread_name_prefix="peaked"),
        "hstab": futures.ThreadPoolExecutor(max_workers=_HSTAB_WORKERS,  thread_name_prefix="hstab"),
    }

    if not hasattr(v, "_dispatcher_thread"):
        v._dispatcher_thread = threading.Thread(target=_dispatcher_loop, args=(v,), name="Dispatcher", daemon=True)
        v._dispatcher_thread.start()

    bt.logging.info(f"Validator bootstrap complete - workers: peaked={_PEAKED_WORKERS}, hstab={_HSTAB_WORKERS}")


def _dispatcher_loop(v: "Validator") -> None:
    while True:
        for kind, q in v._queues.items():
            try:
                item = q.get_nowait()
            except queue.Empty:
                continue
            v._executors[kind].submit(_handle_item, v, item)
        time.sleep(_DISPATCH_SLEEP)


_dendrite_lock = threading.Lock()

def _handle_item(v: "Validator", item) -> None:
    _ensure_event_loop()
    uid, *_ = item
    with _inflight_lock:
        if uid in v._in_flight:
            return
        v._in_flight.add(uid)
    try:
        miner_hotkey = v.metagraph.hotkeys[uid]
        with _dendrite_lock:
            v._resp_proc.process(item, miner_hotkey=miner_hotkey)
    finally:
        with _inflight_lock:
            v._in_flight.discard(uid)
        v._cursor.save(uid)


def _refresh_uid_deps(v: "Validator") -> None:
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
        for p in v._producers.values():
            p.update_uid_list(final_uids)
        bt.logging.info(f"metagraph changed: {len(final_uids)} live miners")


def forward(self: "Validator") -> None:
    """Entry point"""
    if not getattr(self, "_bootstrapped", False):
        _bootstrap(self)

    now = time.time()
    if now - getattr(self, "_last_uid_refresh", 0) >= _REFRESH_SECONDS:
        _refresh_uid_deps(self)
        self._last_uid_refresh = now

    # weight push
    self._weight_mgr.update()


__all__ = ["forward"]
