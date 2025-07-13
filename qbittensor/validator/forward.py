"""
Forward method for a Validator Subnet 63 - Quantum
"""
from __future__ import annotations

import queue
import datetime as dt
from pathlib import Path
from typing import Any, List

# one time validator migration
from .validator_migration import add_difficulty_to_challenges

import bittensor as bt
from qbittensor.validator.utils.challenge_utils import build_challenge
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.services.solution_processor import SolutionProcessor
from qbittensor.validator.services.solution_extractor import SolutionExtractor
from qbittensor.validator.services.weight_manager import WeightManager
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.reward import ScoringManager
from qbittensor.validator.utils.whitelist import load_whitelist
from qbittensor.validator.utils.challenge_logger import (
    log_challenge,
    log_certificate_as_solution,
)
from qbittensor.common.certificate import Certificate

# CONSTANTS
RPC_DEADLINE = 10  # shorter now
CFG_DIR      = Path(__file__).resolve().parent / "config"
CFG_DIR.mkdir(exist_ok=True)
CFG_PATH     = CFG_DIR / "difficulty.json"
CURSOR_PATH  = Path(__file__).resolve().parent / "last_uid.txt"

def _bootstrap(v: "Validator") -> None:
    if getattr(v, "_bootstrapped", False):
        return
    v._bootstrapped = True

    v._in_flight: set[int] = set()

    # UID list
    raw = v.metagraph.uids.tolist()
    uid_list = [
        u for u in raw
        if v.metagraph.axons[u].ip not in ("0.0.0.0", "", None)
        and v.metagraph.axons[u].port != 0
    ]

    from qbittensor.validator.services.cursor_store import CursorStore
    v._cursor  = CursorStore(Path(__file__).parent / "last_uid.txt")
    start_uid  = v._cursor.load()
    if start_uid in uid_list:
        uid_list = uid_list[uid_list.index(start_uid):] + uid_list[:uid_list.index(start_uid)]

    v._whitelist = load_whitelist(CFG_DIR / "whitelist_validators.json")

    # Helpers
    v.certificate_issuer = CertificateIssuer(wallet=v.wallet)
    dbdir = Path(__file__).parent / "database"; dbdir.mkdir(exist_ok=True)
    db    = dbdir / "validator_data.db"

    # Difficulty config
    v._diff_cfg = DifficultyConfig(
        CFG_PATH,
        uids = uid_list,
        db_path = db,
        hotkey_lookup = lambda u: v.metagraph.hotkeys[u],
    )    

    # one time vali migration
    add_difficulty_to_challenges(str(db))

    v._sol_proc    = SolutionProcessor(cert_issuer=v.certificate_issuer)
    v._scoring_mgr = ScoringManager(str(db))
    v._weight_mgr  = WeightManager(v)

    # ChallengeProducer
    from qbittensor.validator.services.challenge_producer import ChallengeProducer
    v._producer = ChallengeProducer(
        wallet      = v.wallet,
        diff_cfg    = v._diff_cfg,
        uid_list    = uid_list,
        validator   = v,
    )
    v._producer.start()

    # ResponseProcessor
    from qbittensor.validator.services.response_processor import ResponseProcessor
    v._resp_proc = ResponseProcessor(v)

    bt.logging.info("✅ validator bootstrap complete")


# Forward method
def forward(self: "Validator") -> None:
    """
    Called by bittensor runtime once per block.
    """
    if not getattr(self, "_bootstrapped", False):
        _bootstrap(self)

    try:
        item = self._producer.queue.get_nowait() # non-blocking
    except queue.Empty:
        return

    uid = item.uid
    miner_hotkey = self.metagraph.hotkeys[uid]
    if uid in self._in_flight: # re-entrancy guard
        return
    self._in_flight.add(uid)
    try:
        self._resp_proc.process(item, miner_hotkey = miner_hotkey)
    finally:
        self._in_flight.discard(uid)
        self._weight_mgr.update() # cheap
        self._cursor.save(uid)


__all__ = ["forward"]
