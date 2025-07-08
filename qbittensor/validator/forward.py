"""
Validator implementation for Subnet63 - Quantum.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import json
from pathlib import Path
from typing import Any, Tuple

import bittensor as bt

from qbittensor.protocol import ChallengeCircuits
from qbittensor.validator.config.difficulty_config import DifficultyConfig
from qbittensor.validator.services.challenge_producer import ChallengeProducer, QItem
from qbittensor.validator.services.solution_processor import SolutionProcessor
from qbittensor.validator.services.solution_extractor import SolutionExtractor
from qbittensor.validator.services.weight_manager import WeightManager
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.reward import ScoringManager
from qbittensor.validator.utils.challenge_logger import (
    log_challenge,
    log_certificate_as_solution,
)
from qbittensor.common.certificate import Certificate

# Constants
RPC_DEADLINE = 20
HARD_TIMEOUT = RPC_DEADLINE + 10
BATCH_SIZE = 8

CFG_DIR = Path(__file__).resolve().parent / "config"
CFG_DIR.mkdir(exist_ok=True)
CFG_PATH = CFG_DIR / "difficulty.json"


async def _bootstrap(v: "Validator") -> None:
    """Run once, lazily, the first time forward() is called."""
    v._in_flight: set[int] = set()

    # metagraph helpers
    uid_list = v.metagraph.uids.tolist()
    v._diff_cfg = DifficultyConfig(CFG_PATH, uids=uid_list)
    v._uid_list = uid_list
    v._uid_cycle = itertools.cycle(uid_list)

    # challenge machinery
    v.certificate_issuer = CertificateIssuer(wallet=v.wallet)
    v.challenge_producer = ChallengeProducer(
        wallet=v.wallet,
        diff_cfg=v._diff_cfg,
        batch_size=BATCH_SIZE,
        queue_size=BATCH_SIZE,  # keeps on same pace as circuit gen
        uid_list=uid_list,
        validator=v,
    )
    v.challenge_producer.start()

    # load trusted‑validator whitelist
    whitelist_path = CFG_DIR / "whitelist_validators.json"
    try:
        with whitelist_path.open() as fh:
            data = json.load(fh)
            # accept either the old plain-list format or the new wrapped format
            # TODO: pick one when safe
            if isinstance(data, dict) and "whitelist" in data:
                data = data["whitelist"]
            v._whitelist = set(data)
            bt.logging.trace(f"Validator whitelist: {v._whitelist}")
    except FileNotFoundError:
        bt.logging.warning(
            f"[bootstrap] no whitelist found at {whitelist_path}; accepting none"
        )
        v._whitelist = set()

    # wait until the producer has at least one full batch ready
    while v.challenge_producer.queue.qsize() < BATCH_SIZE:
        await asyncio.sleep(0.02)

    # database & scoring
    db_dir = Path(__file__).resolve().parent / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "validator_data.db"

    v._sol_proc = SolutionProcessor(cert_issuer=v.certificate_issuer)
    v._scoring_mgr = ScoringManager(str(db_path))
    v._weight_mgr = WeightManager(v)

    bt.logging.info("✅ Validator bootstrap complete.")


async def forward(self: "Validator") -> None:
    """Main entry‑point called by the bittensor runtime each block."""
    if not hasattr(self, "challenge_producer"):
        await _bootstrap(self)

    # dequeue a full batch of challenges
    items: list[QItem] = []
    uids: list[int] = []

    for _ in range(BATCH_SIZE):
        qitem = await self.challenge_producer.queue.get()
        items.append(qitem)
        uids.append(qitem.uid)
        self.challenge_producer.queue.task_done()
    for uid in uids:
        self._in_flight.add(uid)

    try:
        await _handle_batch_miners(
            self,
            uids,
            [(i.syn, i.meta, i.target, i.file_path) for i in items],
        )
    finally:
        for uid in uids:
            self._in_flight.discard(uid)
        # refresh stake‑weights every block
    await self._weight_mgr.update()


async def _handle_batch_miners(
    v: "Validator",
    uids: list[int],
    items: list[Tuple[ChallengeCircuits, Any, Any, Path | None]],
) -> None:
    """Send a batch, collect replies, process solutions & certificates."""
    axons, syns, metas, fps, t_sents = [], [], [], [], []
    failed_fps: list[Path | None] = []

    # build synapses
    for uid, (syn, meta, target_state, fp) in zip(uids, items):
        t_sent = dt.datetime.utcnow()
        t_sents.append(t_sent)

        # log challenge before sending
        try:
            log_challenge(
                challenge_id=meta.challenge_id,
                validator_hotkey=meta.validator_hotkey,
                miner_uid=uid,
                entanglement_entropy=meta.entanglement_entropy,
                nqubits=meta.nqubits,
                rqc_depth=meta.rqc_depth,
                solution=target_state,
                time_sent=t_sent,
            )
        except Exception:
            bt.logging.error("[batch] log_challenge failed", exc_info=True)

        # skip miners without axons
        try:
            axon = v.metagraph.axons[uid]
        except IndexError:
            bt.logging.warning(f"[batch] UID {uid} has no axon")
            failed_fps.append(fp)
            continue

        syn.solution_bitstring = None  # don’t send the answer back
        # attach any queued certificates destined for this miner
        certs = v.certificate_issuer.pop_for(uid)
        syn.attach_certificates(certs)

        axons.append(axon)
        syns.append(syn)
        metas.append(meta)
        fps.append(fp)
        bt.logging.info(f"[batch] ▶️  UID {uid} challenge {meta.challenge_id[:10]}")

    # early‑exit if nothing to send
    if not axons:
        for fp in failed_fps:
            _cleanup_file(fp)
        return

    # fire the RPCs
    async def _query(axon, syn):
        return await v.dendrite(
            axons=[axon],
            synapse=syn,
            deserialize=True,
            timeout=RPC_DEADLINE,
        )

    resp_lists = []
    try:
        resp_lists = await asyncio.wait_for(
            asyncio.gather(
                *[_query(ax, sy) for ax, sy in zip(axons, syns)], return_exceptions=True
            ),
            timeout=HARD_TIMEOUT,
        )
    except asyncio.TimeoutError:
        bt.logging.error(f"[batch] hard timeout after {HARD_TIMEOUT}s")
        return

    # process each miner reply
    successful_sends: list[int] = []

    for uid, meta, fp, t_sent, resp_list in zip(uids, metas, fps, t_sents, resp_lists):
        ok_conn = False

        if isinstance(resp_list, Exception):
            bt.logging.error(f"[batch] dendrite error for UID {uid}: {resp_list}")
            continue
        if not resp_list:
            bt.logging.warning(f"[batch] no response from UID {uid}")
            continue

        ok_conn = True
        resp = resp_list[0]
        miner_hotkey: str = getattr(resp.dendrite, "hotkey", "<unknown>")
        bt.logging.debug(f"[batch] got {type(resp).__name__} from {miner_hotkey}")

        # certificates must go before solutions
        for raw in resp.certificates:
            cert = raw if isinstance(raw, Certificate) else Certificate(**raw)

            if not cert.verify():
                bt.logging.warning(f"[cert] bad signature {cert.challenge_id[:8]}")
                continue
            if cert.validator_hotkey not in v._whitelist:
                bt.logging.warning(
                    f"[cert] hotkey {cert.validator_hotkey[:8]} not whitelisted"
                )
                continue
            try:
                log_certificate_as_solution(cert, miner_hotkey)
                bt.logging.info(
                    f"[cert] ✅ stored {cert.challenge_id[:8]} from UID {uid}"
                )
            except Exception as e:
                bt.logging.error(f"[cert] DB insert failed: {e}", exc_info=True)

        # solutions
        for sol in SolutionExtractor.extract(resp):
            v._sol_proc.process(
                uid=uid, miner_hotkey=miner_hotkey, sol=sol, time_sent=t_sent
            )
        bt.logging.debug(f"[cert] current whitelist = {v._whitelist}")

        # optional difficulty feedback
        desired: float | None = None
        if hasattr(resp, "desired_difficulty") and resp.desired_difficulty is not None:
            desired = resp.desired_difficulty
        else:
            for sol in SolutionExtractor.extract(resp):
                if getattr(sol, "desired_difficulty", None) is not None:
                    desired = sol.desired_difficulty
                    break
        if desired is not None:
            v._diff_cfg.set(uid, max(0.0, float(desired)))
            bt.logging.info(f"[batch] UPDATED difficulty[{uid}] → {desired}")

        # housekeeping
        if ok_conn:
            _cleanup_file(fp)
            successful_sends.append(uid)
        else:
            bt.logging.info(f"[batch] preserving circuit for UID {uid} due to failure")

    bt.logging.info(
        f"[batch] successfully sent {len(successful_sends)} circuits, "
        f"preserved {len(fps) - len(successful_sends)} for retry"
    )


def _cleanup_file(fp: Path | None) -> None:
    """Delete a temporary circuit file; ignore errors."""
    if fp and fp.exists():
        try:
            fp.unlink(missing_ok=True)
            bt.logging.debug(f"[cleanup] deleted circuit file {fp}")
        except Exception:
            bt.logging.warning(f"[cleanup] could not delete {fp}")
