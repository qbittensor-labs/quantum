from __future__ import annotations

import datetime as dt
from typing import Any, List
import bittensor as bt

from qbittensor.validator.services.solution_extractor import SolutionExtractor
from qbittensor.validator.utils.challenge_logger import (
    log_challenge,
    log_certificate_as_solution,
)
from qbittensor.common.certificate import Certificate

RPC_DEADLINE = 10  # seconds


class ResponseProcessor:
    """
    Given a queued QItem from ChallengeProducer,
    send the circuit to the miner, process certificates & solutions,
    and update DifficultyConfig
    """

    def __init__(self, validator: "Validator"):
        self.v = validator

    # public entry
    def process(self, item: "ChallengeProducer.QItem") -> None:
        uid, syn, meta, target_state, _ = item
        bt.logging.info(f"[send] ▶️  UID {uid}   cid={meta.challenge_id[:10]}")
        _service_one_uid(self.v, uid, syn, meta, target_state)


# internal worker
def _service_one_uid(
    v: "Validator",
    uid: int,
    syn,
    meta,
    target_state: str,
) -> None:
    """
    Send syn to miner *uid* and handle the reply.
    All heavy lifting (certs, solutions, diff feedback)
    """

    t_sent = dt.datetime.utcnow()

    # log challenge
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
        bt.logging.error("[single] log_challenge failed", exc_info=True)

    # make sure the axon exists
    try:
        axon = v.metagraph.axons[uid]
    except IndexError:
        bt.logging.warning(f"[single] UID {uid} has no axon")
        return

    syn.solution_bitstring = None  # hide the answer
    syn.attach_certificates(v.certificate_issuer.pop_for(uid))

    # blocking RPC
    try:
        resp_list: List[Any] = v.dendrite.query(
            axons=[axon],
            synapse=syn,
            deserialize=True,
            timeout=RPC_DEADLINE,
        )
    except Exception as e:
        bt.logging.error(f"[single] dendrite error for UID {uid}: {e}")
        return

    if not resp_list:
        bt.logging.warning(f"[single] no response from UID {uid}")
        return

    resp = resp_list[0]
    miner_hotkey = getattr(resp.dendrite, "hotkey", "<unknown>")

    # certificates
    total = inserted = 0
    for raw in resp.certificates:
        total += 1
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
            if log_certificate_as_solution(cert, miner_hotkey):
                inserted += 1
        except Exception as exc:
            bt.logging.error(f"[cert] DB insert failed: {exc}", exc_info=True)

    if total:
        bt.logging.info(
            f"[cert] received {total}, inserted {inserted}, "
            f"skipped {total - inserted} from UID {uid}"
        )

    # solutions
    stored = 0
    for sol in SolutionExtractor.extract(resp):
        if v._sol_proc.process(
            uid=uid,
            miner_hotkey=miner_hotkey,
            sol=sol,
            time_sent=t_sent,
        ):
            stored += 1

    if stored:
        bt.logging.info(f"[solution] ✅ stored {stored} from UID {uid}")

    # difficulty feedback
    desired = getattr(resp, "desired_difficulty", None)
    if desired is None:
        for sol in SolutionExtractor.extract(resp):
            desired = getattr(sol, "desired_difficulty", None)
            if desired is not None:
                break

    if desired is not None:
        allowed_max = v._sol_proc.allowed_max_difficulty(uid)
        new_diff = max(0.0, min(float(desired), allowed_max))
        v._diff_cfg.set(uid, new_diff)
        bt.logging.info(
            f"[single] UPDATED diff[{uid}] → {new_diff:.2f} "
            f"(req {desired:.2f}, ≤ {allowed_max:.2f})"
        )
