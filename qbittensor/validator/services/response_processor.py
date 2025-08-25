from __future__ import annotations

import datetime as dt
from datetime import timezone
from typing import Any, List
import bittensor as bt

from qbittensor.validator.services.solution_extractor import SolutionExtractor
from qbittensor.validator.utils.challenge_logger import (
    log_challenge,
    log_certificate_as_solution,
)
from qbittensor.common.certificate import Certificate
from qbittensor.validator.utils.uid_utils import as_int_uid
from qbittensor.validator.utils.challenge_utils import (
    _convert_peaked_difficulty_to_qubits,
)

RPC_DEADLINE = 5  # seconds
_CUTOFF_TS = dt.datetime(2025, 8, 6, 12, 0, 0, tzinfo=timezone.utc) # legacy certs


class ResponseProcessor:
    """
    Given a queued QItem from ChallengeProducer,
    send the circuit to the miner, process certificates & solutions,
    and update DifficultyConfig
    """

    def __init__(self, validator):
        self.v = validator

    # public entry
    def process(self, item, miner_hotkey: str) -> None:
        uid, syn, meta, target_state, _ = item
        bt.logging.info(f"[send] ▶️ UID {uid} cid={meta.challenge_id[:10]}")
        _service_one_uid(self.v, uid, syn, meta, target_state, miner_hotkey)


# internal worker
def _service_one_uid(
    v, uid: int, syn, meta, target_state: str, miner_hotkey: str
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
            circuit_type=meta.circuit_kind,
            validator_hotkey=meta.validator_hotkey,
            miner_uid=uid,
            miner_hotkey=miner_hotkey,
            difficulty_level=meta.difficulty,
            entanglement_entropy=meta.entanglement_entropy,
            nqubits=meta.nqubits,
            rqc_depth=meta.rqc_depth,
            solution=target_state,
            time_sent=t_sent,
        )
    except Exception:
        bt.logging.error("[single] log_challenge failed", exc_info=True)

    # Record the circuit sent metric before sending
    v.metrics_service.record_circuit_sent(meta.circuit_kind, uid, miner_hotkey)

    # make sure the axon exists
    try:
        axon = v.metagraph.axons[uid]
    except IndexError:
        bt.logging.warning(f"[single] UID {uid} has no axon")
        return

    syn.solution_bitstring = None # hide the answer
    syn.attach_certificates(v.certificate_issuer.pop_for(miner_hotkey))

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

    kind = getattr(resp, "circuit_kind", getattr(meta, "circuit_kind", "")).lower() or "peaked"

    # certificates
    total = inserted = 0
    for raw in resp.certificates:
        total += 1
        cert = raw if isinstance(raw, Certificate) else Certificate(**raw)
        cert_uid = as_int_uid(cert.miner_uid)
        if cert_uid != uid:
            continue
        cert_hkey = getattr(cert, "miner_hotkey", None)

        full_id_ok = (cert_uid == uid and cert_hkey == miner_hotkey)

        # legacy certificate - no hotkey & timestamp before cutoff
        try:
            cert_ts = dt.datetime.fromisoformat(cert.timestamp)
            if cert_ts.tzinfo is None:
                cert_ts = cert_ts.replace(tzinfo=timezone.utc)
            cert_ts = cert_ts.astimezone(timezone.utc)
        except Exception:
            cert_ts = dt.datetime.max.replace(tzinfo=timezone.utc) # force-fail

        legacy_ok = (cert_hkey in (None, "")) and cert_uid == uid and cert_ts < _CUTOFF_TS

        if not (full_id_ok or legacy_ok):
            continue

        # cryptographic proof (handles legacy bytes via Certificate.verify fallback)
        if not cert.verify():
            continue

        if cert.validator_hotkey not in v._whitelist:
            bt.logging.warning(
                f"[cert] hotkey {cert.validator_hotkey[:8]} not whitelisted"
            )
            continue

        try:
            current_hotkey_for_uid = v.metagraph.hotkeys[cert_uid]

            if log_certificate_as_solution(cert, cert_hkey or current_hotkey_for_uid):
                inserted += 1

        except IndexError:
            bt.logging.warning(
                f"[cert] Received gossiped cert for UID {cert.miner_uid}, but UID not in metagraph. Skipping."
            )
        except Exception as exc:
            bt.logging.error(f"[cert] DB insert failed: {exc}", exc_info=True)

    if total:
        bt.logging.info(f"[cert] inserted {inserted} certificates")

    # solutions
    stored = 0
    sols = list(SolutionExtractor.extract(resp))
    for sol in sols:
        # Guarantee per-solution circuit type so SolutionProcessor wont default to peaked
        if not getattr(sol, "circuit_type", None):
            try:
                sol.circuit_type = kind
            except Exception:
                pass
        if v._sol_proc.process(
            uid=uid,
            miner_hotkey=miner_hotkey,
            sol=sol,
            time_sent=t_sent,
        ):
            stored += 1

            # Record the solution received metric
            v.metrics_service.record_solution_received(kind, uid, miner_hotkey)
    if stored:
        bt.logging.info(f"[solution] ✅ Processed solutions from UID {uid}")

    desired = getattr(resp, "desired_difficulty", None)
    if desired is None:
        for sol in SolutionExtractor.extract(resp):
            desired = getattr(sol, "desired_difficulty", None)
            if desired is not None:
                break
    if desired is None:
        return

    cfg = _select_diff_cfg(v, kind) # will raise if kind unknown
    current = float(cfg.get(uid))

    if kind == "hstab":
        cap = 50.0
        new_diff = max(0.0, min(float(desired), cap))

    elif kind == "peaked":
        MIN_Q = 16.0
        MAX_Q = 40.0
        STEP = 7.0

        if 0.0 <= current <= 10.0:
            current = float(_convert_peaked_difficulty_to_qubits(current))

        current_q = current if current > 0.0 else 30.0
        cap = min(MAX_Q, current_q + STEP)

        try:
            desired_val = float(desired)
        except Exception:
            desired_val = current_q

        if 0.0 <= desired_val <= 10.0:
            desired_val = float(_convert_peaked_difficulty_to_qubits(desired_val))

        new_q = max(MIN_Q, min(desired_val, cap))
        new_diff = new_q

    else: # defensive: should never happen
        raise ValueError(f"Unhandled circuit kind {kind!r}")
    bt.logging.debug(
        f"[difficulty] {kind} → file {cfg._path.name} "
        f"uid {uid}: {current:.3f} → {new_diff:.3f}"
    )
    cfg.set(uid, new_diff)

    # Record the difficulty set metric
    v.metrics_service.set_circuit_difficulty(kind, uid, miner_hotkey, new_diff)

def _select_diff_cfg(v, kind: str):
    """
    Map 'peaked' and 'hstab'
    """
    kind = (kind or "").lower()
    try:
        return v._diff_cfg["hstab" if kind.startswith("h") else "peaked"]
    except KeyError:
        raise ValueError(f"Unknown circuit kind {kind!r}")