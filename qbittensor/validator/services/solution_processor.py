"""
SolutionProcessor. verifies a miners answer, persists it to SQLite
and returns a bool for correctness.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bittensor as bt
from qbittensor.validator.utils.challenge_logger import (
    _DB_PATH,
    log_solution,
    log_solution_with_correctness,
)
from qbittensor.validator.utils.challenge_logger import get_shors_core
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.utils.uid_utils import as_int_uid

# Import verification functions from the canonical Shors verifier
from qbittensor.validator.shor_circuit_creation.q_verify_shor_pro import (
    pack_bits_from_positions,
    spacing_bins,
    choose_radius,
    nearest_center_int,
)


@dataclass(slots=True)
class Solution:
    challenge_id: str
    solution_bitstring: str
    circuit_type: str | None = None
    difficulty_level: float | None = None
    entanglement_entropy: float | None = None
    nqubits: int | None = None
    rqc_depth: int | None = None


class SolutionProcessor:
    def __init__(self, cert_issuer: CertificateIssuer, db_path: Path | str = _DB_PATH):
        self._cert_issuer = cert_issuer
        self._db_path = str(db_path)

    # public
    def process(
        self,
        *,
        uid: int,
        miner_hotkey: str,
        sol: Solution,
        time_sent: dt.datetime,
    ) -> bool:
        """
        Checks the bitstring against the canonical answer in the DB. Logs the attempt via
        `log_solution()`.
        """
        ch_row = self._challenge_row(sol.challenge_id)
        if ch_row is None:
            bt.logging.trace(
                f"[solution-proc] no challenge row for {sol.challenge_id[:10]}"
            )
            return False

        expected_uid = as_int_uid(ch_row["miner_uid"])
        if expected_uid != uid:
            return False

        is_correct = False
        # Shors circuits: verify statistically using counts + core params
        if (
            getattr(sol, "circuit_type", None) or ch_row.get("circuit_type")
        ) == "shors":
            try:
                core = get_shors_core(sol.challenge_id) or {}
                # Expect miner_solution to carry counts JSON or TXT lines; support minimal JSON dict
                counts = self._parse_counts(sol.solution_bitstring)
                is_correct = self._verify_shors(core, counts)
            except Exception as e:
                bt.logging.trace(
                    f"[solution-proc][shors] verify failed: {e}", exc_info=True
                )
                is_correct = False
        else:
            is_correct = self._verify(sol.challenge_id, sol.solution_bitstring)

        ch_row = self._challenge_row(sol.challenge_id)

        # tolerate absent columns gracefully
        def _col(key: str, default):
            return ch_row[key] if (ch_row and key in ch_row.keys()) else default

        challenge_circuit_type = _col("circuit_type", "peaked")
        if (
            getattr(sol, "circuit_type", None)
            and sol.circuit_type != challenge_circuit_type
        ):
            bt.logging.trace(
                f"[solution-proc] miner-reported circuit_type diff from challenges table"
            )

        nqubits = _col("nqubits", sol.nqubits or 0)

        # issue certificate only after _col exists
        if is_correct:
            try:
                self._cert_issuer.issue(
                    challenge_id=sol.challenge_id,
                    miner_uid=uid,
                    miner_hotkey=miner_hotkey,
                    circuit_type=challenge_circuit_type,
                    entanglement_entropy=0.0,
                    nqubits=nqubits,
                    rqc_depth=_col("rqc_depth", sol.rqc_depth or 0),
                )
            except Exception as e:
                bt.logging.error(
                    f"[solution-proc] could not issue cert: {e}", exc_info=True
                )

        try:
            # Shors has no canonical bitstring; log with explicit correctness set by the verifier.
            is_shors = (sol.circuit_type or challenge_circuit_type) == "shors"
            log_fn = (
                (
                    lambda **kw: log_solution_with_correctness(
                        correct=int(1 if is_correct else 0), **kw
                    )
                )
                if is_shors
                else log_solution
            )

            log_fn(
                challenge_id=sol.challenge_id,
                circuit_type=challenge_circuit_type,
                validator_hotkey=_col("validator_hotkey", "<unknown>"),
                miner_uid=uid,
                miner_hotkey=miner_hotkey,
                miner_solution=sol.solution_bitstring,
                difficulty_level=_col("difficulty_level", sol.difficulty_level or 0.0),
                entanglement_entropy=0.0,
                nqubits=nqubits,
                rqc_depth=_col("rqc_depth", sol.rqc_depth or 0),
                time_received=time_sent,
            )
        except Exception as e:
            bt.logging.error(
                f"[solution-proc] log_solution failed for {sol.challenge_id[:10]}: {e}",
                exc_info=True,
            )
        return is_correct

    def highest_correct_difficulty(self, uid: int, circuit_type: str) -> float | None:
        uid = as_int_uid(uid)
        with sqlite3.connect(self._db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT MAX(s.difficulty_level) AS max_difficulty
                FROM   solutions s
                JOIN   challenges c ON c.challenge_id = s.challenge_id
                WHERE  s.miner_uid = ?
                AND    c.circuit_type = ?
                AND    s.correct_solution = 1;
                """,
                (uid, circuit_type),
            ).fetchone()

        val = row["max_difficulty"] if row else None
        return float(val) if val is not None else None

    def allowed_max_difficulty(self, uid: int, circuit_type: str) -> float:
        """
        If the miner has never solved a circuit above 0.0, cap = 0.7
        Otherwise  cap = (highest_solved + 0.4)
        """
        hi = self.highest_correct_difficulty(uid, circuit_type) or 0.0
        return 0.7 if hi <= 0.0 else hi + 0.4

    # private
    def _verify(self, cid: str, bitstring: str) -> bool:
        row = self._challenge_row(cid)
        if not row:
            bt.logging.trace(f"[solution-proc] challenge {cid} not in DB")
            return False

        expected_solution = None
        try:
            expected_solution = row["solution"]
        except (KeyError, IndexError):
            try:
                expected_solution = row["target_state"]
            except (KeyError, IndexError):
                pass

        if expected_solution is None:
            bt.logging.warning(
                f"[solution-proc] no target solution found for challenge {cid}"
            )
            return False

        ok = expected_solution == bitstring
        if not ok:
            bt.logging.trace(f"[solution-proc] invalid solution")
        return ok

    def _challenge_row(self, cid: str) -> Optional[sqlite3.Row]:
        with sqlite3.connect(self._db_path, timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM challenges WHERE challenge_id = ? LIMIT 1",
                (cid,),
            ).fetchone()

    # shors helpers
    def _parse_counts(self, payload: str) -> dict:
        """Accepts a JSON object mapping bitstrings to counts or a TXT-like multi-line string
        Returns dict[str,int] of bitstrings to counts.
        """
        try:
            import json

            obj = json.loads(payload)
            if isinstance(obj, dict):
                return {str(k).replace(" ", ""): int(v) for k, v in obj.items()}
        except Exception:
            pass
        counts: dict[str, int] = {}
        for line in (payload or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                bits, cnt = parts[0], 1
            else:
                bits, cnt = parts[0], parts[1]
            try:
                counts[str(bits).replace(" ", "")] = int(cnt)
            except Exception:
                continue
        return counts

    def _verify_shors(self, core: dict, counts: dict) -> bool:
        """Verify Shors using functions from q_verify_shor_pro (canonical verifier).
        Core expects keys: t_bits, r_value, nonce_delta, k_list
        Counts: dict of bitstrings -> counts
        """
        if not core or not counts:
            return False
        
        t = int(core.get("t_bits") or 0)
        r = int(core.get("r_value") or 0)
        delta_t = int(core.get("nonce_delta") or 0)
        k_list = list(core.get("k_list") or [])
        
        if not k_list or t <= 0 or r <= 0:
            return False

        # Convert counts dict to [(value_int, count)] sorted by count desc
        items = []
        for bits, c in counts.items():
            try:
                v = int(str(bits).replace(" ", ""), 2)
                items.append((v, int(c)))
            except Exception:
                continue
        if not items:
            return False
        items.sort(key=lambda kv: kv[1], reverse=True)

        # Build positions and window parameters using canonical verifier functions
        positions = sorted(int(k) for k in k_list)
        W = len(positions)
        if W <= 0 or W > t:
            return False

        mask = (1 << W) - 1
        delta_w = pack_bits_from_positions(delta_t, positions) & mask
        Delta = spacing_bins(W, r)
        shots = sum(c for _, c in items)
        Bw = choose_radius(Delta, shots, Bw_user=None, coverage_target=0.30)
        
        # Threshold calculation (matches q_verify_shor_pro baseline)
        p0 = min(1.0, (2 * Bw + 1) / max(1.0, Delta))
        import math
        sigma = math.sqrt(max(0.0, min(1.0, p0)) * (1 - max(0.0, min(1.0, p0))) / max(1, shots))
        thr = max(0.15, min(0.35, 3.0 * p0))
        eps = max(0.005, 1.5 * sigma)

        # Score mass within radius (with early exit)
        good = 0
        seen = 0
        for s_raw, c in items:
            sW = pack_bits_from_positions(s_raw, positions)
            sW = (sW + delta_w) & mask
            center = nearest_center_int(sW, W, r)
            d = (sW - center) & mask
            d = min(d, (mask + 1) - d)
            if d <= Bw:
                good += c
            seen += c
            potential = (good + (shots - seen)) / max(1, shots)
            if potential + 1e-12 < (thr - eps):
                break

        mass = good / max(1, shots)
        margin = (mass + eps) - thr
        return margin >= 0.0
