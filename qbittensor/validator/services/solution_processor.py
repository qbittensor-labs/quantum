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
from qbittensor.validator.utils.challenge_logger import _DB_PATH, log_solution
from qbittensor.validator.services.certificate_issuer import CertificateIssuer
from qbittensor.validator.utils.uid_utils import as_int_uid


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
        # Skip if a correct solution for this challenge is already recorded
        with sqlite3.connect(self._db_path, timeout=30.0) as _conn:
            if _conn.execute(
                "SELECT 1 FROM solutions WHERE challenge_id = ? LIMIT 1;",
                (sol.challenge_id,),
            ).fetchone():
                return False

        ch_row = self._challenge_row(sol.challenge_id)
        if ch_row is None:
            bt.logging.trace(
                f"[solution-proc] no challenge row for {sol.challenge_id[:10]}"
            )
            return False

        expected_uid = as_int_uid(ch_row["miner_uid"])
        if expected_uid != uid:
            return False

        is_correct = self._verify(sol.challenge_id, sol.solution_bitstring)

        ch_row = self._challenge_row(sol.challenge_id)

        # tolerate absent columns gracefully
        def _col(key: str, default):
            return ch_row[key] if (ch_row and key in ch_row.keys()) else default

        challenge_circuit_type = _col("circuit_type", "peaked")
        if getattr(sol, "circuit_type", None) and sol.circuit_type != challenge_circuit_type:
            bt.logging.trace(
                f"[solution-proc] miner-reported circuit_type diff from challenges table"
            )

        # Clamp recorded nqubits for hstab from >50 to 50
        raw_nqubits = _col("nqubits", sol.nqubits or 0)
        if (challenge_circuit_type or "").lower().startswith("hstab"):
            capped_nqubits = min(50, int(raw_nqubits or 0))
        else:
            capped_nqubits = int(raw_nqubits or 0)

        # issue certificate only after _col exists
        if is_correct:
            try:
                self._cert_issuer.issue(
                    challenge_id=sol.challenge_id,
                    miner_uid=uid,
                    miner_hotkey=miner_hotkey,
                    circuit_type=challenge_circuit_type,
                    entanglement_entropy=0.0,
                    nqubits=capped_nqubits,
                    rqc_depth=_col("rqc_depth", sol.rqc_depth or 0),
                )
            except Exception as e:
                bt.logging.error(
                    f"[solution-proc] could not issue cert: {e}", exc_info=True
                )

        try:
            log_solution(
                challenge_id=sol.challenge_id,
                circuit_type=challenge_circuit_type,
                validator_hotkey=_col("validator_hotkey", "<unknown>"),
                miner_uid=uid,
                miner_hotkey=miner_hotkey,
                miner_solution=sol.solution_bitstring,
                difficulty_level=_col("difficulty_level", sol.difficulty_level or 0.0),
                entanglement_entropy=0.0,
                nqubits=capped_nqubits,
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
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA cache_size=10000;")
                conn.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                pass
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
            bt.logging.trace(
                f"[solution-proc] invalid solution"
            )
        return ok

    def _challenge_row(self, cid: str) -> Optional[sqlite3.Row]:
        with sqlite3.connect(self._db_path, timeout=30.0) as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA cache_size=10000;")
                conn.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                pass
            conn.row_factory = sqlite3.Row
            return conn.execute(
                "SELECT * FROM challenges WHERE challenge_id = ? LIMIT 1",
                (cid,),
            ).fetchone()
