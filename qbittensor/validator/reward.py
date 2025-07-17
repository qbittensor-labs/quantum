import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional, Any
import bittensor as bt
import numpy as np
from qbittensor.validator.database.database_manager import DatabaseManager

def size_function(
    nqubits: int, knee: int = 32, min_score: float = 0.1, 
    knee_score: float = 0.4,
    exponential_base: float = 2.0
) -> float:
    """
    Calculates a size-based score for a given number of qubits.

    Parameters:
    - nqubits: The number of qubits in the quantum challenge.
    - knee: The inflection point (default 32 qubits) where scoring switches to exponential.
    - min_score: The minimum score assigned.
    - knee_score: The score at the knee point (default 0.4).
    - exponential_base: Base m for m^n (default 2.0).

    Returns:
    - float: The calculated size score.
    """
    min_qubits = 12
    
    if nqubits <= min_qubits:
        return min_score
    if nqubits <= knee:
        t = (nqubits - min_qubits) / (knee - min_qubits)
        return min_score + t * (knee_score - min_score)
    
    exponent = nqubits - knee
    return knee_score * (exponential_base ** exponent)

class ScoringManager:
    """
    Manages the scoring logic for miners based on their quantum solution submissions.
    It calculates scores on-the-fly from 'entanglement_entropy' and 'nqubits'
    stored in the 'solutions' table, without persisting the combined score.
    It also maintains a 'score_history' for daily aggregations.
    """
    def __init__(self, database_path: str):

        self.database_path = database_path
        self._ensure_scoring_tables()
        self.weight_ee = 0.3  # Entropy weight score
        self.knee = 32  # Knee point qubits
        self.min_score = 0.15  # Min score
        self.half_life_hours = 72.0

        bt.logging.info("ScoringManager initialized")

    def _ensure_scoring_tables(self) -> None:
        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            create_solutions_table = """
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id TEXT,
                validator_hotkey TEXT,
                validator_signature TEXT,
                solution_hash TEXT,
                miner_uid INTEGER,
                miner_hotkey TEXT,
                miner_signature TEXT,
                miner_solution TEXT,
                difficulty_level REAL,
                entanglement_entropy REAL,
                nqubits INTEGER,
                rqc_depth INTEGER,
                time_sent TEXT,
                time_received TEXT,
                timestamp TEXT,
                correct_solution INTEGER,
                reward_sent INTEGER
            );
            """
            db.execute_query(create_solutions_table)
            bt.logging.info("[scoring] Ensured 'solutions' table exists.")

            create_score_history_table = """
            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                miner_uid INTEGER,
                date TEXT,
                daily_score REAL,
                solution_count INTEGER,
                avg_entropy REAL,
                avg_qubits REAL,
                created_at TEXT
            );
            """
            db.execute_query(create_score_history_table)
            bt.logging.info("[scoring] Ensured 'score_history' table exists.")

        except sqlite3.Error as e:
            bt.logging.error(f"[scoring] Database error during table setup: {e}")
        finally:
            db.close()

    def normalize_ee(self, entropy: float, nqubits: int) -> float:
        """
        Normalizes the entanglement entropy (EE) based on the number of qubits.
        """
        if entropy is None or entropy <= 0 or nqubits is None or nqubits <= 0:
            return 0.0

        ln2 = np.log(2)
        # Smax is the theoretical maximum entanglement entropy for nqubits.
        Smax = (nqubits / 2) * ln2 - ln2**2

        if Smax <= 0:
            return 0.0

        normalized = entropy / Smax
        return max(0.0, min(1.0, normalized))

    def calculate_combined_score(
        self, entropy: float, nqubits: int
    ) -> Tuple[float, float, float]:
        """
        Calculates the combined score for a solution based on entanglement entropy
        and the number of qubits. This is the core scoring logic.

        Parameters:
        - entropy: The entanglement entropy of the solution.
        - nqubits: The number of qubits involved in the solution.

        Returns:
        - Tuple[float, float, float]: A tuple containing (normalized_entropy_score, size_score, combined_score).
        """
        ee = self.normalize_ee(entropy, nqubits)
        size_func = size_function(
            nqubits, knee=self.knee, min_score=self.min_score
        )

        weight_size = 1 - self.weight_ee
        combined = self.weight_ee * ee + weight_size * size_func

        return ee, size_func, combined

    def calculate_single_solution_score(
        self, entropy: float, nqubits: int, is_correct: bool
    ) -> float:

        if not is_correct:
            return 0.0

        if entropy is None or entropy < 0:
            bt.logging.warning(
                f"Invalid entanglement_entropy ({entropy}). Using 0 for score calculation."
            )
            entropy = 0.0

        _, _, combined_score = self.calculate_combined_score(entropy, nqubits)
        return combined_score

    def calculate_decayed_scores(self, lookback_days: int = 3) -> Dict[int, float]:
        """
        Calculates time-decayed scores for all miners based on their correct solutions
        within a specified lookback period. Scores are normalized to a maximum of 1.0.
        Scores are calculated on-the-fly from stored entropy and qubit count.

        Parameters:
        - lookback_days: The number of days to look back for solutions to include in scoring.

        Returns:
        - Dict[int, float]: A dictionary mapping miner UIDs to their normalized decayed scores.
        """
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(days=lookback_days)
        decay_constant = (math.log(2) / self.half_life_hours)
        scores = defaultdict(float)
        
        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            rows = db.fetch_all(
                """
                SELECT miner_uid, entanglement_entropy, nqubits, time_received, correct_solution
                FROM solutions
                WHERE time_received >= ?
                ORDER BY time_received DESC
                """,
                (cutoff_time.isoformat(),),
            )

            bt.logging.info(
                f"[scoring] Processing {len(rows)} solution records for decayed scores."
            )

            for row in rows:
                miner_uid = row["miner_uid"]
                entropy = row["entanglement_entropy"]
                nqubits = row["nqubits"]
                is_correct = row["correct_solution"] == 1
                timestamp_str = row["time_received"]

                combined_score = self.calculate_single_solution_score(
                    entropy, nqubits, is_correct
                )

                if combined_score > 0:
                    try:
                        score_time = datetime.fromisoformat(timestamp_str)
                        if score_time.tzinfo is None:
                            score_time = score_time.replace(
                                tzinfo=timezone.utc
                            )
                    except ValueError:
                        bt.logging.warning(
                            f"Could not parse timestamp {timestamp_str}. Skipping score."
                        )
                        continue

                    age_hours = (current_time - score_time).total_seconds() / 3600.0  # Age of the solution in hours
                    decay_factor = math.exp(-decay_constant * age_hours)
                    decayed_score = combined_score * decay_factor

                    scores[
                        miner_uid
                    ] += decayed_score

        except sqlite3.Error as e:
            bt.logging.error(
                f"[scoring] Database error during decayed score calculation: {e}"
            )
        finally:
            db.close()

        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {uid: score / max_score for uid, score in scores.items()}
            else:
                bt.logging.warning(
                    "[scoring] Max decayed score is 0, cannot normalize. All scores will be 0."
                )
                scores = {uid: 0.0 for uid in scores}

        bt.logging.info(
            f"[scoring] Calculated normalized decayed scores for {len(scores)} miners."
        )
        return dict(scores)

    def update_daily_score_history(self) -> None:
        """
        Aggregates daily scoring statistics from the 'solutions' table
        and stores them in the 'score_history' table. This provides
        a historical view of miner performance. Scores are calculated on-the-fly.
        """
        today = datetime.now(timezone.utc).date()
        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            rows = db.fetch_all(
                """
                SELECT
                    miner_uid,
                    entanglement_entropy,
                    nqubits,
                    correct_solution
                FROM solutions
                WHERE date(time_received) = ?
                """,
                (today.isoformat(),),
            )

            daily_miner_data = defaultdict(
                lambda: {
                    "total_score": 0.0,
                    "solution_count": 0,
                    "total_entropy": 0.0,
                    "total_qubits": 0,
                }
            )

            for row in rows:
                miner_uid = row["miner_uid"]
                entropy = row["entanglement_entropy"]
                nqubits = row["nqubits"]
                is_correct = row["correct_solution"] == 1

                combined_score = self.calculate_single_solution_score(
                    entropy, nqubits, is_correct
                )

                if is_correct:  # Only count correct solutions for daily stats
                    daily_miner_data[miner_uid]["total_score"] += combined_score
                    daily_miner_data[miner_uid]["solution_count"] += 1
                    daily_miner_data[miner_uid]["total_entropy"] += (
                        entropy if entropy is not None else 0.0
                    )
                    daily_miner_data[miner_uid]["total_qubits"] += (
                        nqubits if nqubits is not None else 0
                    )

            for miner_uid, data in daily_miner_data.items():
                solution_count = data["solution_count"]
                avg_score = (
                    data["total_score"] / solution_count if solution_count > 0 else 0.0
                )
                avg_entropy = (
                    data["total_entropy"] / solution_count
                    if solution_count > 0
                    else 0.0
                )
                avg_qubits = (
                    data["total_qubits"] / solution_count if solution_count > 0 else 0.0
                )

                db.execute_query(
                    """
                    INSERT OR REPLACE INTO score_history (
                        miner_uid, date, daily_score, solution_count,
                        avg_entropy, avg_qubits, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        miner_uid,
                        today.isoformat(),
                        avg_score,
                        solution_count,
                        avg_entropy,
                        avg_qubits,
                        datetime.utcnow().isoformat(),
                    ),
                )

            bt.logging.info(
                f"[scoring] Updated daily history for {len(daily_miner_data)} miners."
            )

        except sqlite3.Error as e:
            bt.logging.error(
                f"[scoring] Database error during daily score history update: {e}"
            )
        finally:
            db.close()

    def get_scoring_stats(self) -> Dict[str, Any]:
        """
        Retrieves recent scoring statistics for monitoring purposes.
        Currently provides stats for the last 24 hours. Scores are calculated on-the-fly.

        Returns:
        - Dict[str, any]: A dictionary containing various scoring statistics.
        """
        db = DatabaseManager(self.database_path)
        db.connect()
        stats = {}
        try:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            # Fetch all solutions (correct or not) in the last 24 hours, including EE and nqubits
            rows = db.fetch_all(
                """
                SELECT
                    miner_uid,
                    entanglement_entropy,
                    nqubits,
                    correct_solution
                FROM solutions
                WHERE time_received >= ?
                """,
                (yesterday.isoformat(),),
            )

            total_score_sum = 0.0
            total_entropy_sum = 0.0
            total_qubits_sum = 0
            correct_solution_count = 0
            active_miners = set()

            for row in rows:
                miner_uid = row["miner_uid"]
                entropy = row["entanglement_entropy"]
                nqubits = row["nqubits"]
                is_correct = row["correct_solution"] == 1

                if is_correct:
                    # Calculate score on-the-fly for this correct solution
                    combined_score = self.calculate_single_solution_score(
                        entropy, nqubits, is_correct
                    )
                    total_score_sum += combined_score
                    total_entropy_sum += entropy if entropy is not None else 0.0
                    total_qubits_sum += nqubits if nqubits is not None else 0
                    correct_solution_count += 1
                    active_miners.add(miner_uid)

            avg_score = (
                total_score_sum / correct_solution_count
                if correct_solution_count > 0
                else 0.0
            )
            avg_entropy = (
                total_entropy_sum / correct_solution_count
                if correct_solution_count > 0
                else 0.0
            )
            avg_qubits = (
                total_qubits_sum / correct_solution_count
                if correct_solution_count > 0
                else 0.0
            )

            stats["last_24h"] = {
                "score_count": correct_solution_count,
                "avg_score": avg_score,
                "avg_entropy": avg_entropy,
                "avg_qubits": avg_qubits,
                "active_miners": len(active_miners),
            }

            bt.logging.info("[scoring] Retrieved scoring statistics.")
            return stats

        except sqlite3.Error as e:
            bt.logging.error(f"[scoring] Database error during stats retrieval: {e}")
            return {}
        finally:
            db.close()

    def cleanup(self, retention_days: int = 7) -> None:

        cutoff_time_solutions = datetime.now(timezone.utc) - timedelta(days=retention_days)
        history_cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=30)

        db = DatabaseManager(self.database_path)
        db.connect()

        try:
            db.execute_query(
                "DELETE FROM solutions WHERE time_received < ?",
                (cutoff_time_solutions.isoformat(),),
            )
            bt.logging.info(
                f"[scoring] Cleaned up old solutions (retention: {retention_days} days)."
            )

            db.execute_query(
                "DELETE FROM score_history WHERE date < ?",
                (history_cutoff_date.isoformat(),),
            )
            bt.logging.info(
                f"[scoring] Cleaned up old score history (retention: 30 days)."
            )

        except sqlite3.Error as e:
            bt.logging.error(f"[scoring] Database error during cleanup: {e}")
        finally:
            db.close()
