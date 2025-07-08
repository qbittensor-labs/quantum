import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import bittensor as bt
import numpy as np

from qbittensor.validator.database.database_manager import DatabaseManager

# Assuming entanglement_entropy calculation is handled elsewhere or is directly stored in the DB
# from qbittensor.validator.utils.entanglement_entropy import entanglement_entropy


def size_function(
    nqubits: int, knee: int = 32, target_qubits: int = 50, min_score: float = 0.1
) -> float:
    """
    Calculates a size-based score for a given number of qubits.
    This function rewards miners for solving challenges with more qubits,
    with a smooth transition and diminishing returns after a certain knee point.

    Parameters:
    - nqubits: The number of qubits in the quantum challenge.
    - knee: The inflection point (default 32 qubits) where the scoring curve changes.
    - target_qubits: The number of qubits at which the size score reaches 1.0 (default 50).
    - min_score: The minimum score assigned for the lowest practical number of qubits (e.g., 12).

    Returns:
    - float: The calculated size score (G).
    """
    knee_score = 0.4  # Score at the knee point
    if nqubits <= knee:
        # Linear growth up to the knee point
        t = (
            (nqubits - 12) / (knee - 12) if knee > 12 else 0
        )  # Normalize nqubits between 12 and knee
        return min_score + (knee_score - min_score) * t
    else:
        # Polynomial growth above the knee point, approaching 1.0 at target_qubits
        excess = nqubits - knee
        max_excess = target_qubits - knee
        if max_excess <= 0:
            return 1.0  # If target_qubits is not greater than knee, cap at 1.0
        t = excess / max_excess
        return knee_score + (1.0 - knee_score) * (
            t**1.5
        )  # 1.5 power for a gentle curve


class ScoringManager:
    """
    Manages the scoring logic for miners based on their quantum solution submissions.
    It calculates scores on-the-fly from 'entanglement_entropy' and 'nqubits'
    stored in the 'solutions' table, without persisting the combined score.
    It also maintains a 'score_history' for daily aggregations.
    """

    def __init__(self, database_path: str):
        """
        Initializes the ScoringManager with the path to the SQLite database.

        Parameters:
        - database_path: The file path to the SQLite database.
        """
        self.database_path = database_path
        self._ensure_scoring_tables()  # Ensure necessary tables exist
        self.weight_F = 0.3  # Weight for the normalized entanglement entropy score (F)
        self.knee = 32  # Qubit knee point for size_function
        self.target_qubits = 50  # Target qubits for size_function to reach max score
        self.min_G_score = 0.15  # Minimum size score (G)
        self.half_life_hours = 24.0  # Half-life for score decay in hours

    def _ensure_scoring_tables(self) -> None:
        """
        Ensures the 'solutions' and 'score_history' tables exist.
        """
        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            # Create 'solutions' table if it doesn't exist
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

            # Create 'score_history' table if it doesn't exist.
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
        The normalization aims to scale the EE to a value between 0 and 1.

        Parameters:
        - entropy: The calculated entanglement entropy.
        - nqubits: The number of qubits for which the entropy was calculated.

        Returns:
        - float: The normalized entanglement entropy score (F).
        """
        if entropy is None or entropy <= 0 or nqubits is None or nqubits <= 0:
            return 0.0

        ln2 = np.log(2)
        # Smax is the theoretical maximum entanglement entropy for nqubits.
        Smax = (nqubits / 2) * ln2 - ln2**2

        if Smax <= 0:
            return 0.0

        normalized = entropy / Smax
        return max(0.0, min(1.0, normalized))  # Ensure score is within [0, 1]

    def calculate_combined_score(
        self, entropy: float, nqubits: int
    ) -> Tuple[float, float, float]:
        """
        Calculates the combined score for a solution based on entanglement entropy (F)
        and the number of qubits (G). This is the core scoring logic.

        Parameters:
        - entropy: The entanglement entropy of the solution.
        - nqubits: The number of qubits involved in the solution.

        Returns:
        - Tuple[float, float, float]: A tuple containing (normalized_entropy_score_F, size_score_G, combined_score).
        """
        F = self.normalize_ee(entropy, nqubits)  # Entanglement entropy score
        G = size_function(
            nqubits, self.knee, self.target_qubits, self.min_G_score
        )  # Size score

        weight_G = 1 - self.weight_F  # Weight for the size score
        combined = self.weight_F * F + weight_G * G  # Weighted average

        return F, G, combined

    def calculate_single_solution_score(
        self, entropy: float, nqubits: int, is_correct: bool
    ) -> float:
        """
        Calculates the combined score for a single solution.
        Returns 0.0 if the solution is not correct.

        Parameters:
        - entropy: The entanglement entropy of the solution.
        - nqubits: The number of qubits for the solution.
        - is_correct: Boolean indicating if the solution was correct.

        Returns:
        - float: The calculated combined score, or 0.0 if incorrect.
        """
        if not is_correct:
            return 0.0

        # Log warning if entropy is negative, as it's typically non-negative
        if entropy is None or entropy < 0:
            bt.logging.warning(
                f"Invalid entanglement_entropy ({entropy}). Using 0 for score calculation."
            )
            entropy = 0.0  # Use 0 for score calculation if invalid

        _, _, combined_score = self.calculate_combined_score(entropy, nqubits)
        return combined_score

    def calculate_decayed_scores(self, lookback_days: int = 2) -> Dict[int, float]:
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
        decay_constant = (
            math.log(2) / self.half_life_hours
        )  # Decay constant for exponential decay

        scores = defaultdict(
            float
        )  # Use defaultdict to easily sum scores for each miner

        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            # Fetch all correct solutions within the lookback period,
            # including entanglement_entropy and nqubits for on-the-fly calculation.
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

                # Calculate the combined score for this specific solution on-the-fly
                # Only correct solutions contribute to the score.
                combined_score = self.calculate_single_solution_score(
                    entropy, nqubits, is_correct
                )

                if combined_score > 0:  # Only apply decay to positive scores
                    try:
                        score_time = datetime.fromisoformat(timestamp_str)
                        if score_time.tzinfo is None:
                            score_time = score_time.replace(
                                tzinfo=timezone.utc
                            )  # Assume UTC if no timezone info
                    except ValueError:
                        bt.logging.warning(
                            f"Could not parse timestamp {timestamp_str}. Skipping score."
                        )
                        continue

                    age_hours = (
                        current_time - score_time
                    ).total_seconds() / 3600.0  # Age of the solution in hours
                    decay_factor = math.exp(
                        -decay_constant * age_hours
                    )  # Exponential decay
                    decayed_score = combined_score * decay_factor

                    scores[
                        miner_uid
                    ] += decayed_score  # Sum decayed scores for each miner

        except sqlite3.Error as e:
            bt.logging.error(
                f"[scoring] Database error during decayed score calculation: {e}"
            )
        finally:
            db.close()

        # Normalize scores: scale all scores such that the highest score is 1.0
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {uid: score / max_score for uid, score in scores.items()}
            else:
                bt.logging.warning(
                    "[scoring] Max decayed score is 0, cannot normalize. All scores will be 0."
                )
                scores = {uid: 0.0 for uid in scores}  # Set all to 0 if max is 0

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
            # Fetch all correct solutions for today, including EE and nqubits
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

            # Aggregate scores and other stats per miner for today
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

                # Calculate score on-the-fly for this solution
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

                # Insert or replace daily aggregated data into score_history
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

    def get_scoring_stats(self) -> Dict[str, any]:
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
            return {}  # Return empty dict on error
        finally:
            db.close()

    def cleanup(self, retention_days: int = 7) -> None:
        """
        Cleans up old data from the 'solutions' and 'score_history' tables
        to manage database size.

        Parameters:
        - retention_days: Number of days to retain individual solution records.
                          Score history is retained for 30 days.
        """
        # Calculate cutoff times for solutions and score history
        cutoff_time_solutions = datetime.now(timezone.utc) - timedelta(
            days=retention_days
        )
        history_cutoff_date = datetime.now(timezone.utc).date() - timedelta(
            days=30
        )  # Keep history longer

        db = DatabaseManager(self.database_path)
        db.connect()
        try:
            # Delete old records from the SOLUTIONS table
            db.execute_query(
                "DELETE FROM solutions WHERE time_received < ?",
                (cutoff_time_solutions.isoformat(),),
            )
            bt.logging.info(
                f"[scoring] Cleaned up old solutions (retention: {retention_days} days)."
            )

            # Delete old records from the score_history table
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
