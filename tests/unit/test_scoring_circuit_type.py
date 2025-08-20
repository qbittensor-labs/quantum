import os
import sqlite3
from datetime import datetime, timezone
from qbittensor.validator.reward import ScoringManager
from qbittensor.validator.database.database_manager import DatabaseManager


def _create_challenges_table(db_path: str) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS challenges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        challenge_id TEXT UNIQUE,
        circuit_type TEXT
    );
    """
    db = DatabaseManager(db_path)
    db.connect()
    try:
        db.execute_query(ddl)
    finally:
        db.close()


def _insert_challenge(db_path: str, challenge_id: str, circuit_type: str) -> None:
    db = DatabaseManager(db_path)
    db.connect()
    try:
        db.execute_query(
            "INSERT OR IGNORE INTO challenges (challenge_id, circuit_type) VALUES (?, ?);",
            (challenge_id, circuit_type),
        )
    finally:
        db.close()


def _insert_solution(
    db_path: str,
    *,
    challenge_id: str,
    miner_uid: int,
    miner_hotkey: str,
    nqubits: int,
    correct: int,
):
    now = datetime.now(timezone.utc).isoformat()
    db = DatabaseManager(db_path)
    db.connect()
    try:
        db.execute_query(
            """
            INSERT OR IGNORE INTO solutions (
                challenge_id,
                validator_hotkey,
                miner_uid,
                miner_hotkey,
                miner_solution,
                difficulty_level,
                entanglement_entropy,
                nqubits,
                rqc_depth,
                time_received,
                timestamp,
                correct_solution
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                challenge_id,
                "vkey",
                int(miner_uid),
                miner_hotkey,
                "<bits>",
                0.0,
                0.0,
                nqubits,
                0,
                now,
                now,
                int(correct),
            ),
        )
    finally:
        db.close()


def test_scoring_uses_challenge_circuit_type(tmp_path):
    db_path = os.path.join(str(tmp_path), "scoring_test.db")

    mgr = ScoringManager(db_path)
    mgr.weight_ee = 0.0

    _create_challenges_table(db_path)

    _insert_challenge(db_path, "c_peaked", "peaked")
    _insert_challenge(db_path, "c_hstab", "hstab")

    _insert_solution(
        db_path,
        challenge_id="c_peaked",
        miner_uid=1,
        miner_hotkey="hk1",
        nqubits=30,
        correct=1,
    )

    _insert_solution(
        db_path,
        challenge_id="c_hstab",
        miner_uid=2,
        miner_hotkey="hk2",
        nqubits=40,
        correct=1,
    )

    # only peaked counts
    mgr.weight_peaked = 1.0
    mgr.weight_hstab = 0.0
    scores = mgr.calculate_decayed_scores(lookback_days=2)
    assert scores.get(1, 0.0) == 1.0  # miner 1 has peaked credit via challenge type
    assert scores.get(2, 0.0) == 0.0  # miner 2 shouldn't get peaked credit (challenge is hstab)

    # only hstab counts
    mgr.weight_peaked = 0.0
    mgr.weight_hstab = 1.0
    scores2 = mgr.calculate_decayed_scores(lookback_days=2)
    assert scores2.get(1, 0.0) == 0.0  # miner 1 shouldn't get hstab credit (challenge is peaked)
    assert scores2.get(2, 0.0) == 1.0  # miner 2 has hstab credit via challenge type

