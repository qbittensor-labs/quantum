"""qbittensor.validator.utils.challenge_logger
Persists challenges and solutions into SQLite (`validator_data.db`).
"""

from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

import bittensor as bt

from qbittensor.validator.database.database_manager import DatabaseManager
from qbittensor.validator.utils.uid_utils import as_int_uid

# paths
_DB_DIR = Path(__file__).resolve().parent.parent / "database"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = str(_DB_DIR / "validator_data.db")
bt.logging.info(f"[logger] Writing DB to: {_DB_PATH}")

_CREATE_CHALLENGES_SQL = """
 CREATE TABLE IF NOT EXISTS challenges (
     id                   INTEGER PRIMARY KEY AUTOINCREMENT,
     challenge_id         TEXT UNIQUE,
     circuit_type         TEXT,
     validator_hotkey     TEXT,
     miner_uid            INTEGER,
     miner_hotkey         TEXT,
     difficulty_level     REAL,
     entanglement_entropy REAL,
     nqubits              INTEGER,
     rqc_depth            INTEGER,
     solution             TEXT
 );
 """

_CREATE_SOLUTIONS_SQL = """
 CREATE TABLE IF NOT EXISTS solutions (
     id                   INTEGER PRIMARY KEY AUTOINCREMENT,
     challenge_id         TEXT UNIQUE,
     circuit_type         TEXT,
     validator_hotkey     TEXT,
     miner_uid            INTEGER,
     miner_hotkey         TEXT,
     miner_solution       TEXT,
     difficulty_level     REAL,
     entanglement_entropy REAL,
     nqubits              INTEGER,
     rqc_depth            INTEGER,
     time_received        TEXT,
     timestamp            TEXT,
     correct_solution     INTEGER CHECK (correct_solution IN (0,1)) DEFAULT NULL,
     reward_sent          INTEGER DEFAULT 0
 );
"""

_EXTRA_COLUMNS = {
    "challenges": {"circuit_type": "TEXT", "miner_hotkey": "TEXT"},
    "solutions": {"circuit_type": "TEXT"},
}


def _add_missing_columns(conn):
    for table, cols in _EXTRA_COLUMNS.items():
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table});")
        present = {row[1] for row in cursor.fetchall()}
        for col, ddl in cols.items():
            if col not in present:
                bt.logging.info(f"[logger] ALTER {table} ADD COLUMN {col}")
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl};")
    conn.commit()


def ensure_tables() -> None:
    with sqlite3.connect(_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(_CREATE_CHALLENGES_SQL)
        cur.execute(_CREATE_SOLUTIONS_SQL)
        conn.commit()
        _add_missing_columns(conn)


ensure_tables()

# inserts
_INSERT_CHALLENGE_SQL = """
 INSERT OR IGNORE INTO challenges (
     challenge_id,
     circuit_type,
     validator_hotkey,
     miner_uid,
     miner_hotkey,
     difficulty_level,
     entanglement_entropy,
     nqubits,
     rqc_depth,
     solution
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
 """

_INSERT_SOLUTION_SQL = """
 INSERT OR IGNORE INTO solutions (
     challenge_id,
     circuit_type,
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
 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


_INSERT_CERT_AS_SOLUTION_SQL = """
INSERT OR IGNORE INTO solutions (
    challenge_id,
    circuit_type,
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
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1);
"""


# helpers
_MAX_LATENCY_HOURS = 48


def _is_correct(
    db: DatabaseManager, challenge_id: str, miner_solution: str | None
) -> int | None:
    if not miner_solution:
        return None
    row = db.fetch_one(
        "SELECT solution FROM challenges WHERE challenge_id = ? LIMIT 1;",
        (challenge_id,),
    )
    return 1 if row and row["solution"] == miner_solution else 0


def log_challenge(
    *,
    challenge_id: str,
    circuit_type: str,
    validator_hotkey: str,
    miner_uid: int,
    miner_hotkey: str,
    difficulty_level: float,
    entanglement_entropy: float,
    nqubits: int,
    rqc_depth: int,
    solution: str,
    time_sent: _dt.datetime | None = None,
) -> None:
    db = DatabaseManager(_DB_PATH)
    db.connect()
    try:
        miner_uid = as_int_uid(miner_uid)
        db.execute_query(
            _INSERT_CHALLENGE_SQL,
            (
                challenge_id,
                circuit_type,
                validator_hotkey,
                miner_uid,
                miner_hotkey,
                difficulty_level,
                entanglement_entropy,
                nqubits,
                rqc_depth,
                solution,
            ),
        )
    finally:
        db.close()


def log_solution(
    *,
    challenge_id: str,
    circuit_type: str,
    validator_hotkey: str,
    miner_uid: int,
    miner_hotkey: str,
    miner_solution: str,
    difficulty_level: float,
    entanglement_entropy: float,
    nqubits: int,
    rqc_depth: int,
    time_received: _dt.datetime | None = None,
) -> None:

    now = time_received or _dt.datetime.utcnow()
    miner_uid = as_int_uid(miner_uid)
    db = DatabaseManager(_DB_PATH)
    db.connect()
    try:
        correct = _is_correct(db, challenge_id, miner_solution)
        db.execute_query(
            _INSERT_SOLUTION_SQL,
            (
                challenge_id,
                circuit_type,
                validator_hotkey,
                miner_uid,
                miner_hotkey,
                miner_solution,
                difficulty_level,
                entanglement_entropy,
                nqubits,
                rqc_depth,
                now.isoformat(timespec="seconds"),
                _dt.datetime.utcnow().isoformat(timespec="seconds"),
                correct,
            ),
        )
    finally:
        db.close()


def log_certificate_as_solution(cert: Certificate, miner_hotkey: str) -> bool:
    """
    Insert the certificate into solutions
    """
    # direct sqlite3 so we can measure changes()
    import sqlite3, datetime as _dt

    conn = sqlite3.connect(_DB_PATH, timeout=30.0)
    try:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA cache_size=10000;")
            conn.execute("PRAGMA busy_timeout=30000;")
        except Exception:
            pass
        before = conn.total_changes  # snapshot
        conn.execute(
            _INSERT_CERT_AS_SOLUTION_SQL,
            (
                cert.challenge_id,
                cert.circuit_type,
                cert.validator_hotkey,
                as_int_uid(cert.miner_uid),
                miner_hotkey or "",
                "<certificate>",
                0.0,
                cert.entanglement_entropy,
                cert.nqubits,
                cert.rqc_depth,
                cert.timestamp,
                _dt.datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return (conn.total_changes - before) == 1
    finally:
        conn.close()
