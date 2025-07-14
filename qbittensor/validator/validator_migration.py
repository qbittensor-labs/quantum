# ─── validator/db_migrations.py ──────────────────────────────────────────────
from __future__ import annotations
import sqlite3, bittensor as bt

def add_difficulty_to_challenges(db_path: str) -> None:
    """
    Ensure challenges.difficulty_level exists and is initialised to 0.0.

    The function is *safe to call on every startup*:
    • If the column already exists, no ALTER TABLE is executed.
    • If the column exists but has NULLs (older runs), they are set to 0.0.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # 1. Does the column already exist?
        cur.execute("PRAGMA table_info(challenges);")
        columns = [row[1] for row in cur.fetchall()]

        if "difficulty_level" not in columns:
            bt.logging.info("[migrate] adding difficulty_level to challenges …")
            cur.execute(
                "ALTER TABLE challenges "
                "ADD COLUMN difficulty_level REAL DEFAULT 0.0;"
            )

        # 2. Back-fill old rows that might still be NULL
        cur.execute(
            "UPDATE challenges SET difficulty_level = 0.0 "
            "WHERE difficulty_level IS NULL;"
        )
        conn.commit()
