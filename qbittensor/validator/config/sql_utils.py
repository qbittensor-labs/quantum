import sqlite3
from pathlib import Path

def max_solved_difficulty(db_path: Path, miner_hotkey: str) -> float:
    """
    Returns the highest difficulty_level the given miner has solved
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute(
                """
                SELECT MAX(difficulty_level)
                FROM   solutions
                WHERE  miner_hotkey = ?
                  AND  correct_solution = 1
                  AND circuit_type = 'peaked'
                """,
                (miner_hotkey,),
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
    except Exception as exc:
        import bittensor as bt
        bt.logging.warning(f"[difficulty] could not read DB: {exc}")
        return 0.0
