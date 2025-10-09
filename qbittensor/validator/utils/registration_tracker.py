"""
Registration tracking utilities
"""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import bittensor as bt


def initialize_registration_table(db_path: Path, metagraph) -> None:
    """
    Initialize the registration_time table with all current UIDs on first run.
    Sets time_first_seen to January 1, 2025 (dummy value) for initial setup.
    
    Args:
        db_path: Path to the validator database
        metagraph: The validator's metagraph instance
    """
    try:
        with sqlite3.connect(str(db_path), timeout=3.0) as conn:
            # Check if table is empty (first run only)
            count = conn.execute("SELECT COUNT(*) FROM registration_time").fetchone()[0]
            if count > 0:
                bt.logging.trace("[registration] Table already initialized, skipping")
                return
            
            # Batch insert all UIDs with January 1, 2025 (dummy value)
            initial_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()
            rows = [(uid, metagraph.hotkeys[uid], initial_time) 
                    for uid in range(min(256, len(metagraph.hotkeys))) 
                    if metagraph.hotkeys[uid]]
            
            conn.executemany(
                "INSERT OR IGNORE INTO registration_time (uid, hotkey, time_first_seen) VALUES (?, ?, ?)",
                rows
            )
            conn.commit()
            bt.logging.trace("[registration] Registration table initialized")
    except Exception as e:
        bt.logging.error(f"[registration] Initialization error: {e}")


def check_hotkey_changed(db_path: Path, uid: int, hotkey: str) -> bool:
    """
    Check if a UID exists in the database and if its hotkey differs.
    
    Args:
        db_path: Path to the validator database
        uid: The miner's UID
        hotkey: The miner's current hotkey
        
    Returns:
        True if UID doesn't exist OR hotkey changed, False if same hotkey
    """
    try:
        with sqlite3.connect(str(db_path), timeout=3.0) as conn:
            conn.row_factory = sqlite3.Row
            result = conn.execute(
                "SELECT hotkey FROM registration_time WHERE uid = ?",
                (uid,)
            ).fetchone()
            
            if result is None:
                # UID doesn't exist in database - it's new
                return True
            
            # UID exists - check if hotkey changed
            return result['hotkey'] != hotkey
            
    except sqlite3.Error as e:
        bt.logging.error(f"[registration] Database error checking uid {uid}: {e}")
        return False


def register_miner_if_new(db_path: Path, uid: int, hotkey: str) -> None:
    """
    Register/update a miner in the registration_time table.
    Only updates if the UID is new OR the hotkey changed.
    
    Behavior:
    - If UID doesn't exist: Inserts with current time
    - If UID exists with different hotkey: Updates hotkey and time to NOW
    - If UID exists with same hotkey: Does nothing (preserves original time)
    
    Args:
        db_path: Path to the validator database
        uid: The miner's UID
        hotkey: The miner's hotkey
    """
    try:
        current_time = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(db_path), timeout=3.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO registration_time (uid, hotkey, time_first_seen) VALUES (?, ?, ?)",
                (uid, hotkey, current_time)
            )
            conn.commit()
            bt.logging.info(f"[registration] Registered miner uid={uid} hotkey={hotkey}")
    except sqlite3.Error as e:
        bt.logging.error(f"[registration] Database error registering uid {uid}: {e}")

