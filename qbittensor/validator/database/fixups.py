from pathlib import Path
import sqlite3, datetime as dt

def apply_fixups(db_path, *, backup_days: int = 4):
    db = Path(db_path)
    if not db.exists():
        return

    # backup last 4 days to qbittensor/validator/database/backup_data_YYYYMMDD.db
    backup = db.parent / f"backup_data_{dt.datetime.utcnow():%Y%m%d}.db"
    backup.unlink(missing_ok=True)

    with sqlite3.connect(str(db)) as c:
        c.execute("ATTACH DATABASE ? AS backup;", (str(backup),))
        c.executescript(f"""
            CREATE TABLE backup.solutions AS
              SELECT *
              FROM main.solutions
              WHERE datetime(COALESCE(time_received, timestamp))
                >= datetime('now','-{backup_days} days');

            CREATE TABLE backup.challenges AS
              SELECT c.*
              FROM main.challenges c
              WHERE c.challenge_id IN (SELECT challenge_id FROM backup.solutions);

            DETACH DATABASE backup;

            UPDATE main.solutions
               SET circuit_type = 'hstab'
             WHERE lower(COALESCE(circuit_type,'')) = 'peaked'
               AND CAST(nqubits AS INTEGER) > 41;
        """)
