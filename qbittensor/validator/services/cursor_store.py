from pathlib import Path
class CursorStore:
    def __init__(self, path: Path):
        self._path = path
    def load(self) -> int | None:
        try:
            return int(self._path.read_text())
        except Exception:
            return None
    def save(self, uid: int) -> None:
        try:
            self._path.write_text(str(uid))
        except Exception as exc:
            import bittensor as bt
            bt.logging.warning(f"[cursor] {exc}")
