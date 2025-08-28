import os, time, json, pathlib
from typing import TYPE_CHECKING, List, Dict
from pathlib import Path
import bittensor as bt

if TYPE_CHECKING:
    from qbittensor.validator.services.certificate_issuer import CertificateIssuer

CERT_SWEEP_INTERVAL_S = float(os.getenv("CERT_SWEEP_INTERVAL_S", str(60 * 60)))  # 1 hour
CERT_KEEP_AGE_S       = float(os.getenv("CERT_KEEP_AGE_S", str(48 * 3600))) # 48 hours

_BASE   = (pathlib.Path(__file__).resolve().parents[2] / "certificates")
_OUTBOX = _BASE / "pending"
_SENT   = _BASE / "sent"

_BASE.mkdir(exist_ok=True)
_OUTBOX.mkdir(exist_ok=True)
_SENT.mkdir(exist_ok=True)

class CertificateManager:
    """
    Periodically sweeps pending/ into sent/ for items older than CERT_KEEP_AGE_S (48h)
    """
    def __init__(self, issuer: "CertificateIssuer"):
        self.issuer = issuer
        self.last_sweep_time = 0.0

    def update(self) -> None:
        now = time.time()
        if (now - self.last_sweep_time) < CERT_SWEEP_INTERVAL_S:
            return

        cutoff = now - CERT_KEEP_AGE_S
        try:
            summary = self.sweep_outbox_to_sent(cutoff_ts=cutoff)
            if summary:
                total = sum(summary.values())
                bt.logging.info(f"[cert] swept {total} old certificates to sent/")
            else:
                bt.logging.debug("[cert] sweep: nothing to move")
        except Exception:
            bt.logging.warning("[cert] sweep error", exc_info=True)
        finally:
            self.last_sweep_time = now

    def pop_for(
        self,
        miner_hotkey: str,
        *,
        max_items: int | None = None,
        max_age_s: int | None = None,
    ) -> List[dict]:
        """Return & MOVE recent certs pending/<hk> -> sent/<hk> (filtered by mtime if max_age_s)."""
        miner_dir = _OUTBOX / miner_hotkey
        if not miner_dir.exists():
            return []
        now = time.time()
        files = list(miner_dir.iterdir())
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if max_age_s is not None:
            files = [p for p in files if (now - p.stat().st_mtime) <= max_age_s]
        if max_items is not None:
            files = files[:max_items]

        sent_dir = _SENT / miner_hotkey
        sent_dir.mkdir(exist_ok=True)

        out: List[dict] = []
        for f in files:
            try:
                dst = sent_dir / f.name
                os.replace(f, dst)
                out.append(json.loads(dst.read_text()))
            except Exception as e:
                bt.logging.warning(f"[cert] pop_for error on {f}: {e}")
        bt.logging.info(f"[cert] attaching {len(out)} certificates to synapse for {miner_hotkey}")
        return out

    def sweep_outbox_to_sent(self, *, cutoff_ts: float) -> Dict[str, int]:
        """
        Move all certificates strictly older than cutoff_ts from pending/* -> sent/*.
        Returns a summary {miner_hotkey: moved_count}
        """
        moved_by_hk: Dict[str, int] = {}
        for miner_dir in _OUTBOX.iterdir():
            if not miner_dir.is_dir():
                continue
            sent_dir = _SENT / miner_dir.name
            sent_dir.mkdir(exist_ok=True)

            count = 0
            for f in list(miner_dir.iterdir()):
                try:
                    if f.stat().st_mtime < cutoff_ts:
                        os.replace(f, sent_dir / f.name)
                        count += 1
                except FileNotFoundError:
                    pass
                except Exception as e:
                    bt.logging.warning(f("[cert] sweep could not move {f}: {e}"))
            if count:
                moved_by_hk[miner_dir.name] = count
        return moved_by_hk
