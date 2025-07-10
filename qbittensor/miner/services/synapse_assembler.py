import json, pathlib
import bittensor as bt
from qbittensor.protocol import ChallengeCircuits
from qbittensor.common.certificate import Certificate


_CERT_DIR = pathlib.Path(__file__).resolve().parents[1] / "certificates"
_CERT_DIR.mkdir(exist_ok=True, parents=True)


class CertificateStore:
    """
    Keeps the last N verified certificates in memory, persists them
    to disk, and hands out a fresh batch on every call to drain().
    """

    def __init__(self, cap: int = 256):
        self._cap = cap
        self._queue: list[dict] = []

    def add(self, certs: list):
        """
        Accepts a mix of Certificate objects **or** already-serialised dicts.
        Normalises everything to a plain dict before queuing / persisting.
        """
        for c in certs:
            # normalize
            if isinstance(c, Certificate):
                data = c.model_dump()
            elif isinstance(c, dict):
                data = c
            else:
                bt.logging.warning(f"[cstore] ignoring unsupported type: {type(c)}")
                continue

            # push into RAM queue
            self._queue.append(data)

            # write to disk (audit / restart-safety)
            cid = data.get("challenge_id", "unknown")
            vhk = data.get("validator_hotkey", "unknown")
            fname = _CERT_DIR / f"{cid}__{vhk}.json"
            fname.write_text(json.dumps(data, separators=(",", ":")))

        # keep only the newest <cap> certs in RAM
        self._queue = self._queue[-self._cap :]

    def drain(self, n: int = 32) -> list[dict]:
        return self._queue[:n]


# a single global store shared by all assemblers
_cstore = CertificateStore()


class SynapseAssembler:
    """Builds the outbound ChallengeCircuits reply."""

    def embed(
        self,
        syn: ChallengeCircuits,
        ready: list[tuple[str, str]],
        newly_verified: list[Certificate] | None = None,
    ) -> ChallengeCircuits:
        # pull every JSON file on disk into memory
        disk_batch = []
        for fp in _CERT_DIR.glob("*.json"):
            try:
                disk_batch.append(json.loads(fp.read_text()))
            except Exception as e:
                bt.logging.warning(f"[assembler] could not load {fp}: {e}")

        if disk_batch:
            _cstore.add(disk_batch)

        # enqueue certificates we just accepted from the validator
        if newly_verified:
            _cstore.add(newly_verified)

        # attach up to 32 certs to *this* synapse
        gossip = _cstore.drain(n=32)
        if gossip:
            syn.attach_certificates(gossip)  # ← top-level list, no JSON wrapper
            bt.logging.info(f"[assembler] 📤 gossiping {len(gossip)} certs")

        # embed solutions exactly like before
        bt.logging.debug(f"[assembler] embedding {len(ready)} solutions")

        if not ready:
            syn.solution_bitstring = ""
        else:
            # Always use batch format for consistency
            syn.solution_bitstring = json.dumps(
                {
                    "type": "batch",
                    "solutions": [
                        {"challenge_id": cid, "solution_bitstring": bits}
                        for cid, bits in ready
                    ],
                },
                separators=(",", ":"),
            )

        bt.logging.info(
            f"[assembler] sending solution payload: {syn.solution_bitstring}"
        )
        return syn
