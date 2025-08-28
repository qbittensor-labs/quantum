import json
import pathlib
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

    # Persist, but don't load into the queue.
    def store(self, certs: list, validator_hotkey: str) -> None:
        if not certs:
            bt.logging.trace("[cert] No certs to store.")
            return

        stored_count = 0
        for c in certs:
            # Normalize to dict
            if isinstance(c, Certificate):
                data = c.model_dump()
            elif isinstance(c, dict):
                data = c
            else:
                bt.logging.warning(f"[cstore] ignoring unsupported type: {type(c)}")
                continue

            # Optional: Check for hotkey mismatch
            internal_vhk = data.get("validator_hotkey", "unknown")
            if internal_vhk != validator_hotkey:
                bt.logging.warning(f"[cstore] Hotkey mismatch: cert '{internal_vhk}' != passed '{validator_hotkey}'")

            cid = data.get("challenge_id", "unknown")
            fname = _CERT_DIR / f"{cid}__{validator_hotkey}.json"
            try:
                fname.write_text(json.dumps(data, separators=(",", ":")))
                stored_count += 1
            except Exception as e:
                bt.logging.warning(f"[cstore] Failed to write {fname}: {e}")

        bt.logging.trace(
            f"[cert] ✅ stored {stored_count} certs "
            f"from {validator_hotkey or '<?>'} in {_CERT_DIR}"
        )

    # Collect all certificates from disk
    def load(self, validator_hotkey: str, n: int = 10) -> list[dict]:
        certs: list[dict] = []
        file_paths = sorted(
            _CERT_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:n]

        for fp in file_paths:
            try:
                data = json.loads(fp.read_text())
                vhk = data.get("validator_hotkey", "unknown")
                if vhk != validator_hotkey:
                    certs.append(data)
            except Exception as e:
                bt.logging.warning(f"[assembler] could not load {fp}: {e}")

        return certs

class SynapseAssembler:
    """Builds the outbound ChallengeCircuits reply."""

    def embed(
        self,
        syn: ChallengeCircuits,
        ready: list[tuple[str, str]],
        newly_verified: list[Certificate] | None = None,
        validator_hotkey: str = "",
    ) -> ChallengeCircuits:
        cstore = CertificateStore()
        gossip = cstore.load(validator_hotkey=validator_hotkey, n=1000)

        # save certificates we just accepted from the validator, but don't include them
        if newly_verified:
            cstore.store(newly_verified, validator_hotkey)

        # attach up to certs to *this* synapse
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