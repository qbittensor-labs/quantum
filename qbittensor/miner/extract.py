from __future__ import annotations

"""
helper functions for pulling QASM and CIDs
from files or Bittensor synapses.
"""

import json
from pathlib import Path
from typing import Optional

import bittensor as bt

from qbittensor.protocol import ChallengeCircuits

__all__ = ["cid_from_filename", "qasm_from_file", "qasm_from_synapse"]


# Basic heuristics

def cid_from_filename(path: str | Path) -> str:
    """Return everything before first '_' or '.' as the *challenge_id*."""
    name = Path(path).name
    return name.split("_")[0].split(".")[0]


def _is_raw_qasm(text: str) -> bool:
    return text.lstrip().startswith(("OPENQASM", "qreg", "creg"))

# Public extraction helpers
def qasm_from_file(fp: Path) -> Optional[str]:
    """return OPENQASM code contained in fp. else ``None``."""
    try:
        text = fp.read_text()
        # Raw QASM straight in file
        if _is_raw_qasm(text):
            return text

        # JSON file – look for usual keys
        if fp.suffix == ".json":
            data = json.loads(text)
            for k in ("qasm", "circuit_qasm", "circuit_data", "qasm_code"):
                val = data.get(k)
                if isinstance(val, str) and val.strip():
                    return val

        # 3. // header\n\nQASM hack
        if text.startswith("//"):
            _, _, rest = text.partition("\n\n")
            if _is_raw_qasm(rest):
                return rest
    except Exception as exc:
        bt.logging.debug(f"Failed to extract QASM from {fp.name}: {exc}")
    return None


def qasm_from_synapse(syn: ChallengeCircuits) -> Optional[str]:
    """Extract QASM from "circuit_data" of a ChallengeCircuits synapse."""
    payload = syn.circuit_data or ""

    # Raw string
    if isinstance(payload, str):
        if _is_raw_qasm(payload):
            return payload.replace("\\n", "\n")
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload.replace("\\n", "\n")

    if isinstance(payload, dict):
        for k in ("qasm", "circuit_qasm"):
            val = payload.get(k)
            if isinstance(val, str):
                return val

    return None
