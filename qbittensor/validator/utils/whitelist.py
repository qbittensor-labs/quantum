from pathlib import Path
import json
import bittensor as bt

def load_whitelist(fp: Path) -> set[str]:
    """
    Return a set of whitelisted validator hotkeys for certificates
    """
    try:
        with fp.open() as fh:
            data = json.load(fh)
            if isinstance(data, dict) and "whitelist" in data:
                data = data["whitelist"]
            wl = set(data)
            bt.logging.trace(f"Validator whitelist: {wl}")
            return wl
    except FileNotFoundError:
        bt.logging.warning(f"[whitelist] no file at {fp}; accepting none")
        return set()
    except Exception as exc:
        bt.logging.error(f"[whitelist] bad file {fp}: {exc}", exc_info=True)
        return set()
