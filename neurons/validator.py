import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.getLogger("torch.autograd").setLevel(logging.ERROR)

import torch
torch.autograd.set_detect_anomaly(False)

import bittensor as bt
import signal
import sys
from pathlib import Path
from qbittensor.base.validator import BaseValidatorNeuron

from qbittensor.validator.forward import forward, shutdown
from qbittensor.validator.utils.auto_updater import start_updater, stop_updater

CLEANUP_FLAG = Path("/tmp/validator_cleanup_done")

def _graceful_shutdown(signum, frame):
    bt.logging.info(f"[validator] Received signal {signum}, shutting down gracefully...")
    try:
        CLEANUP_FLAG.write_text("done")
    except Exception as e:
        bt.logging.warning(f"[validator] Could not write cleanup flag: {e}")
    try:
        global _VALIDATOR_SINGLETON
        if _VALIDATOR_SINGLETON is not None:
            import os
            timeout_env = os.getenv("VALIDATOR_SHUTDOWN_TIMEOUT_S")
            timeout_s = float(timeout_env) if timeout_env else None
            shutdown(_VALIDATOR_SINGLETON, timeout_s=timeout_s)
    except Exception:
        bt.logging.error("[validator] error during graceful shutdown", exc_info=True)
    finally:
        # Exit non‑zero so PM2 autorestart brings the process back up automatically
        sys.exit(1)


class Validator(BaseValidatorNeuron):
    """Thin wrapper around BaseValidatorNeuron with a synchronous forward loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.step = 0 

    def forward(self):
        return forward(self)

    def run(self):
        # sanity-check hotkey
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error("Hotkey not registered, aborting")
            return

        bt.logging.info("Validator starting (sync mode)")

        # Validator git repo update worker
        #start_updater(check_interval_minutes=5)

        global _VALIDATOR_SINGLETON
        _VALIDATOR_SINGLETON = self
        try:
            while True:
                try:
                    #bt.logging.info(f"step={self.step} uid={self.uid}")
                    self.metagraph.sync(subtensor=self.subtensor)
                    self.forward() # sync now
                    #self.step += 1
                except KeyboardInterrupt:
                    bt.logging.warning("Shutting down…")
                    break
                except Exception as e:
                    bt.logging.error(f"loop error: {e}", exc_info=True)
        finally:
            bt.logging.info("Stopping auto-updater...")
            #stop_updater()
            if self.is_running:
                self.stop_run_thread()
            try:
                shutdown(self, timeout_s=1800.0)
            except Exception:
                pass


_VALIDATOR_SINGLETON = None

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    
    try:
        if CLEANUP_FLAG.exists():
            CLEANUP_FLAG.unlink()
    except Exception:
        pass

    Validator().run()
