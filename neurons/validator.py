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

from qbittensor.validator.forward import forward
from qbittensor.validator.services.metrics import MetricsService
from qbittensor.validator.utils.auto_updater import start_updater, stop_updater

CLEANUP_FLAG = Path("/tmp/validator_cleanup_done")

def _graceful_shutdown(signum, frame):
    bt.logging.info(f"[validator] Received signal {signum}, shutting down gracefully...")
    #stop_updater()
    try:
        CLEANUP_FLAG.write_text("done")
        bt.logging.info("[validator] Cleanup flag written")
    except Exception as e:
        bt.logging.warning(f"[validator] Could not write cleanup flag: {e}")
    
    sys.exit(0)


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

        self.metrics_service = MetricsService(validator_hotkey=self.wallet.hotkey.ss58_address, network=self.subtensor.network)

        # Validator git repo update worker
        #start_updater(check_interval_minutes=5)

        try:
            while True:
                try:
                    self.metrics_service.record_heardbeat()
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
            self.metrics_service.shutdown()

            bt.logging.info("Stopping auto-updater...")
            #stop_updater()
            if self.is_running:
                self.stop_run_thread()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    
    try:
        if CLEANUP_FLAG.exists():
            CLEANUP_FLAG.unlink()
    except Exception:
        pass

    Validator().run()
