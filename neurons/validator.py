import logging
logging.getLogger("torch.autograd").setLevel(logging.ERROR)

import torch
torch.autograd.set_detect_anomaly(False)

import bittensor as bt
from qbittensor.base.validator import BaseValidatorNeuron

from qbittensor.validator.forward import forward


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
            # clean teardown
            if self.is_running:
                self.stop_run_thread()


if __name__ == "__main__":
    Validator().run()
