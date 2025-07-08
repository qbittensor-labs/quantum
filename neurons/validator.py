import asyncio

import bittensor as bt
from qbittensor.base.validator import BaseValidatorNeuron
from qbittensor.validator.forward import forward


async def main_loop(v: "Validator"):
    while True:
        try:
            bt.logging.info(f"step={v.step} uid={v.uid}")
            v.metagraph.sync(subtensor=v.subtensor)
            await v.forward()
            await asyncio.sleep(30)  # pacing; adjust as you like
            v.step += 1
        except KeyboardInterrupt:
            bt.logging.warning("Shutting down…")
            break
        except Exception as e:
            bt.logging.error(f"loop error: {e}", exc_info=True)
            await asyncio.sleep(30)


class Validator(BaseValidatorNeuron):
    """Thin wrapper around BaseValidatorNeuron with custom forward loop."""

    async def forward(self):
        return await forward(self)

    async def shutdown_background(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error("Hotkey not registered, aborting")
            return

        try:
            asyncio.run(main_loop(self))
        finally:
            if hasattr(self, "shutdown_background"):
                asyncio.run(self.shutdown_background())
            if self.is_running:
                self.stop_run_thread()


if __name__ == "__main__":
    Validator().run()
