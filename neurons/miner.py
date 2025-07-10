import argparse
import time
import typing

import bittensor as bt
from qbittensor.base.miner import BaseMinerNeuron
from qbittensor.miner.miner import _solve_challenge_sync
from qbittensor.protocol import ChallengeCircuits

SYNC_INTERVAL_S = 300  # 5-minute metagraph sync, keeps logs tidy

# Desired difficulty level for circuit challenges
DESIRED_DIFFICULTY = 0.0  # Change this value to request different difficulty levels


class Miner(BaseMinerNeuron):

    # periodic metagraph sync
    def should_sync_metagraph(self) -> bool:
        now = time.time()
        last = getattr(self, "_last_sync_ts", 0)
        if now - last >= SYNC_INTERVAL_S:
            self._last_sync_ts = now
            return True
        return False
    def save_state(self):
        """Silencing log spam, will implement soon"""
        pass
    # main RPC handler
    async def forward(self, synapse: ChallengeCircuits) -> ChallengeCircuits:
        bt.logging.info(f"circuit received")
        
        # Set the desired difficulty before processing
        synapse.desired_difficulty = DESIRED_DIFFICULTY
        
        return _solve_challenge_sync(synapse, wallet=self.wallet)

    async def blacklist(self, synapse: ChallengeCircuits) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"


    async def priority(self, synapse: ChallengeCircuits) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Miner.add_args(parser)
    config = bt.config(parser)
    
    # Set blacklist config to avoid security warnings
    if not hasattr(config, 'blacklist'):
        config.blacklist = bt.config()
    config.blacklist.allow_non_registered = False
    config.blacklist.force_validator_permit = True

    bt.logging.info(f"Launching miner with desired difficulty: {DESIRED_DIFFICULTY}")
    Miner(config=config).run()
