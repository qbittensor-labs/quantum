import argparse
import time
import typing
from typing import Union

import bittensor as bt
from qbittensor.base.miner import BaseMinerNeuron
from qbittensor.miner.miner import _solve_challenge_sync
from qbittensor.protocol import ChallengePeakedCircuit, ChallengeHStabCircuit, _CircuitSynapseBase

SYNC_INTERVAL_S = 300  # 5-minute metagraph sync, keeps logs tidy


class Miner(BaseMinerNeuron):
    
    def __init__(self, config=None):
        super().__init__(config=config)
        
        self.axon = bt.axon(
            wallet=self.wallet,
            port=self.config.axon.port,
            external_ip=self.config.axon.external_ip,
        )

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
    
    # Default forward method to satisfy abstract class requirement
    async def forward(self, synapse: _CircuitSynapseBase) -> _CircuitSynapseBase:
        """
        Default forward method. routes to appropriate handler based on synapse type
        """
        if isinstance(synapse, ChallengePeakedCircuit):
            return await self.forward_peaked(synapse)
        elif isinstance(synapse, ChallengeHStabCircuit):
            return await self.forward_hstab(synapse)
        else:
            raise ValueError(f"Unknown synapse type: {type(synapse)}")
    
    # Default blacklist method to satisfy abstract class requirement
    async def blacklist(self, synapse: _CircuitSynapseBase) -> typing.Tuple[bool, str]:
        """
        Default blacklist method
        """
        if isinstance(synapse, ChallengePeakedCircuit):
            return await self.blacklist_peaked(synapse)
        elif isinstance(synapse, ChallengeHStabCircuit):
            return await self.blacklist_hstab(synapse)
        else:
            return True, f"Unknown synapse type: {type(synapse)}"
    
    # Default priority method to satisfy abstract class requirement
    async def priority(self, synapse: _CircuitSynapseBase) -> float:
        """
        Default priority method
        """
        if isinstance(synapse, ChallengePeakedCircuit):
            return await self.priority_peaked(synapse)
        elif isinstance(synapse, ChallengeHStabCircuit):
            return await self.priority_hstab(synapse)
        else:
            return 0.0
    
    # ChallengePeakedCircuit handler
    async def forward_peaked(self, synapse: ChallengePeakedCircuit) -> ChallengePeakedCircuit:
        bt.logging.info(f"Peaked circuit received")
        
        # Set the desired difficulty before processing
        synapse.desired_difficulty = self.config.difficulty
        
        return _solve_challenge_sync(synapse, wallet=self.wallet)
    
    # ChallengeHStabCircuit handler
    async def forward_hstab(self, synapse: ChallengeHStabCircuit) -> ChallengeHStabCircuit:
        bt.logging.info(f"H-Stab circuit received")
        
        # Set the desired difficulty before processing
        synapse.desired_difficulty = self.config.difficulty
        
        return _solve_challenge_sync(synapse, wallet=self.wallet)

    # Blacklist for ChallengePeakedCircuit
    async def blacklist_peaked(self, synapse: ChallengePeakedCircuit) -> typing.Tuple[bool, str]:
        return await self._blacklist_common(synapse)
    
    # Blacklist for ChallengeHStabCircuit
    async def blacklist_hstab(self, synapse: ChallengeHStabCircuit) -> typing.Tuple[bool, str]:
        return await self._blacklist_common(synapse)
    
    # Common blacklist logic
    async def _blacklist_common(self, synapse: Union[ChallengePeakedCircuit, ChallengeHStabCircuit]) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Handle un-registered hotkey
            if not self.config.blacklist.allow_non_registered:
                bt.logging.trace(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"
            else:
                # If we allow non-registered, we can't check validator permit
                bt.logging.trace(
                    f"Allowing non-registered hotkey {synapse.dendrite.hotkey}"
                )
                return False, "Hotkey allowed (non-registered)"

        # Now we know the hotkey is registered, get the UID
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self.config.blacklist.force_validator_permit:
            # only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    # Priority for ChallengePeakedCircuit
    async def priority_peaked(self, synapse: ChallengePeakedCircuit) -> float:
        return await self._priority_common(synapse)
    
    # Priority for ChallengeHStabCircuit
    async def priority_hstab(self, synapse: ChallengeHStabCircuit) -> float:
        return await self._priority_common(synapse)
    
    # Common priority logic
    async def _priority_common(self, synapse: Union[ChallengePeakedCircuit, ChallengeHStabCircuit]) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # Check if hotkey is registered
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        ) # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        ) # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority
    
    def run(self):
        """
        Override run method to register multiple synapse handlers
        """
        # Check that miner is registered on the network.
        self.sync()

        # Register handlers for both synapse types
        self.axon.attach(
            forward_fn=self.forward_peaked,
            blacklist_fn=self.blacklist_peaked,
            priority_fn=self.priority_peaked,
        ).attach(
            forward_fn=self.forward_hstab,
            blacklist_fn=self.blacklist_hstab,
            priority_fn=self.priority_hstab,
        )

        # Start the axon server
        self.axon.start()

        # Log startup
        bt.logging.info(
            f"Miner running on network: {self.config.subtensor.chain_endpoint} | "
            f"block: {self.block} | step: {self.step} | uid: {self.uid} | "
        )
        bt.logging.info(
            f"Miner is accepting both ChallengePeakedCircuit and ChallengeHStabCircuit synapses"
        )

        # Main loop
        while True:
            try:
                # exit
                if self.should_exit:
                    break

                # Sync metagraph periodically
                if self.should_sync_metagraph():
                    self.sync()

                # Sleep for a bit
                time.sleep(5)

            except KeyboardInterrupt:
                # Graceful shutdown
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(f"Miner error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Miner.add_args(parser)
    parser.add_argument(
        "--difficulty",
        type=float,
        default=0.0,
        help="Desired difficulty for circuit challenges",
    )
    config = bt.config(parser)
    
    # Set blacklist config to avoid security warnings
    if not hasattr(config, 'blacklist'):
        config.blacklist = bt.config()
    config.blacklist.allow_non_registered = False
    config.blacklist.force_validator_permit = True
    bt.logging.info(f"Launching miner with desired difficulty: {config.difficulty}")
    Miner(config=config).run()