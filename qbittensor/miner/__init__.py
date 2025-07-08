from pathlib import Path

import bittensor as bt

from qbittensor.protocol import ChallengeCircuits

from .services.circuit_solver import CircuitSolver
from .services.reward_verifier import reward_is_valid  # if you made it
from .services.synapse_assembler import SynapseAssembler

_BASE = Path(__file__).resolve().parent

_solver = CircuitSolver(base_dir=_BASE)
_assembler = SynapseAssembler()

DESIRED_DIFFICULTY: float = 0.0

# public handlers expected by Bittensor


def _handle_challenge(syn: ChallengeCircuits) -> ChallengeCircuits:
    _solver.submit(syn)
    validator_hotkey = getattr(syn, "validator_hotkey", None)
    ready = _solver.drain(n=10, validator_hotkey=validator_hotkey)

    syn = _assembler.embed(syn, ready)
    syn.desired_difficulty = DESIRED_DIFFICULTY

    return syn


# aliases that the wrapper looks for

solve_challenge_sync = lambda s, *, wallet: _handle_challenge(s)
handle_challenge_circuits = lambda s, *, wallet: _handle_challenge(s)

bt.logging.info("SUCCESS: Miner initialised. modular certificate gossip active")
