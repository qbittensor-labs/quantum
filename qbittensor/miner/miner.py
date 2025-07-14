# coding: utf-8
"""
Miner for Subnet63 - Quantum.
"""
from __future__ import annotations

from pathlib import Path

import bittensor as bt

# local service layers
from qbittensor.miner.services.circuit_solver import CircuitSolver
from qbittensor.miner.services.synapse_assembler import SynapseAssembler
from qbittensor.protocol import ChallengeCircuits
from qbittensor.miner.services.certificate_verifier import CertificateVerifier
from qbittensor.miner.services.certificate_cleanup import CertificateCleanup

# bootstrap singletons
_BASE_DIR = Path(__file__).resolve().parent
_CERT_DIR = _BASE_DIR / "certificates"
_OLD_CERT_DIR = _BASE_DIR / "old_certificates"
for subdir in ("solved_circuits", "unsolved_circuits"):
    (_BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)

_verifier = CertificateVerifier()
_solver = CircuitSolver(base_dir=_BASE_DIR)
_assembler = SynapseAssembler()
_cleanup = CertificateCleanup(
    cert_dir=_CERT_DIR,
    historical_dir=_OLD_CERT_DIR,
    archive_after_hours=12, # move after 12 h
    delete_after_days=None, # never delete by default – just keep them archived
    cleanup_interval_minutes=60,
)

# internal helpers

def _handle_challenge(syn: ChallengeCircuits) -> ChallengeCircuits:
    # cleanup old certs
    _cleanup.run_cleanup_if_needed()

    # verify certificates just received
    received = _verifier.validate_batch(syn)
    if received:
        save_to = _BASE_DIR / "certificates"
        _verifier.persist(received, save_to)
        bt.logging.info(
            f"[cert] ✅ stored {len(received)} certs "
            f"from {syn.validator_hotkey or '<?>'} in {save_to}"
        )

    # only give back solutions that belong to this validator
    validator = getattr(syn, "validator_hotkey", None)
    ready = _solver.drain(n=100, validator_hotkey=validator)

    # enqueue the new circuit for background solving
    _solver.submit(syn)

    # echo the challenge + our own certificates
    return _assembler.embed(syn, ready, newly_verified=received)


# public symbols exposed to bittensor wrapper

solve_challenge_sync = lambda s, *, wallet: _handle_challenge(s)

handle_challenge_circuits = lambda s, *, wallet: _handle_challenge(s)  # type: ignore[assignment]


# Legacy alias (for old runner scripts importing directly) TODO: remove if safe
_solve_challenge_sync = solve_challenge_sync

bt.logging.info(" Miner initialised - modular certificate gossip active")
