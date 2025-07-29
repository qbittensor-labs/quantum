# coding: utf-8
"""
Miner for Subnet63 - Quantum.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import bittensor as bt

from qbittensor.miner.services.circuit_solver import CircuitSolver
from qbittensor.miner.services.synapse_assembler import SynapseAssembler
from qbittensor.protocol import ChallengePeakedCircuit, ChallengeHStabCircuit, _CircuitSynapseBase
from qbittensor.miner.services.certificate_verifier import CertificateVerifier
from qbittensor.miner.services.certificate_cleanup import CertificateCleanup

CircuitSynapse = Union[ChallengePeakedCircuit, ChallengeHStabCircuit]

# bootstrap singletons
_BASE_DIR = Path(__file__).resolve().parent
_CERT_DIR = _BASE_DIR / "certificates"
_OLD_CERT_DIR = _BASE_DIR / "old_certificates"
for subdir in ("solved_circuits", "unsolved_circuits", "peaked_circuits", "peaked_circuits/solved_circuits", 
    "peaked_circuits/unsolved_circuits", "hstab_circuits", "hstab_circuits/solved_circuits",
    "hstab_circuits/unsolved_circuits"):
    (_BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)

_verifier = CertificateVerifier()
_circuit_solver = CircuitSolver(base_dir=_BASE_DIR)

_assembler = SynapseAssembler()
_cleanup = CertificateCleanup(
    cert_dir=_CERT_DIR,
    historical_dir=_OLD_CERT_DIR,
    archive_after_hours=12,
    delete_after_days=None,
    cleanup_interval_minutes=60,
)

# Solver registry
SOLVERS = {
    ChallengePeakedCircuit: _circuit_solver,
    ChallengeHStabCircuit: _circuit_solver,
}

def _get_solver_for_synapse(syn: CircuitSynapse) -> CircuitSolver:
    """Get the appropriate solver based on synapse type."""
    solver = SOLVERS.get(type(syn), _circuit_solver)
    bt.logging.debug(f"Using {solver} for synapse type {type(syn).__name__}")
    return solver

def _handle_challenge(syn: CircuitSynapse) -> CircuitSynapse:
    """routes to appropriate solver based on synapse type."""
    # cleanup old certs
    _cleanup.run_cleanup_if_needed()

    # verify certificates just received
    received = _verifier.validate_batch(syn)
    if received:
        save_to = _BASE_DIR / "certificates"
        _verifier.persist(received, save_to)
        bt.logging.trace(
            f"[cert] ✅ stored {len(received)} certs "
            f"from {syn.validator_hotkey or '<?>'} in {save_to}"
        )

    solver = _get_solver_for_synapse(syn)
    
    validator = getattr(syn, "validator_hotkey", None)
    ready = solver.drain(n=100, validator_hotkey=validator)

    solver.submit(syn)
    
    circuit_type = getattr(syn, 'circuit_kind', type(syn).__name__)
    bt.logging.info(f"Processing {circuit_type} circuit from {validator or 'unknown'}")

    out = _assembler.embed(syn, ready, newly_verified=received)
    if hasattr(syn, "desired_difficulty"):
        out.desired_difficulty = syn.desired_difficulty

    return out

def _handle_peaked_challenge(syn: ChallengePeakedCircuit) -> ChallengePeakedCircuit:
    """handler for peaked circuits"""
    bt.logging.trace("Handling peaked circuit")
    return _handle_challenge(syn)

def _handle_hstab_challenge(syn: ChallengeHStabCircuit) -> ChallengeHStabCircuit:
    """handler for H-stabilizer circuits"""
    bt.logging.trace("Handling H-stab circuit")
    return _handle_challenge(syn)

# Generic handler that works with any circuit synapse type
def solve_challenge_sync(s: CircuitSynapse, *, wallet) -> CircuitSynapse:
    """Route to appropriate handler"""
    if isinstance(s, ChallengePeakedCircuit):
        return _handle_peaked_challenge(s)
    elif isinstance(s, ChallengeHStabCircuit):
        return _handle_hstab_challenge(s)
    else:
        bt.logging.warning(f"Unknown synapse type {type(s)}, using generic handler")
        return _handle_challenge(s)

handle_peaked_circuit = lambda s, *, wallet: _handle_peaked_challenge(s)
handle_hstab_circuit = lambda s, *, wallet: _handle_hstab_challenge(s)

_solve_challenge_sync = solve_challenge_sync
handle_challenge_circuits = solve_challenge_sync
