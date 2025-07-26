from dataclasses import dataclass
from itertools import starmap
# from multiprocessing import Pool
# POOL = Pool()
from typing import List, Optional, Self, Tuple

import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    QuantumRegister,
    Qubit,
)
from qiskit.compiler import transpile
import qiskit.qasm2 as qasm
from qiskit.quantum_info import random_clifford
from stim import Tableau, PauliString

from qbittensor.validator.hidden_stabilizers_creation.lib.obfuscate.q1 import Rot1Q
from qbittensor.validator.hidden_stabilizers_creation.lib.obfuscate.q2 import RotCnot, RotSwap, Rot2Q

def _pauli(x: bool, z: bool) -> str:
    if not x and not z:
        return "I"
    elif x and not z:
        return "X"
    elif x and z:
        return "Y"
    elif not x and z:
        return "Z"

def _paulistring(
    xi: np.ndarray[bool, 1],
    zi: np.ndarray[bool, 1],
    ri: bool,
) -> PauliString:
    return PauliString(
        ("-" if ri else "+") + "".join(
            _pauli(xij, zij) for (xij, zij) in zip(xi, zi)
        )
    )

def sample_clifford(
    gen: np.random.Generator,
    nqubits: int,
) -> Tuple[QuantumCircuit, List[PauliString]]:
    """
    Sample a circuit implementing a random Clifford group element on the given
    range of qubits.

    Args:
        gen (numpy.random.Generator):
            The PRNG source.
        nqubits (int > 0):
            The number of qubits in the circuit.

    Returns:
        circuit (qiskit.circuit.QuantumCircuit):
            The circuit implementing the Clifford group element.
        stabilizers (List[stim.PauliString]):
            List of canonicalized stabilizers for the Clifford group element.

    Raises:
        ValueError:
            - `nqubits` is less than 1
    """
    if nqubits < 1:
        raise ValueError(f"expected at least 1 qubit, got {nqubits}")
    cliff = random_clifford(nqubits, gen)
    circ = cliff.to_circuit()
    # convert to stim objects -- I would use stim for the whole generation, but
    # for some reason stim.Tableau.random can't be seeded
    paulis = [
        _paulistring(xi, zi, ri)
        for (xi, zi, ri) in zip(cliff.stab_x, cliff.stab_z, cliff.stab_phase)
    ]
    tab = Tableau.from_stabilizers(paulis)
    stabs = tab.to_stabilizers(canonicalize=True)
    return (circ, stabs)

def do_obfuscate(
    gen: np.random.Generator,
    circ: QuantumCircuit,
    instr: CircuitInstruction,
    decompose_q2: bool,
) -> QuantumCircuit:
    """
    Helper for `obfuscate_cliffords`. This performs the actual gate
    decompositions and is pulled out as a top-level function so it can be used
    with `multiprocessing.pool.Pool.starmap`.

    Args:
        gen (numpy.random.Generator):
            RNG source.
        circ (qiskit.circuit.QuantumCircuit):
            Original quantum circuit. Used only for `.find_bit` because
            `Clifford.to_circuit` outputs qubits with UIDs only. Assumed to be
            operations on a single qubit register only.
        instr (qiskit.circuit.CircuitInstruction):
            Circuit instruction to decompose. Assumed to be one of:
                { H, S, Sdg, X, Y, Z, CX, SWAP }
        decompose_q2 (bool):
            If True, perform decompositions for two-qubit gates. Otherwise,
            return a single-gate circuit containing the original gate.

    Returns:
        decomp (qiskit.circuit.QuantumCircuit):
            Circuit containing the decomposition.

    Raises:
        ValueError:
            - The input gate name is not one of the expected set.
    """
    nqubits = circ.num_qubits
    if instr.name in ["h", "s", "sdg", "x", "y", "z"]:
        target = circ.find_bit(instr.qubits[0]).index
        decomp = Rot1Q.sample(gen)
        angles = decomp.compute_params(gen, instr.matrix)
        return decomp.to_circuit(nqubits, angles, target)
    elif instr.name == "cx":
        # assume the first qubit in the instruction is the control because
        # they're not called out specifically otherwise
        control = circ.find_bit(instr.qubits[0]).index
        target = circ.find_bit(instr.qubits[1]).index
        if decompose_q2:
            decomp = RotCnot.sample(gen)
            angles = decomp.compute_params(gen)
            return decomp.to_circuit(nqubits, angles, control, target)
        else:
            ret = QuantumCircuit(nqubits)
            ret.cx(control, target)
            return ret
    elif instr.name == "swap":
        target0 = circ.find_bit(instr.qubits[0]).index
        target1 = circ.find_bit(instr.qubits[1]).index
        if decompose_q2:
            decomp = RotSwap.sample(gen)
            angles = decomp.compute_params(gen)
            return decomp.to_circuit(nqubits, angles, target0, target1)
        else:
            ret = QuantumCircuit(nqubits)
            ret.swap(target0, target1)
            return ret
    else:
        raise ValueError(f"unknown gate type {instr.name}")

def obfuscate_cliffords(
    gen: np.random.Generator,
    cliffords: QuantumCircuit,
    decompose_q2: bool = True,
    mix_boundaries: bool = True,
) -> QuantumCircuit:
    """
    Perform Clifford gate obfuscation, accumulating results in a new
    `QuantumCircuit`.

    Args:
        gen (numpy.random.Generator):
            RNG source.
        cliffords (qiskit.circuit.QuantumCircuit):
            Circuit containing gates to be obfuscated.
        decompose_q2 (bool):
            If True, perform decompositions for two-qubit gates.
        mix_boundaries (bool):
            If True, fuse the last gate of one decomposition with the first gate
            of the next into a single `U3` gate. This makes it harder to detect
            the underlying Clifford gates. Two-qubit gates will not be mixed
            with others in order to keep the eventual QASM output clean.

    Returns:
        circuit (qiskit.circuit.QuantumCircuit):
            The circuit containing the obfuscations.
    """
    # decomposed = POOL.starmap(
    # parallelized version was running into some weird pickling errors;
    # non-parallelized processing should still be fast enough, I think
    decomposed = starmap(
        do_obfuscate,
        ((gen, cliffords, instr, decompose_q2) for instr in cliffords),
    )
    acc = QuantumCircuit(cliffords.num_qubits) # to return
    if mix_boundaries:
        # used for mixing via `transpile`
        xpile = QuantumCircuit(cliffords.num_qubits)
        # used for mixing one- and two-qubit gates -- recall that qiskit has
        # little-endian matrices, so kronecker products need to be reversed
        ident = np.eye(2)
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        for decomp in decomposed:
            if len(xpile) > 0:
                assert len(xpile) == 1, \
                    f"leaking into xpile: len = {len(xpile)}\n{xpile}"
                last_gate = xpile[-1]
                last_inds = [xpile.find_bit(q).index for q in last_gate.qubits]
                next_gate = decomp[0]
                next_inds = [decomp.find_bit(q).index for q in next_gate.qubits]

                # adjacent overlapping one-qubit gates
                if (
                    len(last_inds) == 1 and len(next_inds) == 1
                    and last_inds == next_inds
                ):
                    xpile.append(next_gate)
                    mixed = transpile(xpile)
                    if len(mixed) > 0:
                        acc.append(mixed[0])
                    xpile.clear()
                    skip = 1

                # adjacent overlapping one- (last) and two- (next) gates
                elif (
                    len(last_inds) == 1 and len(next_inds) == 2
                    and last_inds[0] in next_inds
                ):
                    if last_inds[0] == next_inds[0]:
                        last_mat = np.kron(ident, last_gate.matrix)
                    else:
                        last_mat = np.kron(last_gate.matrix, ident)
                    target_mat = next_gate.matrix @ last_mat
                    mixer = Rot2Q.sample(gen)
                    angles = mixer.compute_params(gen, target_mat)
                    mixed = mixer.to_circuit(
                        acc.num_qubits, angles, *next_inds)
                    acc.compose(mixed, inplace=True)
                    xpile.clear()
                    skip = 1

                # adjacent overlapping two- (last) and one- (next) gates
                elif (
                    len(last_inds) == 2 and len(next_inds) == 1
                    and next_inds[0] in last_inds
                ):
                    if next_inds[0] == last_inds[0]:
                        next_mat = np.kron(ident, next_gate.matrix)
                    else:
                        next_mat = np.kron(next_gate.matrix, ident)
                    target_mat = next_mat @ last_gate.matrix
                    mixer = Rot2Q.sample(gen)
                    angles = mixer.compute_params(gen, target_mat)
                    mixed = mixer.to_circuit(
                        acc.num_qubits, angles, *last_inds)
                    acc.compose(mixed, inplace=True)
                    xpile.clear()
                    skip = 1

                # adjacent (completely) overlapping two-qubit gates
                elif (
                    len(last_inds) == 2 and len(next_inds) == 2
                    and (last_inds == next_inds or last_inds == next_inds[::-1])
                ):
                    if last_inds == next_inds:
                        last_mat = last_gate.matrix
                    else:
                        last_mat = swap @ last_gate.matrix @ swap
                    target_mat = next_gate.matrix @ last_mat
                    mixer = Rot2Q.sample(gen)
                    angles = mixer.compute_params(gen, target_mat)
                    mixed = mixer.to_circuit(
                        acc.num_qubits, angles, *next_inds)
                    acc.compose(mixed, inplace=True)
                    xpile.clear()
                    skip = 1

                else:
                    acc.append(last_gate)
                    xpile.clear()
                    skip = 0
            else:
                skip = 0
            for k in range(skip, len(decomp) - 1):
                acc.append(decomp[k])
            xpile.append(decomp[-1])
        if len(xpile) > 0:
            acc.compose(xpile, inplace=True)
    else:
        for decomp in decomposed:
            acc.compose(decomp, inplace=True)
    return acc

def _make_circuit(
    gen: np.random.Generator,
    nqubits: int,
    decompose_q2: bool = True,
    mix_boundaries: bool = True,
) -> Tuple[QuantumCircuit, List[PauliString]]:
    """
    Sample a random Clifford circuit and return a fully obfuscated version of
    the circuit along with the canonicalized stabilizers of the output state.

    Args:
        gen (numpy.random.Generator):
            RNG source.
        nqubits (int > 0):
            Number of qubits.
        decompose_q2 (bool):
            Decompose/obfuscate two-qubit gates.
        mix_boundaries (bool):
            Fuse the first and last gates of consecutive one-qubit gates on the
            same qubit.

    Returns:
        circuit (qiskit.circuit.QuantumCircuit):
            Fully obfuscated circuit.
        stabilizers (List[stim.PauliString]):
            Canonicalized stabilizers of the output state of `circuit`.

    Raises:
        ValueError:
            - `nqubits < 1`
    """
    if nqubits < 1:
        raise ValueError(f"expected at least 1 qubit but got {nqubits}")
    (circ, stabs) = sample_clifford(gen, nqubits)
    obf = obfuscate_cliffords(gen, circ, decompose_q2, mix_boundaries)
    return (obf, stabs)

@dataclass
class HStabCircuit:
    """
    Dataclass to store a hidden stabilizer circuit; generated by
    `HStabCircuit.make_circuit`.

    Fields:
        num_qubits (int):
            Number of qubits.
        circuit (qiskit.circuit.QuantumCircuit):
            Qiskit representation of the circuit.
        stabilizers (List[stim.PauliString]):
            Target list of canonicalized stabilizer generators.
    """
    num_qubits: int
    circuit: QuantumCircuit
    stabilizers: List[PauliString]

    @staticmethod
    def make_circuit(gen: np.random.Generator, num_qubits: int) -> Self:
        """
        Sample a random Clifford circuit and return a new `Self` containing a
        fully obfuscated circuit along with the canonicalized stabilizers of the
        output state.

        Args:
            gen (numpy.random.Generator):
                RNG source.
            num_qubits (int > 0):
                Number of qubits.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                Fully obfuscated circuit.
            stabilizers (List[stim.PauliString]):
                Canonicalized stabilizers of the output state of `circuit`.

        Raises:
            ValueError:
                - `nqubits < 1`
        """
        (circuit, stabilizers) = _make_circuit(
            gen, num_qubits, decompose_q2=True, mix_boundaries=True)
        return HStabCircuit(num_qubits, circuit, stabilizers)

    def to_qasm(self) -> str:
        """
        Convert the circuit to a bare QASM string.

        Returns:
            qasm (str):
                QASM 2.0 representation of `self.circuit`.
        """
        return qasm.dumps(self.circuit)

