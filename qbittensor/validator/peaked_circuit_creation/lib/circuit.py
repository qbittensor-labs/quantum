from __future__ import annotations
from abc import (abstractmethod, abstractstaticmethod)
from dataclasses import dataclass
import multiprocessing
import os
import random
from typing import *
import numpy as np
import bittensor as bt
from .decompose import (optim_decomp, cnots, ising)

class Gate:
    """
    Base class for a single unitary gate operation.
    """
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the gate.
        """
        pass

    @abstractmethod
    def controls(self) -> list[int]:
        """
        Get a list of indices for the qubits to use as controls for the gate.
        """
        pass

    @abstractmethod
    def targets(self) -> list[int]:
        """
        Get a list of indices for the qubits on which the gate acts.
        """
        pass

    @abstractmethod
    def args(self) -> list[float]:
        """
        Get a list of gate parameters (these are most likely rotation angles).
        """
        pass

@dataclass
class Hadamard(Gate):
    """
    A single Hadamard gate.

    Fields:
        target (int >= 0):
            The index of the qubit on which the Hadamard gate acts.
    """
    target: int

    def name(self) -> str:
        return "h"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class S(Gate):
    """
    A single S = sqrt(Z) gate.

    Fields:
        target (int >= 0):
            The index of the qubit on which the S gate acts.
    """
    target: int

    def name(self) -> str:
        return "s"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Sdag(Gate):
    """
    A single S^-1 = conj(S) gate.

    Fields:
        target (int >= 0):
            The index of the qubit on which the Sdag gate acts.
    """
    target: int

    def name(self) -> str:
        return "sdg"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Cnot(Gate):
    """
    A single CNOT gate.

    Fields:
        control (int >= 0):
            The index of the qubit controlling the action of the gate.
        target (int >= 0):
            The index of the qubit on which the gate acts.
    """
    control: int
    target: int

    def name(self) -> str:
        return "cx"

    def controls(self) -> list[int]:
        return [self.control]

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return list()

@dataclass
class Rx(Gate):
    """
    A single rotation about X.

    Fields:
        target (int >= 0):
            The index of the qubit on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "rx"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class Ry(Gate):
    """
    A single rotation about Y.

    Fields:
        target (int >= 0):
            The index of the qubit on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "ry"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class Rz(Gate):
    """
    A single rotation about Z.

    Fields:
        target (int >= 0):
            The index of the qubit on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target: int
    angle: float

    def name(self) -> str:
        return "rz"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle]

@dataclass
class U3(Gate):
    """
    A single single-qubit rotation gate with 3 Euler angles:
        U3(theta, phi, lambda) =
            Rz(phi - pi/2) Rx(pi/2) Rz(pi - theta) Rx(pi/2) Rz(lambda - pi/2)

    Fields:
        target (int >= 0):
            The index of the qubit on which the gate acts.
        angle0 (float):
            The middle Euler angle, theta, in radians.
        angle1 (float):
            The left Euler angle, phi, in radians.
        angle2 (float):
            The right Euler angle, lambda, in radians.
    """
    target: int
    angle0: float
    angle1: float
    angle2: float

    def name(self) -> str:
        return "u3"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target]

    def args(self) -> list[float]:
        return [self.angle0, self.angle1, self.angle2]

    def to_pauli_rots(self) -> list[Gate]:
        return [
            Rz(self.target, self.angle2 - np.pi / 2),
            Rx(self.target, np.pi / 2),
            Rz(self.target, np.pi - self.angle0),
            Rx(self.target, np.pi / 2),
            Rz(self.target, self.angle1 - np.pi / 2),
        ]

@dataclass
class Rxx(Gate):
    """
    A single two-qubit rotation about XX.

    Fields:
        target0 (int >= 0):
            Index of one of the qubits on which the gate acts.
        target1 (int >= 0):
            Index of one of the qubits on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "rxx"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Hadamard(self.target0),
            Hadamard(self.target1),
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
            Hadamard(self.target0),
            Hadamard(self.target1),
        ]

@dataclass
class Ryy(Gate):
    """
    A single two-qubit rotation about YY.

    Fields:
        target0 (int >= 0):
            Index of one of the qubits on which the gate acts.
        target1 (int >= 0):
            Index of one of the qubits on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "ryy"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Sdag(self.target0),
            Hadamard(self.target0),
            S(self.target0),
            Sdag(self.target1),
            Hadamard(self.target1),
            S(self.target1),
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
            Sdag(self.target0),
            Hadamard(self.target0),
            S(self.target0),
            Sdag(self.target1),
            Hadamard(self.target1),
            S(self.target1),
        ]

@dataclass
class Rzz(Gate):
    """
    A single two-qubit rotation about ZZ.

    Fields:
        target0 (int >= 0):
            Index of one of the qubits on which the gate acts.
        target1 (int >= 0):
            Index of one of the qubits on which the gate acts.
        angle (float):
            The rotation angle of the gate, in radians.
    """
    target0: int
    target1: int
    angle: float

    def name(self) -> str:
        return "rzz"

    def controls(self) -> list[int]:
        return list()

    def targets(self) -> list[int]:
        return [self.target0, self.target1]

    def args(self) -> list[float]:
        return [self.angle]

    def to_cnots(self) -> list[Gate]:
        return [
            Cnot(self.target0, self.target1),
            Rz(self.target1, self.angle),
            Cnot(self.target0, self.target1),
        ]

class SU4Decomp:
    """
    Base class for decompositions of a general SU(4) (i.e. two-qubit) unitary
    gate.
    """
    @abstractstaticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        """
        Construct a decomposition from a particular two-qubit unitary.

        Args:
            uni (numpy.ndarray[complex, 2]):
                Two-qubit unitary as a complex-valued 4x4 matrix.

        Returns:
            decomp (Self):
                The decomposition of `uni` into some number of ordinary gates.
        """
        pass

    @abstractmethod
    def to_uni(self) -> np.ndarray[complex, 2]:
        """
        Convert `self` back into a two-qubit unitary, should be a complex-valued
        4x4 matrix.

        Returns:
            uni (numpy.ndarray[complex, 2]):
                `self` as a complex-valued 4x4 numpy array.
        """
        pass

    @abstractmethod
    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        """
        Convert `self` into a list of `Gate`s, provided target qubits.

        Args:
            target0 (int >= 0):
                Index of the leftmost qubit.
            target1 (int >= 0):
                Index of the rightmost qubit.

        Returns:
            gates (list[Gate]):
                List of ordinary gates operating on `target0` and `target1`.
        """
        pass

class IsingDecomp(SU4Decomp):
    """
    An "Ising-like" decomposition of a general SU(4) gate according to [this
    form][pennylane]:
        U =
            U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
            R_XX([eta0]) R_YY([eta1]) R_ZZ([eta2])
            U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])

    Fields:
        params (numpy.ndarray[float, 1]):
            List of rotation angles, in radians. Angles are ordered as
                [
                  alpha0, .., alpha2,
                  beta0, .., beta2,
                  eta0, .., eta2,
                  gamma0, .., gamma2,
                  delta0, .., delta2
                ]

    [pennylane]: https://pennylane.ai/qml/demos/tutorial_kak_decomposition#kokcu-fdhs
    """
    params: np.ndarray[float, 1]

    def __init__(self, params: np.ndarray[float, 1]):
        if params.shape != (15,):
            raise ValueError("requires 15 params")
        self.params = params

    @staticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        # params = do_grad_ascent(
        #     U_target=uni,
        #     obj=ising.fidelity,
        #     grad=ising.fidelity_grad,
        #     init_learning_param=1e-5,
        #     maxiters=10000,
        #     epsilon=1e-9,
        # )
        params = optim_decomp(
            U_target=uni,
            fidelity=ising.fidelity,
        )
        return IsingDecomp(params)

    def to_uni(self) -> np.ndarray[complex, 2]:
        return ising.make_uni(self.params)

    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        return [
            U3(target0, *self.params[9:12]),
            U3(target1, *self.params[12:15]),
            Rxx(target0, target1, self.params[6]),
            Ryy(target0, target1, self.params[7]),
            Rzz(target0, target1, self.params[8]),
            U3(target0, *self.params[0:3]),
            U3(target1, *self.params[3:6]),
        ]

class CnotsDecomp(SU4Decomp):
    """
    A "CNOT-based" decomposition of a general SU(4) gate according to [this
    form][cnot-based]:
        U =
            U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
            CNOT
            Rx(0, eta0) Rz(1, eta1)
            CNOT
            U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])
            CNOT
            Rz(1, eta2)

    Fields:
        params (numpy.ndarray[float, 1]):
            List of rotation angles, in radians. Angles are ordered as
                [
                  alpha0, .., alpha2,
                  beta0, .., beta2,
                  eta0, .., eta2,
                  gamma0, .., gamma2,
                  delta0, .., delta2
                ]

    [cnot-based]: https://arxiv.org/abs/quant-ph/0308033
    """
    params: np.ndarray[float]

    def __init__(self, params: np.ndarray[float, 1]):
        if params.shape != (15,):
            raise ValueError("requires 15 params")
        self.params = params

    @staticmethod
    def from_uni(uni: np.ndarray[complex, 2]):
        params = optim_decomp(
            U_target=uni,
            fidelity=cnots.fidelity,
        )
        return CnotsDecomp(params)

    def to_uni(self) -> np.ndarray[complex, 2]:
        return cnots.make_uni(self.params)

    def to_gates(self, target0: int, target1: int) -> list[Gate]:
        return [
            Rz(target1, self.params[8]),
            Cnot(target0, target1),
            U3(target0, *self.params[9:12]),
            U3(target1, *self.params[12:15]),
            Cnot(target0, target1),
            Rx(target0, self.params[6]),
            Rz(target1, self.params[7]),
            Cnot(target0, target1),
            U3(target0, *self.params[0:3]),
            U3(target1, *self.params[3:6]),
        ]

@dataclass
class SU4:
    """
    A two-qubit unitary matrix with associated target qubit indices.
    """
    target0: int
    target1: int
    mat: np.ndarray[complex, 2]

def _cnots_decomp(uni: SU4) -> list[Gate]:
    return CnotsDecomp.from_uni(uni.mat).to_gates(uni.target0, uni.target1)

def _ising_decomp(uni: SU4) -> list[Gate]:
    return IsingDecomp.from_uni(uni.mat).to_gates(uni.target0, uni.target1)

@dataclass
class PeakedCircuit:
    """
    A generated circuit containing only unitary operations, producing a
    computational basis state with modal probability for an assumed all-zero
    initial state.

    Fields:
        seed (int):
            The seed value used to generate the circuit.
        gen (numpy.random.Generator):
            Internal RNG created from `seed`.
        num_qubits (int):
            The number of qubits in the circuit.
        gates (list[Gate]):
            A list of gate operations.
        target_state (str):
            The target (i.e. solution) basis state as a string of all '0's and
            '1's. The length of this string should equal `num_qubits`.
        peak_prob (float >= 0):
            The probability of the target state.
    """
    seed: int
    gen: np.random.Generator
    num_qubits: int
    gates: list[Gate]
    target_state: str # all 0's and 1's
    peak_prob: float

    @staticmethod
    def from_su4_series(
        target_state: str,
        peak_prob: float,
        unis: list[SU4],
        seed: int,
        pool: multiprocessing.pool.Pool | None = None,
    ):
        """
        Construct from a list of two-qubit unitaries, decomposing into ordinary
        gates as needed.

        Args:
            target_state (str):
                Target peaked state. Should be all '0's and '1's.
            peak_prob (float):
                Output probability of the peaked state.
            unis (list[SU4]):
                List of two-qubit unitaries.
            seed (int):
                Seed value used to originally generate `unis`.

        Returns:
            circuit (Self):
                The input circuit as a list of gates.
        """
        gen = np.random.Generator(np.random.PCG64(seed))
        # cnot or ising decomp based on seed
        choice = hash(str(seed)) % 2
        decomp_method = "CNOT" if choice == 0 else "Ising"
        bt.logging.info(f"convert to ordinary gates (using {decomp_method} decomposition):")

        # this assumes every qubit is touched
        num_qubits = max(max(uni.target0, uni.target1) for uni in unis) + 1

        def _pool_initializer():
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
            os.environ.setdefault('MKL_NUM_THREADS', '1')

        def _map_with_pool(func, items):
            if pool is not None:
                return pool.map(func, items)
            with multiprocessing.Pool(initializer=_pool_initializer) as local_pool:
                return local_pool.map(func, items)

        if choice == 0:
            mapped = _map_with_pool(_cnots_decomp, unis)
        else:
            mapped = _map_with_pool(_ising_decomp, unis)
        gates = [gate for subcirc in mapped for gate in subcirc]

        return PeakedCircuit(
            seed,
            gen,
            num_qubits,
            gates,
            target_state,
            peak_prob,
        )

    # really dumb rendering because the circuits are pretty simple
    def to_qasm(self) -> str:
        """
        Render to a bare string giving the OpenQASM (2.0) circuit, with final
        measurements attached.

        Returns:
            qasm (str):
                Output QASM circuit description as a single string.
        """
        acc = (
f"""
OPENQASM 2.0;
include "qelib1.inc";

qreg q[{self.num_qubits}];
creg c[{self.num_qubits}];

"""
        )
        for gate in self.gates:
            # TODO: maybe change this for more randomness
            if isinstance(gate, (Rxx, Ryy, Rzz)):
                decomp = gate.to_cnots()
                for subgate in decomp:
                    acc += _write_gate(subgate)
            # Keep U3 gates as-is (no random decomposition)
            else:
                acc += _write_gate(gate)
                
        acc += f"measure q -> c;\n"
        return acc

def _write_gate(gate: Gate) -> str:
    acc = gate.name()
    if (n := len(args := gate.args())) > 0:
        acc += "("
        acc += ",".join(f"{arg}" for arg in args)
        acc += ")"
    operands = gate.controls() + gate.targets()
    n = len(operands)
    assert n > 0, "unexpected gate with no operands"
    acc += " "
    acc += ",".join(f"q[{op}]" for op in operands)
    acc += ";\n"
    return acc

