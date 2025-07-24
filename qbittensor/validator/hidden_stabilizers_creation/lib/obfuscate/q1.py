from enum import IntEnum
from functools import reduce
from typing import Self, List

import numpy as np
from numpy import exp, sin, cos, conjugate
from qiskit.circuit import QuantumCircuit

from . import Decomp

def rx(angle: float) -> np.ndarray[complex, 2]:
    """
    Compute `exp(-i X angle / 2)`.
    """
    return np.array([
        [ cos(angle / 2), -1j * sin(angle / 2) ],
        [ -1j * sin(angle / 2), cos(angle / 2) ],
    ])

def ry(angle: float) -> np.ndarray[complex, 2]:
    """
    Compute `exp(-i Y angle / 2)`.
    """
    return np.array([
        [ cos(angle / 2), -sin(angle / 2) ],
        [ sin(angle / 2),  cos(angle / 2) ],
    ])

def rz(angle: float) -> np.ndarray[complex, 2]:
    """
    Compute `exp(-i Z angle / 2)`.
    """
    return np.diag([exp(-1j * angle / 2), exp(1j * angle / 2)])

class RotGate(IntEnum):
    """
    A single-qubit rotation axis.
    """
    X = 1
    Y = 2
    Z = 3

    @staticmethod
    def sample(gen: np.random.Generator) -> Self:
        """
        Sample a random rotation axis with uniform probability.

        Args:
            gen (numpy.random.Generator):
                RNG source.

        Returns:
            axis (Self):
                Sampled rotation axis.
        """
        r = gen.random()
        if r < 0.333:
            return RotGate.X
        elif r < 0.667:
            return RotGate.Y
        else:
            return RotGate.Z

    def sample_next(self, gen: np.random.Generator) -> Self:
        """
        Sample a rotation axis not equal to `self` with uniform probability.

        Args:
            gen (numpy.random.Generator):
                RNG source.

        Returns:
            axis (Self):
                Sampled rotation axis.
        """
        r = gen.random()
        b = r < 0.5
        if self == RotGate.X:
            return RotGate.Y if b else RotGate.Z
        elif self == RotGate.Y:
            return RotGate.X if b else RotGate.Z
        else:
            return RotGate.X if b else RotGate.Y

    def matrix(self, angle: float) -> np.ndarray[complex, 2]:
        """
        Compute the rotation matrix for a given angle.

        Args:
            angle (float):
                Rotation angle in radians.

        Returns:
            matrix (numpy.ndarray[complex, 2]):
                Rotation matrix.
        """
        if self == RotGate.X:
            return rx(angle)
        elif self == RotGate.Y:
            return ry(angle)
        else:
            return rz(angle)

class Rot1Q(Decomp):
    """
    Randomized obfuscation scheme for single-qubit gates. Under this scheme, a
    single-qubit gate (e.g. S or H) is "decomposed" as a series of X, Y, or Z
    rotation gates. Both the length and composition of this series is randomly
    determined via `Rot1Q.sample`.

    Fields:
        rots (List[RotGate]):
            The target series of X, Y, or Z rotation gates for decomposition.
    """
    rots: List[RotGate]

    def __init__(self, rots: List[RotGate]):
        self.rots = rots

    def sample(gen: np.random.Generator) -> Self:
        decomp_len = int(4 + 3 * gen.random())
        rots = [RotGate.sample(gen)]
        rot = rots[0]
        for _ in range(decomp_len - 1):
            rot = rot.sample_next(gen)
            rots.append(rot)
        return Rot1Q(rots)

    def num_params(self) -> int:
        return len(self.rots)

    def fidelity(
        self,
        target: np.ndarray[complex, 2],
        params: np.ndarray[float, 1],
    ) -> float:
        uni = self.make_uni(params)
        return abs(np.diag(uni.T.conjugate() @ target).sum()) ** 2 / 4

    def make_uni(self, params: np.ndarray[float, 1]) -> np.ndarray[complex, 2]:
        """
        Compute the matrix for a given set of parameters.

        Args:
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition scheme.

        Returns:
            uni (numpy.ndarray[complex, 2]):
                Matrix for the parameterization.

        Raises:
            ValueError:
                - `len(params)` is not equal to `self.num_params()`
        """
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        return reduce(
            (lambda acc, gate: gate[0].matrix(gate[1]) @ acc),
            zip(self.rots, params),
            np.eye(2),
        )

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        target: int,
    ) -> QuantumCircuit:
        """
        Convert `self` to a circuit object on `nqubits` qubits with gates
        applied to the `target`-th qubit.

        Args:
            nqubits (int > 0):
                The number of qubits in the circuit.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition.
            target (int >= 0, < `nqubits`):
                The qubit in the circuit to which gates will be applied.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `target` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        if nqubits < 1:
            raise ValueError(f"expected at least 1 qubit, got {nqubits}")
        if target < 0 or target >= nqubits:
            raise ValueError(
                f"invalid target qubit {target} for {nqubits} qubits")
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        circ = QuantumCircuit(nqubits)
        for (rot, param) in zip(self.rots, params):
            if rot == RotGate.X:
                circ.rx(param, target)
            elif rot == RotGate.Y:
                circ.ry(param, target)
            else:
                circ.rz(param, target)
        return circ

