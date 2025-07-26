from typing import Self, List, Optional

import numpy as np
from numpy import conjugate, cos, sin, exp
from qiskit.circuit import QuantumCircuit

from .q1 import Rot1Q, RotGate
from . import optim_decomp, Decomp

class RotCnot:
    """
    Randomized obfuscation scheme for CNOT gates. Under this scheme, a CNOT gate
    is "decomposed" as a series of controlled X, Y, or Z gates. Both the length
    and composition of this series is randomly determined via `RotCnot.sample`.

    This class does *not* adhere to `obfuscate.Decomp` because it is only built
    to decompose CNOT gates.

    Fields:
        decomp (q1.Rot1Q):
            The decomposition scheme for the (controlled) X operation.
        z_insert (int):
            Index for where to insert a phase-correction Rz gate on the control
            qubit. This commutes with all two-qubit gates, so we can insert it
            anywhere within the block.
    """
    decomp: Rot1Q
    z_insert: int

    def __init__(self, decomp: Rot1Q, z_insert: int):
        self.decomp = decomp
        self.z_insert = z_insert

    def sample(gen: np.random.Generator) -> Self:
        """
        Sample a random decomposition scheme.

        Args:
            gen (numpy.random.Generator):
                The RNG source.

        Returns:
            decomp (Self):
                The decomposition scheme.
        """
        decomp = Rot1Q.sample(gen)
        # avoid placing z_insert at the ends of the block so that two-qubit gate
        # mixing will be more effective
        z_insert = int(1 + (decomp.num_params() - 2) * gen.random())
        return RotCnot(decomp, z_insert)

    def num_params(self) -> int:
        """
        Number of parameters for the decomposition scheme.

        Returns:
            num_params (int > 0):
                Number of parameters.
        """
        return self.decomp.num_params()

    def compute_params(
        self,
        gen: np.random.Generator,
        epsilon: Optional[float] = 1e-6,
    ) -> np.ndarray[float, 1]:
        """
        Compute decomposition parameters using `self.fidelity` and
        `optim_decomp`.

        Args:
            gen (numpy.random.Generator):
                The RNG source.
            epsilon (Optional[float]):
                Tolerance value for optimization.

        Returns:
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition scheme.
        """
        xmatrix = np.array([[0, 1], [1, 0]], dtype=complex)
        return optim_decomp(gen, self.decomp, xmatrix, epsilon)

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        control: int,
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
            control (int >= 0, < `nqubits`):
                The control qubit in the circuit.
            target (int >= 0, < `nqubits`):
                The qubit in the circuit to which gates will be applied.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `control` is negative, or greater than or equal to `nqubits`
                - `target` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        if nqubits < 1:
            raise ValueError(f"expected at least 1 qubit, got {nqubits}")
        if control < 0 or control >= nqubits:
            raise ValueError(
                f"invalid control qubit {control} for {nqubits} qubits")
        if target < 0 or target >= nqubits:
            raise ValueError(
                f"invalid target qubit {target} for {nqubits} qubits")
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        # overall scalars in the single-qubit X decomposition become local when
        # we add in controls; need to compute the single-qubit matrix and divide
        # out the scalar via an RZ(angle) on the control qubit
        mat = self.decomp.make_uni(params)
        z = 2 / (mat[0, 1] + mat[1, 0])
        angle = np.arctan2(z.imag, z.real)
        circ = QuantumCircuit(nqubits)
        for (k, (rot, param)) in enumerate(zip(self.decomp.rots, params)):
            if k == self.z_insert:
                circ.rz(angle, control)
            if rot == RotGate.X:
                circ.crx(param, control, target)
            elif rot == RotGate.Y:
                circ.cry(param, control, target)
            else:
                circ.crz(param, control, target)
        return circ

class RotSwap:
    """
    Randomized obfuscation scheme for SWAP gates. Under this scheme, a SWAP gate
    is "decomposed" as a series of controlled X, Y, or Z gates using three
    separate CNOT decompositions via `RotCnot`. The lengths and compositions of
    each CNOT decomposition is determined independently.

    This class does *not* adhere to `obfuscate.Decomp` because it is only built
    to decompose SWAP gates.

    Fields:
        cnot0 (RotCnot):
            Decomposition scheme for the first CNOT.
        cnot1 (RotCnot):
            Decomposition scheme for the second CNOT.
        cnot2 (RotCnot):
            Decomposition scheme for the third CNOT.
    """
    cnot0: RotCnot
    cnot1: RotCnot
    cnot2: RotCnot

    def __init__(self, cnot0: RotCnot, cnot1: RotCnot, cnot2: RotCnot):
        self.cnot0 = cnot0
        self.cnot1 = cnot1
        self.cnot2 = cnot2

    def sample(gen: np.random.Generator) -> Self:
        """
        Sample a random decomposition scheme.

        Args:
            gen (numpy.random.Generator):
                The RNG source.

        Returns:
            decomp (Self):
                The decomposition scheme.
        """
        cnot0 = RotCnot.sample(gen)
        cnot1 = RotCnot.sample(gen)
        cnot2 = RotCnot.sample(gen)
        return RotSwap(cnot0, cnot1, cnot2)

    def num_params(self) -> int:
        """
        Number of parameters for the decomposition scheme.

        Returns:
            num_params (int > 0):
                Number of parameters.
        """
        return (
            self.cnot0.num_params()
            + self.cnot1.num_params()
            + self.cnot2.num_params()
        )

    def compute_params(
        self,
        gen: np.random.Generator,
        epsilon: Optional[float] = 1e-6,
    ) -> np.ndarray[float, 1]:
        """
        Compute decomposition parameters using `self.fidelity` and
        `optim_decomp`.

        Args:
            gen (numpy.random.Generator):
                The RNG source.
            epsilon (Optional[float]):
                Tolerance value for optimization.

        Returns:
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition scheme.
        """
        params0 = self.cnot0.compute_params(gen, epsilon)
        params1 = self.cnot1.compute_params(gen, epsilon)
        params2 = self.cnot2.compute_params(gen, epsilon)
        return np.array([*params0, *params1, *params2])

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        target0: int,
        target1: int,
    ) -> QuantumCircuit:
        """
        Convert `self` to a circuit object on `nqubits` qubits with gates
        applied to the `target`-th qubit.

        Args:
            nqubits (int > 0):
                The number of qubits in the circuit.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition.
            target0 (int >= 0, < `nqubits`):
                The first target qubit in the circuit.
            target1 (int >= 0, < `nqubits`):
                The second target qubit in the circuit.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `target0` is negative, or greater than or equal to `nqubits`
                - `target1` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        if nqubits < 1:
            raise ValueError(f"expected at least 1 qubit, got {nqubits}")
        if target0 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid qubit {target0} for {nqubits} qubits")
        if target0 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid qubit {target1} for {nqubits} qubits")
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        n0 = self.cnot0.num_params()
        n1 = self.cnot1.num_params()
        n2 = self.cnot2.num_params()
        params0 = params[0 : n0]
        circ0 = self.cnot0.to_circuit(nqubits, params0, target0, target1)
        params1 = params[n0 : n0 + n1]
        circ1 = self.cnot1.to_circuit(nqubits, params1, target1, target0)
        params2 = params[n0 + n1 : n0 + n1 + n2]
        circ2 = self.cnot2.to_circuit(nqubits, params2, target0, target1)
        circ0.compose(circ1, inplace=True)
        circ0.compose(circ2, inplace=True)
        return circ0

### reimplementation of `cnots` and `ising` decompositions from peaked circuits;
### needs to be reimplemented because we're working with matrices generated by
### qiskit, which are little-endian and the ones in peaked circuits are
### big-endian

def u3u3(
    alpha0: float,
    alpha1: float,
    alpha2: float,
    beta0: float,
    beta1: float,
    beta2: float,
) -> np.ndarray[complex, 2]:
    """
    Return the Kronecker product of two independent U3 gates, with `alpha*`
    corresponding to the left qubit and `beta` to the right.
    """
    cos_a0 = cos(alpha0 / 2)
    sin_a0 = sin(alpha0 / 2)
    cos_b0 = cos(beta0 / 2)
    sin_b0 = sin(beta0 / 2)
    ei_a1 = exp(1j * alpha1)
    ei_a2 = exp(1j * alpha2)
    ei_b1 = exp(1j * beta1)
    ei_b2 = exp(1j * beta2)
    ei_a1_a2 = ei_a1 * ei_a2
    ei_a1_b1 = ei_a1 * ei_b1
    ei_a1_b2 = ei_a1 * ei_b2
    ei_a2_b1 = ei_a2 * ei_b1
    ei_a2_b2 = ei_a2 * ei_b2
    ei_b1_b2 = ei_b1 * ei_b2
    ei_a1_a2_b1 = ei_a1_a2 * ei_b1
    ei_a1_a2_b2 = ei_a1_a2 * ei_b2
    ei_a1_b1_b2 = ei_a1 * ei_b1_b2
    ei_a2_b1_b2 = ei_a2 * ei_b1_b2
    ei_a1_a2_b1_b2 = ei_a1_a2 * ei_b1_b2
    return np.array(
        [
            [
                cos_a0 * cos_b0,
                -ei_a2 * sin_a0 * cos_b0,
                -ei_b2 * cos_a0 * sin_b0,
                ei_a2_b2 * sin_a0 * sin_b0,
            ],
            [
                ei_a1 * sin_a0 * cos_b0,
                ei_a1_a2 * cos_a0 * cos_b0,
                -ei_a1_b2 * sin_a0 * sin_b0,
                -ei_a1_a2_b2 * cos_a0 * sin_b0,
            ],
            [
                ei_b1 * cos_a0 * sin_b0,
                -ei_a2_b1 * sin_a0 * sin_b0,
                ei_b1_b2 * cos_a0 * cos_b0,
                -ei_a2_b1_b2 * sin_a0 * cos_b0,
            ],
            [
                ei_a1_b1 * sin_a0 * sin_b0,
                ei_a1_a2_b1 * cos_a0 * sin_b0,
                ei_a1_b1_b2 * sin_a0 * cos_b0,
                ei_a1_a2_b1_b2 * cos_a0 * cos_b0,
            ],
        ]
    )


def cnot_rxrz_cnot(eta0: float, eta1: float) -> np.ndarray[complex, 2]:
    """
    Return the CNOT-conjugation of the Kronecker product of two independent RX
    and RZ gates, with `eta0` corresponding to the RX gate on the left qubit and
    `eta1` to the RZ gate on the right.
    """
    cos_h0 = cos(eta0 / 2)
    sin_h0 = sin(eta0 / 2)
    ei_ph1 = exp(1j * eta1 / 2)
    ei_mh1 = conjugate(ei_ph1)
    return np.array(
        [
            [ ei_mh1 * cos_h0, 0, 0, -1j * ei_mh1 * sin_h0 ],
            [ 0, ei_ph1 * cos_h0, -1j * ei_ph1 * sin_h0, 0 ],
            [ 0, -1j * ei_ph1 * sin_h0, ei_ph1 * cos_h0, 0 ],
            [ -1j * ei_mh1 * sin_h0, 0, 0, ei_mh1 * cos_h0 ],
        ]
    )


def u3u3_cnot_irz(
    gamma0: float,
    gamma1: float,
    gamma2: float,
    delta0: float,
    delta1: float,
    delta2: float,
    eta2: float,
) -> np.ndarray[complex, 2]:
    """
    Return the product: (U3 tensor U3) CNOT (id tensor RZ), with `gamma*` and
    `delta*` the angles of the two U3 gates, and `eta2` the angle of the RZ
    gate.
    """
    cos_g0 = cos(gamma0 / 2)
    sin_g0 = sin(gamma0 / 2)
    cos_d0 = cos(delta0 / 2)
    sin_d0 = sin(delta0 / 2)
    ei_g1 = exp(1j * gamma1)
    ei_g2 = exp(1j * gamma2)
    ei_d1 = exp(1j * delta1)
    ei_d2 = exp(1j * delta2)
    ei_g1_g2 = ei_g1 * ei_g2
    ei_g1_d1 = ei_g1 * ei_d1
    ei_g1_d2 = ei_g1 * ei_d2
    ei_g2_d1 = ei_g2 * ei_d1
    ei_g2_d2 = ei_g2 * ei_d2
    ei_d1_d2 = ei_d1 * ei_d2
    ei_g1_g2_d1 = ei_g1_g2 * ei_d1
    ei_g1_g2_d2 = ei_g1_g2 * ei_d2
    ei_g1_d1_d2 = ei_g1 * ei_d1_d2
    ei_g2_d1_d2 = ei_g2 * ei_d1_d2
    ei_g1_g2_d1_d2 = ei_g1_g2 * ei_d1_d2
    ei_ph2 = exp(1j * eta2 / 2)
    ei_mh2 = conjugate(ei_ph2)
    return np.array(
        [
            [
                ei_mh2 * cos_g0 * cos_d0,
                ei_mh2 * ei_g2_d2 * sin_g0 * sin_d0,
                -ei_ph2 * ei_d2 * cos_g0 * sin_d0,
                -ei_ph2 * ei_g2 * sin_g0 * cos_d0,
            ],
            [
                ei_mh2 * ei_g1 * sin_g0 * cos_d0,
                -ei_mh2 * ei_g1_g2_d2 * cos_g0 * sin_d0,
                -ei_ph2 * ei_g1_d2 * sin_g0 * sin_d0,
                ei_ph2 * ei_g1_g2 * cos_g0 * cos_d0,
            ],
            [
                ei_mh2 * ei_d1 * cos_g0 * sin_d0,
                -ei_mh2 * ei_g2_d1_d2 * sin_g0 * cos_d0,
                ei_ph2 * ei_d1_d2 * cos_g0 * cos_d0,
                -ei_ph2 * ei_g2_d1 * sin_g0 * sin_d0,
            ],
            [
                ei_mh2 * ei_g1_d1 * sin_g0 * sin_d0,
                ei_mh2 * ei_g1_g2_d1_d2 * cos_g0 * cos_d0,
                ei_ph2 * ei_g1_d1_d2 * sin_g0 * cos_d0,
                ei_ph2 * ei_g1_g2_d1 * cos_g0 * sin_d0,
            ],
        ]
    )

def rxx_ryy_rzz(
    eta0: float,
    eta1: float,
    eta2: float,
) -> np.ndarray[complex, 2]:
    """
    Return the product of independent RXX, RYY, and RZZ gates, with angles
    `eta0`, `eta1`, and `eta2`, respectively.
    """
    cos_hdiff = cos((eta0 - eta1) / 2)
    sin_hdiff = sin((eta0 - eta1) / 2)
    cos_hsum = cos((eta0 + eta1) / 2)
    sin_hsum = sin((eta0 + eta1) / 2)
    ei_ph2 = exp(1j * eta2 / 2)
    ei_mh2 = conjugate(ei_ph2)
    return np.array(
        [
            [ ei_mh2 * cos_hdiff, 0, 0, -1j * ei_mh2 * sin_hdiff ],
            [ 0, ei_ph2 * cos_hsum, -1j * ei_ph2 * sin_hsum, 0 ],
            [ 0, -1j * ei_ph2 * sin_hsum, ei_ph2 * cos_hsum, 0 ],
            [ -1j * ei_mh2 * sin_hdiff, 0, 0, ei_mh2 * cos_hdiff ],
        ]
    )

class Cnots2Q(Decomp):
    """
    "CNOT-based" decomposition of a general SU(4) gate according to [this
    form][cnot-based]:
        U =
            U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
            CNOT
            Rx(0, eta0) Rz(1, eta1)
            CNOT
            U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])
            CNOT
            Rz(1, eta2)

    [cnot-based]: https://arxiv.org/abs/quant-ph/0308033
    """

    @staticmethod
    def sample(_gen: np.random.Generator) -> Self:
        """
        Construct a CNOT-based decomposition scheme. This is entirely
        deterministic.
        """
        return Cnots2Q()

    def num_params(self) -> int:
        """
        Fixed to 15.
        """
        return 15

    def _make_uni(self, params: np.ndarray[float, 1]) -> np.ndarray[complex, 2]:
        [alpha0, alpha1, alpha2, beta0, beta1, beta2, eta0, eta1, eta2, gamma0, gamma1, gamma2, delta0, delta1, delta2] = (
            params
        )
        return (
            u3u3(alpha0, alpha1, alpha2, beta0, beta1, beta2)
            @ cnot_rxrz_cnot(eta0, eta1)
            @ u3u3_cnot_irz(gamma0, gamma1, gamma2, delta0, delta1, delta2, eta2)
        )

    def fidelity(
        self,
        target: np.ndarray[complex, 2],
        params: np.ndarray[float, 1],
    ) -> float:
        uni = self._make_uni(params)
        return abs(np.diag(uni.T.conjugate() @ target).sum()) ** 2 / 16

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        target0: int,
        target1: int,
    ) -> QuantumCircuit:
        """
        Convert `self` to a circuit object on `nqubits` qubits with gates
        applied to the `target0` and `target1`-th qubits.

        Args:
            nqubits (int > 0):
                The number of qubits in the circuit.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition.
            target0 (int >= 0, < `nqubits`):
                First qubit in the circuit to which gates will be applied.
            target1 (int >= 0, < `nqubits`):
                Second qubit in the circuit to which gates will be applied.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `target0` is negative, or greater than or equal to `nqubits`
                - `target1` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        if nqubits < 1:
            raise ValueError("expected at least 1 qubit, got {nqubits}")
        if target0 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid target qubit {target0} for {nqubits} qubits")
        if target1 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid target qubit {target1} for {nqubits} qubits")
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        [alpha0, alpha1, alpha2, beta0, beta1, beta2, eta0, eta1, eta2, gamma0, gamma1, gamma2, delta0, delta1, delta2] = (
            params
        )
        circ = QuantumCircuit(nqubits)
        circ.rz(eta2, target1)
        circ.cx(target0, target1)
        circ.u(gamma0, gamma1, gamma2, target0)
        circ.u(delta0, delta1, delta2, target1)
        circ.cx(target0, target1)
        circ.rx(eta0, target0)
        circ.rz(eta1, target1)
        circ.cx(target0, target1)
        circ.u(alpha0, alpha1, alpha2, target0)
        circ.u(beta0, beta1, beta2, target1)
        return circ

class Ising2Q(Decomp):
    """
    Functions in this module compute quantities relevant to an "Ising-like"
    decomposition of a general SU(4) gate according to [this form][pennylane]:
        U =
            U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
            R_XX([eta0]) R_YY([eta1]) R_ZZ([eta2])
            U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])

    [pennylane]: https://pennylane.ai/qml/demos/tutorial_kak_decomposition#kokcu-fdhs
    """

    @staticmethod
    def sample(_gen: np.random.Generator) -> Self:
        """
        Construct an "Ising-like" decomposition scheme. This is entirely
        deterministic.
        """
        return Cnots2Q()

    def num_params(self) -> int:
        """
        Fixed to 15.
        """
        return 15

    def _make_uni(self, params: np.ndarray[float, 1]) -> np.ndarray[complex, 2]:
        [alpha0, alpha1, alpha2, beta0, beta1, beta2, eta0, eta1, eta2, gamma0, gamma1, gamma2, delta0, delta1, delta2] = (
            params
        )
        return (
            u3u3(alpha0, alpha1, alpha2, beta0, beta1, beta2)
            @ rxx_ryy_rzz(eta0, eta1, eta2)
            @ u3u3(gamma0, gamma1, gamma2, delta0, delta1, delta2)
        )

    def fidelity(
        self,
        target: np.ndarray[complex, 2],
        params: np.ndarray[float, 1],
    ) -> float:
        uni = self._make_uni(params)
        return abs(np.diag(uni.T.conjugate() @ target).sum()) ** 2 / 16

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        target0: int,
        target1: int,
    ) -> QuantumCircuit:
        """
        Convert `self` to a circuit object on `nqubits` qubits with gates
        applied to the `target0` and `target1`-th qubits.

        Args:
            nqubits (int > 0):
                The number of qubits in the circuit.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition.
            target0 (int >= 0, < `nqubits`):
                First qubit in the circuit to which gates will be applied.
            target1 (int >= 0, < `nqubits`):
                Second qubit in the circuit to which gates will be applied.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `target0` is negative, or greater than or equal to `nqubits`
                - `target1` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        if nqubits < 1:
            raise ValueError("expected at least 1 qubit, got {nqubits}")
        if target0 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid target qubit {target0} for {nqubits} qubits")
        if target1 < 0 or target1 >= nqubits:
            raise ValueError(
                f"invalid target qubit {target1} for {nqubits} qubits")
        if len(params) != self.num_params():
            raise ValueError(
                f"expected {self.num_params()} params, got {len(params)}")
        [alpha0, alpha1, alpha2, beta0, beta1, beta2, eta0, eta1, eta2, gamma0, gamma1, gamma2, delta0, delta1, delta2] = (
            params
        )
        circ = QuantumCircuit(nqubits)
        circ.u(gamma0, gamma1, gamma2, target0)
        circ.u(delta0, delta1, delta2, target1)
        circ.rzz(eta2, target0, target1)
        circ.ryy(eta1, target0, target1)
        circ.rxx(eta0, target0, target1)
        circ.u(alpha0, alpha1, alpha2, target0)
        circ.u(beta0, beta1, beta2, target1)
        return circ

class Rot2Q(Decomp):
    """
    Decomposition based on either `Cnots2Q` or `Ising2Q`. Although we already
    have `RotCnot` and `RotSwap`, we still want to use this for boundary mixing
    between overlapping adjacent one-and-two-qubit or two-and-two-qubit gates.

    Fields:
        decomp (Union[Cnots2Q, Ising2Q]):
            Either a `Cnots2Q` or `Ising2Q` decomposition, chosen via
            `self.sample` with equal probability.
    """
    decomp: Cnots2Q | Ising2Q

    def __init__(self, decomp: Cnots2Q | Ising2Q):
        self.decomp = decomp

    @staticmethod
    def sample(gen: np.random.Generator) -> Self:
        """
        Choose between a "CNOT-based" or "Ising-like" decomposition scheme with
        equal probability.
        """
        return Rot2Q(Cnots2Q() if gen.random() < 0.5 else Ising2Q())

    def num_params(self) -> int:
        """
        Fixed to 15.
        """
        return self.decomp.num_params()

    def fidelity(
        self,
        target: np.ndarray[complex, 2],
        params: np.ndarray[float, 1],
    ) -> float:
        return self.decomp.fidelity(target, params)

    def to_circuit(
        self,
        nqubits: int,
        params: np.ndarray[float, 1],
        target0: int,
        target1: int,
    ) -> QuantumCircuit:
        """
        Convert `self` to a circuit object on `nqubits` qubits with gates
        applied to the `target0` and `target1`-th qubits.

        Args:
            nqubits (int > 0):
                The number of qubits in the circuit.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition.
            target0 (int >= 0, < `nqubits`):
                First qubit in the circuit to which gates will be applied.
            target1 (int >= 0, < `nqubits`):
                Second qubit in the circuit to which gates will be applied.

        Returns:
            circuit (qiskit.circuit.QuantumCircuit):
                The circuit implementing the decomposed gate.

        Raises:
            ValueError:
                - `nqubits` is less than 1
                - `target0` is negative, or greater than or equal to `nqubits`
                - `target1` is negative, or greater than or equal to `nqubits`
                - `len(params)` is not equal to `self.num_params()`
        """
        return self.decomp.to_circuit(nqubits, params, target0, target1)

