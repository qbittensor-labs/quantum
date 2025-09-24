import numpy as np
from numpy import (sin, cos, exp, kron, conjugate)

"""
Functions in this module compute quantities relevant to a "CNOT-based"
decomposition of a general SU(4) gate according to [this form][cnot-based]:
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

def u3(alpha: float, beta: float, gamma: float) -> np.ndarray[complex, 2]:
    return np.array([
        [
            cos(alpha / 2),
            -exp(1j * gamma) * sin(alpha / 2),
        ],
        [
            exp(1j * beta) * sin(alpha / 2),
            exp(1j * (beta + gamma)) * cos(alpha / 2),
        ],
    ])

def rx(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ cos(angle / 2), -1j * sin(angle / 2), ],
        [ -1j * sin(angle / 2), cos(angle / 2), ],
    ])

def rz(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ exp(-1j * angle / 2), 0 ],
        [ 0, exp( 1j * angle / 2) ],
    ])

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
    return np.array([
        [
            cos_a0 * cos_b0,
            -ei_b2 * cos_a0 * sin_b0,
            -ei_a2 * sin_a0 * cos_b0,
            ei_a2_b2 * sin_a0 * sin_b0,
        ],
        [
            ei_b1 * cos_a0 * sin_b0,
            ei_b1_b2 * cos_a0 * cos_b0,
            -ei_a2_b1 * sin_a0 * sin_b0,
            -ei_a2_b1_b2 * sin_a0 * cos_b0,
        ],
        [
            ei_a1 * sin_a0 * cos_b0,
            -ei_a1_b2 * sin_a0 * sin_b0,
            ei_a1_a2 * cos_a0 * cos_b0,
            -ei_a1_a2_b2 * cos_a0 * sin_b0,
        ],
        [
            ei_a1_b1 * sin_a0 * sin_b0,
            ei_a1_b1_b2 * sin_a0 * cos_b0,
            ei_a1_a2_b1 * cos_a0 * sin_b0,
            ei_a1_a2_b1_b2 * cos_a0 * cos_b0,
        ],
    ])

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
    return np.array([
        [ ei_mh1 * cos_h0, 0, 0, -1j * ei_mh1 * sin_h0 ],
        [ 0, ei_ph1 * cos_h0, -1j * ei_ph1 * sin_h0, 0 ],
        [ 0, -1j * ei_ph1 * sin_h0, ei_ph1 * cos_h0, 0 ],
        [ -1j * ei_mh1 * sin_h0, 0, 0, ei_mh1 * cos_h0 ],
    ])

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
    return np.array([
        [
            ei_mh2 * cos_g0 * cos_d0,
            -ei_ph2 * ei_d2 * cos_g0 * sin_d0,
            ei_mh2 * ei_g2_d2 * sin_g0 * sin_d0,
            -ei_ph2 * ei_g2 * sin_g0 * cos_d0,
        ],
        [
            ei_mh2 * ei_d1 * cos_g0 * sin_d0,
            ei_ph2 * ei_d1_d2 * cos_g0 * cos_d0,
            -ei_mh2 * ei_g2_d1_d2 * sin_g0 * cos_d0,
            -ei_ph2 * ei_g2_d1 * sin_g0 * sin_d0,
        ],
        [
            ei_mh2 * ei_g1 * sin_g0 * cos_d0,
            -ei_ph2 * ei_g1_d2 * sin_g0 * sin_d0,
            -ei_mh2 * ei_g1_g2_d2 * cos_g0 * sin_d0,
            ei_ph2 * ei_g1_g2 * cos_g0 * cos_d0,
        ],
        [
            ei_mh2 * ei_g1_d1 * sin_g0 * sin_d0,
            ei_ph2 * ei_g1_d1_d2 * sin_g0 * cos_d0,
            ei_mh2 * ei_g1_g2_d1_d2 * cos_g0 * cos_d0,
            ei_ph2 * ei_g1_g2_d1 * cos_g0 * sin_d0,
        ],
    ])

def make_uni(params: np.ndarray[float, 1]) -> np.ndarray[complex, 2]:
    """
    Compute the full unitary matrix following the decomposition form above.
    """
    [
        alpha0, alpha1, alpha2,
        beta0, beta1, beta2,
        eta0, eta1, eta2,
        gamma0, gamma1, gamma2,
        delta0, delta1, delta2
    ] = params

    return (
        u3u3(alpha0, alpha1, alpha2, beta0, beta1, beta2)
        @ cnot_rxrz_cnot(eta0, eta1)
        @ u3u3_cnot_irz(gamma0, gamma1, gamma2, delta0, delta1, delta2, eta2)
    )

def fidelity(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
) -> float:
    r"""
    Compute the gate fidelity
        F = |Tr(U^\dagger U')|^2 / 16
    where U' is the unitary to decompose and U (`U_target`) is the result of
    plugging `params` into the "Ising-like" decomposition above. `params` is
    expected as
        params = [
            alpha0, .., alpha2,
            beta0, .., beta2,
            eta0, .., eta2,
            gamma0, .., gamma2,
            delta0, .., delta2,
        ]
    """
    uni = make_uni(params)
    return abs(np.vdot(uni, U_target)) ** 2 / 16.0

def step(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
    stepsize: float,
    pos: int,
) -> float:
    params[pos] += stepsize
    f_plus = fidelity(U_target, params)
    params[pos] -= 2 * stepsize
    f_minus = fidelity(U_target, params)
    params[pos] += stepsize
    return (f_plus - f_minus) / (2 * stepsize)

def fidelity_grad(
    U_target: np.ndarray[complex, 2],
    params: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    """
    Compute the gradient of the fidelity with respect to all parameters.
    """
    return np.array([step(U_target, params, 1e-6, k) for k in range(15)])

