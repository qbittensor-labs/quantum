import numpy as np
from numpy import (sin, cos, exp, kron, conjugate)

"""
Functions in this module compute quantities relevant to an "Ising-like"
decomposition of a general SU(4) gate according to [this form][pennylane]:
    U =
        U3(0, [alpha0, alpha1, alpha2]) U3(1, [beta0, beta1, beta2])
        R_XX([eta0]) R_YY([eta1]) R_ZZ([eta2])
        U3(0, [gamma0, gamma1, gamma2]) U3(1, [delta0, delta1, delta2])

[pennylane]: https://pennylane.ai/qml/demos/tutorial_kak_decomposition#kokcu-fdhs
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

def rxx(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ cos(angle / 2), 0, 0, -1j * sin(angle / 2) ],
        [ 0, cos(angle / 2), -1j * sin(angle / 2), 0 ],
        [ 0, -1j * sin(angle / 2), cos(angle / 2), 0 ],
        [ -1j * sin(angle / 2), 0, 0, cos(angle / 2) ],
    ])

def ryy(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ cos(angle / 2), 0, 0,  1j * sin(angle / 2) ],
        [ 0, cos(angle / 2), -1j * sin(angle / 2), 0 ],
        [ 0, -1j * sin(angle / 2), cos(angle / 2), 0 ],
        [  1j * sin(angle / 2), 0, 0, cos(angle / 2) ],
    ])

def rzz(angle: float) -> np.ndarray[complex, 2]:
    return np.array([
        [ exp(-1j * angle / 2), 0, 0, 0 ],
        [ 0, exp( 1j * angle / 2), 0, 0 ],
        [ 0, 0, exp( 1j * angle / 2), 0 ],
        [ 0, 0, 0, exp(-1j * angle / 2) ],
    ])

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
    return np.array([
        [ ei_mh2 * cos_hdiff, 0, 0, -1j * ei_mh2 * sin_hdiff ],
        [ 0, ei_ph2 * cos_hsum , -1j * ei_ph2 * sin_hsum , 0 ],
        [ 0, -1j * ei_ph2 * sin_hsum , ei_ph2 * cos_hsum , 0 ],
        [ -1j * ei_mh2 * sin_hdiff, 0, 0, ei_mh2 * cos_hdiff ],
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
        @ rxx_ryy_rzz(eta0, eta1, eta2)
        @ u3u3(gamma0, gamma1, gamma2, delta0, delta1, delta2)
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
    return abs(np.diag(uni.T.conjugate() @ U_target).sum()) ** 2 / 16

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

