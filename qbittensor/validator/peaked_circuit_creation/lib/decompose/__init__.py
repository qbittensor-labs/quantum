from __future__ import annotations
from itertools import product
from typing import *
import numpy as np
from scipy.optimize import minimize
import qbittensor.validator.peaked_circuit_creation.lib.decompose.cnots as cnots
import qbittensor.validator.peaked_circuit_creation.lib.decompose.ising as ising

ObjectiveFn = Callable[
    [np.ndarray[complex, 2], np.ndarray[float, 1]],
    float
]

def optim_decomp(
    U_target: np.ndarray[complex, 2],
    fidelity: ObjectiveFn,
    epsilon: Optional[float] = 1e-6,
) -> np.ndarray[float, 1]:
    """
    Compute a decomposition of `U_target` using `scipy.optimize`, given an
    objective function.

    Args:
        U_target (numpy.ndarray[complex, 2]):
            Matrix to decompose.
        fidelity (ObjectiveFn):
            Objective function to maximize. This should take `U_target` as its
            first argument and a 1D array of parameter values as its second, and
            return a single scalar that increases with parameters closer to an
            optimim.
        epsilon (Optional[float]):
            Tolerance value for optimization.

    Returns:
        params (numpy.ndarray[float, 1]):
            1D array of final parameter values for the decomposition.
    """
    # use a completely fixed generator here to keep decompositions consistent
    # without affecting global state, and without requiring a `seed` parameter
    gen = np.random.Generator(np.random.PCG64(10546))
    params0 = max(
        (2 * np.pi * np.random.random(size=15) for _ in range(2000)),
        key=lambda params: fidelity(U_target, params)
    )
    optim_res = minimize(
        lambda params, targ: -fidelity(targ, params),
        params0,
        args=(U_target,),
        tol=epsilon,
        options=dict(maxiter=10000)
    )
    return optim_res.x

