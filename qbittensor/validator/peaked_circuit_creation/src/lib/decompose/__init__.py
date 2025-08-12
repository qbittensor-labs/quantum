from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

ObjectiveFn = Callable[[np.ndarray[complex, 2], np.ndarray[float, 1]], float]


def optim_decomp(
    U_target: np.ndarray[complex, 2],
    fidelity: ObjectiveFn,
    max_tries: int = 1000,
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
        max_tries (int > 0):
            Maximum number of calls to scipy for decomposition before failing
            with `RuntimeError`.
        epsilon (Optional[float]):
            Tolerance value for optimization.

    Returns:
        params (numpy.ndarray[float, 1]):
            1D array of final parameter values for the decomposition.

    Raises:
        RuntimeError:
            - `scipy.optimize` fails to satisfy error tolerance `epsilon` more
              than `max_tries` times.
    """
    # use a completely fixed generator here to keep decompositions consistent
    # without affecting global state, and without requiring a `seed` parameter
    gen = np.random.Generator(np.random.PCG64(10546))
    # scipy.optimize will sometimes get stuck in local minima; just sticking
    # this in a loop should be fine (really shouldn't need more than 2-3 iters
    # in 99% of cases)
    for _ in range(max_tries):
        params0 = 2 * np.pi * gen.random(size=15)
        optim_res = minimize(
            lambda params, targ: -fidelity(targ, params),
            params0,
            args=(U_target,),
            tol=epsilon,
            options=dict(maxiter=10000),
        )
        final_fidelity = fidelity(U_target, optim_res.x)
        if optim_res.success and abs(final_fidelity - 1) < epsilon:
            return optim_res.x
    raise RuntimeError(
        f"optim_decomp: got stuck in local minima >{max_tries} times")
