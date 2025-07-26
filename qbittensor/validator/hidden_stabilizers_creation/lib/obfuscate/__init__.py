from abc import abstractmethod, abstractstaticmethod
from typing import Callable, Optional, Self

import numpy as np
from scipy.optimize import minimize

class Decomp:
    """
    Base class for parameterized gate decompositions.
    """

    @abstractstaticmethod
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

    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters for the decomposition scheme.

        Returns:
            num_params (int > 0):
                Number of parameters.
        """

    @abstractmethod
    def fidelity(
        self,
        target: np.ndarray[complex, 2],
        params: np.ndarray[float, 1],
    ) -> float:
        """
        Compute the gate fidelity of the decomposition, given parameter values.

        Args:
            target (numpy.ndarray[complex, 2]):
                The target unitary to decompose.
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition scheme.

        Returns:
            fidelity (float):
                Gate fidelity of the decomposition. This should range from 0 to
                1, with 1 corresponding to a perfect decomposition.

        Raises:
            ValueError:
                - `len(params)` is not equal to `self.num_params()`
        """

    def compute_params(
        self,
        gen: np.random.Generator,
        target: np.ndarray[complex, 2],
        epsilon: Optional[float] = 1e-6,
    ) -> np.ndarray[float, 1]:
        """
        Compute decomposition parameters using `self.fidelity` and
        `optim_decomp`.

        Args:
            gen (numpy.random.Generator):
                The RNG source.
            target (numpy.ndarray[complex, 2]):
                The target unitary to decompose.
            epsilon (Optional[float]):
                Tolerance value for optimization.

        Returns:
            params (numpy.ndarray[float, 1]):
                Parameters for the decomposition scheme.
        """
        return optim_decomp(gen, self, target, epsilon)

ObjectiveFn = Callable[
    [np.ndarray[complex, 2], np.ndarray[float, 1]],
    float,
]

def optim_decomp(
    gen: np.random.Generator,
    decomp: Decomp,
    target: np.ndarray[complex, 2],
    epsilon: Optional[float] = 1e-6,
) -> np.ndarray[float, 1]:
    """
    Compute a decomposition of `target` using `scipy.optimize`, given an
    objective function.

    Args:
        gen (numpy.random.Generator):
            RNG source to choose an initial condition.
        decomp (Decomp):
            Decomposition scheme.
        target (numpy.ndarray[complex, 2]):
            Matrix to decompose.
        epsilon (Optional[float]):
            Tolerance value for optimization.

    Returns:
        params (numpy.ndarray[float, 1]):
            1D array of final parameter values for the decomposition.
    """
    params0 = 2 * np.pi * gen.random(size=decomp.num_params())
    optim = minimize(
        lambda params, targ, decomp: -decomp.fidelity(targ, params),
        params0,
        args=(target, decomp),
        tol=epsilon,
        options=dict(maxiter=10000),
    )
    return optim.x

