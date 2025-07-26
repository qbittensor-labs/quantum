import numpy as np

def make_gen(seed: int) -> np.random.Generator:
    """
    Seed a new numpy random number generator.

    Args:
        seed (int):
            PRNG seed value.

    Returns:
        gen (numpy.random.Generator):
            The random number generator.
    """
    return np.random.Generator(np.random.PCG64(seed))


