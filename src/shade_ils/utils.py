"""Differential Evolution.
"""
from __future__ import annotations

import numpy as np


def random_population(domain: tuple[float, float],
                      dimension: int,
                      size: int,
                      seed) -> np.ndarray:
    """
    Return an initial population using a uniform random generator
    """
    assert domain[0] < domain[1]
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(low=domain[0],
                          high=domain[1],
                          size=(size, dimension))
    return uniform


def clip(domain: tuple[float, float],
         solution: np.ndarray) -> np.ndarray:
    """
    Returns the solution clippd between the values of the domain.

    Params
    ------
    domain vector with the lower and upper values.
    """
    assert domain[0] < domain[1]
    return np.clip(solution, domain[0], domain[1])


def random_indexes(n: int,
                   size: int,
                   seed,
                   ignore: list[int] = []) -> np.ndarray:
    """
    Returns a group of n indexes between 0 and size, avoiding ignore indexes.

    Params
    ------
    n number of indexes.
    size size of the vectors.
    ignore indexes to ignore.

    >>> random_indexes(1, 1)
    0
    >>> random_indexes(1, 2, [0])
    1
    >>> random_indexes(1, 2, [1])
    0
    >>> random_indexes(1, 3, [0, 1])
    2
    >>> random_indexes(1, 3, [0, 2])
    1
    >>> random_indexes(1, 3, [1, 2])
    0
    """
    rng = np.random.default_rng(seed)
    indexes = [pos for pos in range(size) if pos not in ignore]

    assert len(indexes) >= n
    rng.shuffle(indexes)

    if n == 1:
        return indexes[0]
    else:
        return indexes[:n]
