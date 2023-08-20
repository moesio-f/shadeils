from __future__ import annotations

import numpy as np

from .entities import FitnessFunction


class Sphere(FitnessFunction):
    def __init__(self,
                 dims: int,

                 lower_bound: float = -100.0,
                 upper_bound: float = 100.0) -> None:
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound

    def call(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x ** 2, axis=-1)

    def name(self) -> str:
        return 'Sphere'

    def info(self) -> dict:
        return {
            'lower': self.lower,
            'upper': self.upper,
            'dimension': self.dims
        }


class Ackley(FitnessFunction):
    def __init__(self, dims: int,
                 lower_bound: float = -32.768,
                 upper_bound: float = 32.768,
                 a: float = 20,
                 b: float = 0.2,
                 c: float = 2*np.math.pi):
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound
        self.a = a
        self.b = b
        self.c = c

    def call(self, x: np.ndarray) -> np.ndarray:
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(np.cos(self.c * x), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.math.e

        return result

    def name(self) -> str:
        return 'Ackley'

    def info(self) -> dict:
        return {
            'lower': self.lower,
            'upper': self.upper,
            'dimension': self.dims
        }
