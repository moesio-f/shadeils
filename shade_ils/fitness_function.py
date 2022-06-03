import typing
import numpy as np


class FitnessFunction(typing.NamedTuple):
    fn: typing.Callable[[np.ndarray], float]
    info: typing.Dict[str, typing.Any]
    name: str


class Sphere:
    def __init__(self, dims: int,
                 lower_bound: float = -100.0,
                 upper_bound: float = 100.0) -> None:
        self.dims = dims
        self.lower = lower_bound
        self.upper = upper_bound

    def __call__(self, x: np.ndarray, *args: typing.Any, **kwds: typing.Any) -> float:
        x = x + 50.0
        return np.sum(x**2)

    def as_fitness_function(self) -> FitnessFunction:
        return FitnessFunction(fn=self,
                               info={'lower': self.lower,
                                     'upper': self.upper,
                                     'dimension': self.dims},
                               name='Sphere')


class Ackley:
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

    def __call__(self, x: np.ndarray) -> float:
        x = x + 10.0
        d = x.shape[-1]
        sum1 = np.sum(x * x, axis=-1)
        sum2 = np.sum(np.cos(self.c * x), axis=-1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = np.exp(sum2 / d)
        result = term1 - term2 + self.a + np.math.e

        return result

    def as_fitness_function(self) -> FitnessFunction:
        return FitnessFunction(fn=self,
                               info={'lower': self.lower,
                                     'upper': self.upper,
                                     'dimension': self.dims},
                               name='Ackley')
