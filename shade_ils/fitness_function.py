import typing
import numpy as np


class FitnessFunction(typing.NamedTuple):
    fn: typing.Callable[[np.ndarray], float]
    info: typing.Dict[str, typing.Any]
    name: str

def _fn(x):
    x = x + 50.0
    d = x.shape[-1]
    sum1 = np.sum(x * x, axis=-1)
    sum2 = np.sum(np.cos(2*np.math.pi* x), axis=-1)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / d))
    term2 = np.exp(sum2 / d)
    result = term1 - term2 + 20.0 + np.math.e

    if result.dtype != x.dtype:
        result = result.astype(x.dtype)
    return result


SPHERE = FitnessFunction(fn=lambda x: np.sum(x**2) + 100.0,
                         info={'lower': -100.0,
                               'upper': 100.0,
                               'threshold': 1e-10,
                               'best': 0.0,
                               'dimension': 4},
                         name="Sphere")

ACKLEY = FitnessFunction(fn=_fn,
                            info={'lower': -100.0,
                                  'upper': 100.0,
                                  'threshold': 1e-10,
                                  'best': 0.0,
                                  'dimension': 10},
                            name="Ackley")
