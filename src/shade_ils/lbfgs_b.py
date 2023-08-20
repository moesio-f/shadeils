from __future__ import annotations

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .entities import EAResult, FitnessFunction


class LBFGSBOptimizer:
    def __init__(self,
                 fn: FitnessFunction,
                 max_evaluations: int,
                 bounds: list[tuple[float, float]],
                 initial_solution: EAResult,
                 ensure_evaluations: bool = True) -> None:
        self.fn = fn
        self.max_evaluations = max_evaluations
        self.bounds = bounds
        self.max_evaluations = ensure_evaluations
        self.initial_solution = initial_solution

    def optimize(self) -> EAResult:
        n_evaluations = 0
        result: EAResult = None

        def _fn(x):
            nonlocal n_evaluations
            nonlocal result
            out = self.fn(np.expand_dims(x, axis=0))[0]
            n_evaluations += 1
            candidate = EAResult(fitness=out,
                                 solution=x,
                                 evaluations=n_evaluations)

            if (result is None) or (out <= result.fitness):
                # Caso não tenhamos um resultado ainda
                #   ou o valor seja melhor que o já temos,
                #   atualizamos.
                result = candidate

            if n_evaluations >= self.max_evaluations:
                # Forçando um erro na otimização
                #   pelo L-BFGS-B passando um
                #   valor inválido da função de
                #   fitness.
                return None

            return out

        try:
            sol, fit, info = fmin_l_bfgs_b(
                _fn,
                x0=self.initial_solution.solution,
                approx_grad=True,
                bounds=self.bounds,
                maxfun=self.max_evaluations,
                disp=0)

            if fit <= result.fitness:
                result = EAResult(fitness=fit,
                                solution=sol,
                                n_evaluations=info['funcalls'])
        except Exception:
            pass

        return result
