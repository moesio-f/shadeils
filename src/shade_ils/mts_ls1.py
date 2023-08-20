"""
Implements the MTS-LS1 indicated in MTS 
http://sci2s.ugr.es/EAMHCO/pdfs/contributionsCEC08/tseng08mts.pdf 

Lin-Yu Tseng; Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 
Evolutionary Computation, 2008. CEC 2008. (IEEE World Congress on Computational Intelligence). 
IEEE Congress on , vol., no., pp.3052,3059, 1-6 June 2008
doi: 10.1109/CEC.2008.4631210
and used by MOS
"""
from functools import partial

import numpy as np

from .entities import EAResult, FitnessFunction


class MTSLS1:
    def __init__(self,
                 fn: FitnessFunction,
                 solution: EAResult,
                 max_evaluations: int,
                 seed: int,
                 SR,
                 ensure_evaluations: bool = True) -> None:
        self.fn = fn
        self.solution = solution
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.SR = SR
        self.ensure_evaluations = ensure_evaluations

    def search(self) -> EAResult:
        """
        Implements the MTS LS.
        """
        function = self.fn
        lower, upper = function.info()['lower'], function.info()['upper']
        sol = self.solution.solution
        rng = np.random.default_rng(self.seed)

        dim = len(sol)
        improved_dim = np.zeros(dim, dtype=bool)
        clip = partial(np.clip,
                       a_min=lower,
                       a_max=upper)
        current_best = EAResult(solution=sol,
                                fitness=self.solution.fitness,
                                evaluations=0)
        total_evaluations = 0
        improvement = np.zeros(dim)

        if total_evaluations < self.max_evaluations:
            dim_sorted = rng.permutation(dim)

            # OBS:. essa é uma mudança que não existia
            #   no código original.
            # Podemos configurar uma garantia para que
            #   as avaliações de fitness não passem
            #   do limite desejado selecionando apenas
            #   algumas das dimensões para melhorar.
            if self.ensure_evaluations:
                if 2 * len(dim_sorted) > self.max_evaluations:
                    max_size = self.max_evaluations // 2
                    dim_sorted = dim_sorted[:max_size]

            for i in dim_sorted:
                result = self._mtsls_improve_dim(current_best, i, clip)
                total_evaluations += result.evaluations
                improve = max(current_best.fitness - result.fitness, 0)
                improvement[i] = improve

                if improve:
                    improved_dim[i] = True
                    current_best = result
                else:
                    self.SR[i] /= 2

            dim_sorted = improvement.argsort()[::-1]
            d = 0

        while total_evaluations < self.max_evaluations:
            i = dim_sorted[d]
            result = self._mtsls_improve_dim(current_best, i, clip)
            total_evaluations += result.evaluations
            improve = max(current_best.fitness - result.fitness, 0)
            improvement[i] = improve
            next_d = (d + 1) % dim
            next_i = dim_sorted[next_d]

            if improve:
                improved_dim[i] = True
                current_best = result

                if improvement[i] < improvement[next_i]:
                    dim_sorted = improvement.argsort()[::-1]
            else:
                self.SR[i] /= 2
                d = next_d

        # Check lower value
        initial_SR = 0.2 * (upper - lower)
        self.SR[self.SR < 1e-15] = initial_SR

        return EAResult(solution=current_best.solution,
                        fitness=current_best.fitness,
                        evaluations=total_evaluations)

    def _mtsls_improve_dim(self,
                           solution: EAResult,
                           i: int,
                           clip) -> EAResult:
        function = self.fn
        sol = solution.solution
        best_fitness = solution.fitness

        new_sol = np.copy(sol)
        new_sol[i] -= self.SR[i]
        new_sol = clip(new_sol)
        fitness_new_sol = function(np.expand_dims(new_sol,
                                                  axis=0))[0]
        evaluations = 1

        if fitness_new_sol < best_fitness:
            best_fitness = fitness_new_sol
            sol = new_sol
        elif fitness_new_sol > best_fitness:
            new_sol = np.copy(sol)
            new_sol[i] += 0.5 * self.SR[i]
            new_sol = clip(new_sol)
            fitness_new_sol = function(np.expand_dims(new_sol,
                                                      axis=0))[0]
            evaluations += 1

            if fitness_new_sol < best_fitness:
                best_fitness = fitness_new_sol
                sol = new_sol

        return EAResult(solution=sol,
                        fitness=best_fitness,
                        evaluations=evaluations)
