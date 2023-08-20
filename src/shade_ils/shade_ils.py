from __future__ import annotations

import numpy as np

from . import utils
from .entities import EAResult, FitnessFunction
from .lbfgs_b import LBFGSBOptimizer
from .mts_ls1 import MTSLS1
from .shade import ShadeOptimizer


class PoolLast:
    """
    This class allow us to have a pool of operation. When we ask the Pool one of
    them, the selected operator is decided following the last element whose improvement
    ratio was better. The idea is to apply more times the operator with a better
    improvement.
    """

    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        size = len(options)
        assert size > 0

        self.options = np.copy(options)
        self.improvements = []
        self.count_calls = 0
        self.first = np.random.permutation(self.options).tolist()

        self.new = None
        self.improvements = dict(zip(options, [0] * size))

    def reset(self):
        self.first = np.random.permutation(self.options).tolist()
        self.new = None
        options = self.options
        size = len(options)
        self.improvements = dict(zip(options, [0] * size))

    def has_no_improvement(self):
        return np.all([value == 0 for value in self.improvements.values()])

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        # First time it returns all
        if self.first:
            return self.first.pop()

        if self.new is None:
            self.new = self.update_prob()

        return self.new

    def is_empty(self):
        counts = self.improvements.values()
        return np.all(counts == 0)

    def improvement(self, obj, account, freq_update, minimum=0.15):
        """
        Received how much improvement this object has obtained (higher is better), it only update
        the method improvements

        :param object:
        :param account: improvement obtained (higher is better), must be >= 0
        :param freq_update: Frequency of improvements used to update the ranking
        :return: None
        """
        if account < 0:
            return

        if obj not in self.improvements:
            raise Exception("Error, object not found in PoolProb")

        previous = self.improvements[obj]
        self.improvements[obj] = account
        self.count_calls += 1

        if self.first:
            return

        if not self.new:
            self.new = self.update_prob()
        elif account == 0 or account < previous:
            self.new = self.update_prob()

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """

        if np.all([value == 0 for value in self.improvements.values()]):
            new_method = np.random.choice(list(self.improvements.keys()))
            return new_method

        # Complete the ranking
        indexes = np.argsort(self.improvements.values())
        posbest = indexes[-1]
        best = list(self.improvements.keys())[posbest]
        return best


class SHADEILSOptimizer:
    MAX_SEED: int = 90000

    def __init__(self,
                 fn: FitnessFunction,
                 population_size: int,
                 max_evaluations: int,
                 seed: int,
                 history_size: int,
                 evaluations_gs: int = None,
                 evaluations_de: int = None,
                 evaluations_ls: int = None,
                 threshold=0.05) -> None:
        dims = fn.info()['dimension']

        if evaluations_gs is None:
            evaluations_gs = min(50*dims, 25000)

        if evaluations_de is None:
            evaluations_de = min(50*dims, 25000)

        if evaluations_ls is None:
            evaluations_ls = min(10*dims, 5000)

        sum_evaluations = evaluations_gs + evaluations_de + evaluations_ls
        assert sum_evaluations < max_evaluations

        self.fn = fn
        self.pop_size = min(population_size, 100)
        self.max_evaluations = max_evaluations

        self.evaluations_gs = evaluations_gs
        self.evaluations_de = evaluations_de
        self.evaluations_ls = evaluations_ls
        self.threshold = threshold

        self.SR_global_MTS = []
        self.SR_MTS = []
        self.optima = True

        self.history_size = history_size
        self.shade_memF = 0.5*np.ones(self.history_size)
        self.shade_memCR = 0.5*np.ones(self.history_size)

        self.seed = seed

    def optimize(self) -> EAResult:
        """
        Implementation of the proposal for CEC2015
        """
        lower = self.fn.info()['lower']
        upper = self.fn.info()['upper']
        dims = self.fn.info()['dimension']
        rng = np.random.default_rng(self.seed)

        n_evaluations = 0
        bounds = list(zip(np.ones(dims)*lower,
                          np.ones(dims)*upper))
        bounds_partial = list(zip(np.ones(dims)*lower,
                                  np.ones(dims)*upper))

        population = self._reset_de(seed=rng.integers(0, self.MAX_SEED))
        population_fitness = self.fn(population)
        best_idx = np.argmin(population_fitness)
        n_evaluations += self.pop_size

        initial_sol = np.ones(dims) * (lower+upper) / 2.0
        initial_fitness = self.fn(np.expand_dims(initial_sol,
                                                 axis=0))[0]
        current_best_fitness = initial_fitness
        n_evaluations += 1  # 1 FE

        if initial_fitness < population_fitness[best_idx]:
            population[best_idx] = initial_sol
            population_fitness[best_idx] = initial_fitness

        current_best = EAResult(solution=population[best_idx, :],
                                fitness=population_fitness[best_idx],
                                evaluations=n_evaluations)

        best_global_solution = current_best.solution
        best_global_fitness = current_best.fitness
        current_best_solution = best_global_solution

        self._reset_ls()
        methods = ['mts', 'grad']
        pool_global = PoolLast(methods)
        pool = PoolLast(methods)

        num_worse = 0
        num_restarts = 0

        while n_evaluations < self.max_evaluations:
            method = ""

            if not pool_global.is_empty():
                previous_fitness = current_best.fitness
                method_global = pool_global.get_new()

                current_best = self._local_search("global",
                                                  method_global,
                                                  bounds,
                                                  current_best,
                                                  self.evaluations_gs,
                                                  rng.integers(0, self.MAX_SEED))

                n_evaluations += current_best.evaluations
                improvement = self._get_ratio_improvement(previous_fitness,
                                                          current_best.fitness)

                pool_global.improvement(method_global,
                                        improvement,
                                        2)

                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                if current_best_fitness < best_global_fitness:
                    best_global_solution = np.copy(current_best_solution)
                    best_global_fitness = self.fn(np.expand_dims(best_global_solution,
                                                                 axis=0))[0]
                    n_evaluations += 1

            current_best = EAResult(solution=current_best_solution,
                                    fitness=current_best_fitness,
                                    evaluations=0)
            self._set_region_ls()
            method = pool.get_new()

            # Aplicando o SHADE
            result = self._shade(population,
                                 population_fitness,
                                 best_idx,
                                 current_best,
                                 rng.integers(0, self.MAX_SEED))
            improvement = current_best.fitness - result.fitness
            n_evaluations += result.evaluations
            current_best = result

            # Aplicando Local Search
            result = self._local_search("local",
                                        method,
                                        bounds_partial,
                                        current_best,
                                        self.evaluations_ls,
                                        rng.integers(0, self.MAX_SEED))
            improvement = self._get_ratio_improvement(current_best.fitness,
                                                      result.fitness)
            n_evaluations += result.evaluations
            current_best = result

            # Atualizando a pool
            pool.improvement(method, improvement, 10, .25)

            # Atualizando melhores soluções conhecidas
            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness:
                best_global_fitness = current_best_fitness
                best_global_solution = np.copy(current_best_solution)

            # Restart if it is not improved
            if previous_fitness == 0:
                ratio_improvement = 1
            else:
                diff = previous_fitness - result.fitness
                ratio_improvement = diff/previous_fitness

            if ratio_improvement >= self.threshold:
                num_worse = 0
            else:
                num_worse += 1

                # Random the LS
                self._reset_ls(method)

            if num_worse >= 3:
                num_worse = 0

                # Increase a 1% of values
                pos_i = rng.choice(self.pop_size)

                delta = rng.uniform(-0.01, 0.01, dims)*(upper-lower)
                new_solution = delta + population[pos_i]
                new_solution = np.clip(new_solution, lower, upper)
                new_solution_fitness = self.fn(np.expand_dims(new_solution,
                                                              axis=0))[0]

                current_best = EAResult(solution=new_solution,
                                        fitness=new_solution_fitness,
                                        evaluations=0)

                # Atualizando melhores soluções conhecidas
                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                # 1 FE new solution
                n_evaluations += 1

                # Init DE
                population = self._reset_de(rng.integers(0, self.MAX_SEED))
                population_fitness = self.fn(population)

                # 1 FE for every individual in population
                n_evaluations += self.pop_size

                # Random the LS
                pool_global.reset()
                pool.reset()
                self._reset_ls()

                num_restarts += 1

            if n_evaluations >= self.max_evaluations:
                break

        return EAResult(solution=best_global_solution,
                        fitness=best_global_fitness,
                        evaluations=n_evaluations)

    def _local_search(self,
                      name: str,
                      method: str,
                      bounds: list[tuple[float, float]],
                      current_best: EAResult,
                      max_evaluations: int,
                      seed: int) -> EAResult:
        rng = np.random.default_rng(seed)

        if method == 'grad':
            result = LBFGSBOptimizer(fn=self.fn,
                                     initial_solution=current_best,
                                     max_evaluations=max_evaluations,
                                     bounds=bounds,
                                     ensure_evaluations=True).optimize()
            sol = result.solution
            fit = result.fitness
            n_evaluations = result.evaluations
        elif method == 'mts':
            if name.lower() == "global":
                SR = self.SR_global_MTS
            else:
                SR = self.SR_MTS

            mts_seed = rng.integers(0, self.MAX_SEED)
            result = MTSLS1(fn=self.fn,
                            solution=current_best,
                            max_evaluations=max_evaluations,
                            seed=mts_seed,
                            SR=SR).search()
            sol = result.solution
            fit = result.fitness
            n_evaluations = result.evaluations
        else:
            raise NotImplementedError(method)

        if fit <= current_best.fitness:
            return EAResult(solution=np.array(sol),
                            fitness=fit,
                            evaluations=n_evaluations)

        return EAResult(solution=current_best.solution,
                        fitness=current_best.fitness,
                        evaluations=n_evaluations)

    def _shade(self,
               population: np.ndarray,
               population_fitness: np.ndarray,
               best_idx: int,
               current_best: EAResult,
               seed: int,
               H: int = None) -> EAResult:
        """Executa o algoritmo SHADE utilizando
        como critério de parada a quantidade de
        avaliações para DE.

        Args:
            population (np.ndarray): população.
            population_fitness (np.ndarray): fitness da população.
            best_idx (int): melhor índice.
            current_best (EAResult): melhor resultado.
            seed (int): seed randômica.
            H (int, optional): Tamanho do histórico 
                (default=None, usamos o tamanho da
                população).

        Returns:
            EAResult: novo resultado.
        """
        if current_best.fitness < population_fitness[best_idx]:
            population[best_idx, :] = current_best.solution
            population_fitness[best_idx] = current_best.fitness

        if H is None:
            H = population.shape[0]

        optimizer = ShadeOptimizer(
            fn=self.fn,
            population_size=population.shape[0],
            max_evaluations=self.evaluations_de,
            seed=seed,
            memF=self.shade_memF,
            memCR=self.shade_memCR,
            history_size=H,
            start_population=population,
            start_population_fitness=population_fitness,
            initial_solution=current_best.solution)

        return optimizer.optimize()

    def _reset_ls(self, method: str = 'all'):
        u, l = self.fn.info()['upper'], self.fn.info()['lower']
        dims = self.fn.info()['dimension']

        if method == 'all' or method == 'mts':
            self.SR_global_MTS = np.ones(dims)*(u-l)*0.2
            self.SR_MTS = self.SR_global_MTS

    def _set_region_ls(self):
        self.SR_MTS = np.copy(self.SR_global_MTS)

    def _get_ratio_improvement(self, previous_fitness, new_fitness):
        if previous_fitness == 0:
            improvement = 0
        else:
            improvement = (previous_fitness-new_fitness)/previous_fitness

        return improvement

    def _reset_de(self, seed: int, current_best: EAResult = None):
        rng = np.random.default_rng(seed)
        domain = (self.fn.info()['lower'], self.fn.info()['upper'])
        dims = self.fn.info()['dimension']

        pop_seed = rng.integers(0, self.MAX_SEED)
        population = utils.random_population(domain=domain,
                                             dimension=dims,
                                             size=self.pop_size,
                                             seed=pop_seed)

        if current_best is not None:
            rand_idx = rng.integers(0, self.pop_size)
            population[rand_idx] = current_best.solution

        self.shade_memF = 0.5 * np.ones(self.history_size)
        self.shade_memCR = 0.5 * np.ones(self.history_size)

        return population
