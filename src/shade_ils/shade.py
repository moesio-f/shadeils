"""
This program contains the SHADE algorithm, proposed in [1].

It is a DE that adapts its F and CR parameters, it uses a memory of solutions,
and a new crossover operator.

Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
Congress on , vol., no., pp.71,78, 20-23 June 2013
doi:10.1109/CEC.2013.6557555
"""
from __future__ import annotations

import math

import numpy as np

from . import utils
from .entities import EAResult, FitnessFunction


class ShadeOptimizer:
    MAX_SEED: int = 90000

    def __init__(self,
                 fn: FitnessFunction,
                 population_size: int,
                 max_evaluations: int,
                 seed: int,
                 memF: np.ndarray = None,
                 memCR: np.ndarray = None,
                 history_size: int = 100,
                 start_population: np.ndarray = None,
                 start_population_fitness: np.ndarray = None,
                 initial_solution: np.ndarray = None) -> None:
        if memF is None:
            memF = np.ones(history_size) * 0.5

        if memCR is None:
            memCR = np.ones(history_size) * 0.5

        self.fn = fn
        self.pop_size = population_size
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.memF = memF
        self.memCR = memCR
        self.history_size = history_size
        self.rng = np.random.default_rng(seed)
        self.population = start_population
        self.population_fitness = start_population_fitness
        self.initial_solution = initial_solution
        self.n_evaluations = 0

    def optimize(self) -> EAResult:
        domain = (self.fn.info()['lower'], self.fn.info()['upper'])

        if self.population is None:
            pop_seed = self.rng.integers(0, self.MAX_SEED)
            self.population = utils.random_population(domain,
                                                      self.fn.info()['dimension'],
                                                      self.pop_size,
                                                      seed=pop_seed)

        if self.initial_solution is not None:
            self.population[0] = self.initial_solution

        if self.population_fitness is None:
            self.population_fitness = self.fn(self.population)
            self.n_evaluations = self.pop_size

        # Init memory with population
        memory = self.population.tolist()
        memorySize = self.pop_size * 2
        k = 0
        p_min = 2.0 / self.pop_size

        while self.n_evaluations < self.max_evaluations:
            SCR = []
            SF = []
            F = np.zeros(self.pop_size)
            CR = np.zeros(self.pop_size)
            u = np.zeros((self.pop_size, self.fn.info()['dimension']))
            best_fitness = np.min(self.population_fitness)

            for (i, xi) in enumerate(self.population):
                # Getting F and CR for that solution
                index_H = np.random.randint(0, self.history_size)
                meanF = self.memF[index_H]
                meanCR = self.memCR[index_H]
                Fi = np.random.normal(meanF, 0.1)
                CRi = np.random.normal(meanCR, 0.1)
                p = np.random.rand() * (0.2-p_min) + p_min

                # Get two random values
                seed = self.rng.integers(0, self.MAX_SEED)
                r1 = utils.random_indexes(1,
                                          self.pop_size,
                                          seed=seed,
                                          ignore=[i])
                # Get the second from the memory
                seed = self.rng.integers(0, self.MAX_SEED)
                r2 = utils.random_indexes(1,
                                          len(memory),
                                          seed=seed,
                                          ignore=[i, r1])
                xr1 = self.population[r1]
                xr2 = memory[r2]

                # Get one of the p best values
                maxbest = int(p * self.pop_size)
                bests = np.argsort(self.population_fitness)[:maxbest]
                pbest = np.random.choice(bests)
                xbest = self.population[pbest]

                # Mutation
                v = xi + Fi*(xbest - xi) + Fi*(xr1-xr2)

                # Special clipping
                v = self.shade_clip(domain, v, xi)

                # Crossover
                idxchange = np.random.rand(self.fn.info()['dimension']) < CRi
                u[i] = np.copy(xi)
                u[i, idxchange] = v[idxchange]
                F[i] = Fi
                CR[i] = CRi

            # Update population and SF, SCR
            weights = []

            fitness_us = self.fn(u)
            for i, fitness in enumerate(self.population_fitness):
                fitness_u = fitness_us[i]
                assert not math.isnan(fitness_u)

                if fitness_u <= fitness:
                    # Add to memory
                    if fitness_u < fitness:
                        memory.append(self.population[i])
                        SF.append(F[i])
                        SCR.append(CR[i])
                        weights.append(fitness - fitness_u)

                    if (fitness_u < best_fitness):
                        best_fitness = fitness_u

                    self.population[i] = u[i]
                    self.population_fitness[i] = fitness_u

            self.n_evaluations += self.pop_size

            # Check the memory
            memory = self.limit_memory(memory, memorySize)

            # Update MemCR and MemF
            if len(SCR) > 0 and len(SF) > 0:
                Fnew, CRnew = self.update_FCR(SF, SCR, weights)
                self.memF[k] = Fnew
                self.memCR[k] = CRnew
                k = (k + 1) % self.history_size

        bestIndex = np.argmin(self.population_fitness)
        bestFitness = self.population_fitness[bestIndex]
        bestSolution = self.population[bestIndex]

        return EAResult(fitness=bestFitness,
                        solution=bestSolution,
                        evaluations=self.n_evaluations,
                        best_index=bestIndex)

    def limit_memory(self, memory, memorySize):
        """
        Limit the memory to  the memorySize
        """
        memory = np.array(memory)

        if len(memory) > memorySize:
            indexes = np.random.permutation(len(memory))[:memorySize]
            memory = memory[indexes]

        return memory.tolist()

    def update_FCR(self, SF, SCR, improvements):
        """
        Update the new F and CR using the new Fs and CRs, and its improvements
        """
        total = np.sum(improvements)
        assert total > 0
        weights = improvements/total

        Fnew = np.sum(weights*SF*SF)/np.sum(weights*SF)
        Fnew = np.clip(Fnew, 0, 1)
        CRnew = np.sum(weights*SCR)
        CRnew = np.clip(CRnew, 0, 1)

        return Fnew, CRnew

    def shade_clip(self, domain, solution, original):
        lower = domain[0]
        upper = domain[1]
        clip_sol = np.clip(solution, lower, upper)

        if np.all(solution == clip_sol):
            return solution

        idx_lowest = (solution < lower)
        solution[idx_lowest] = (original[idx_lowest]+lower)/2.0
        idx_upper = (solution > upper)
        solution[idx_upper] = (original[idx_upper]+upper)/2.0
        return solution
