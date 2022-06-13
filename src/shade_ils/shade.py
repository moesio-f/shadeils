"""
This program contains the SHADE algorithm, proposed in [1].

It is a DE that adapts its F and CR parameters, it uses a memory of solutions,
and a new crossover operator.

Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
Congress on , vol., no., pp.71,78, 20-23 June 2013
doi:10.1109/CEC.2013.6557555
"""
import math
import typing
import os
import numpy as np

import shade_ils.fitness_function as fns
from shade_ils import shadeils
from shade_ils.de import EAResult, random_population, get_experiments_file, random_indexes


def improve(fun, run_info, dimension, check_evals, name_output=None, replace=True, popsize=100, H=100, population=None, population_fitness=None, initial_solution=None, MemF=None, MemCR=None):
    """
    It applies the DE elements.

    Params
    ------
    fun function of the problem to optimize.

    run_info is a dictionary with the following information:
         lower: double lower bounds
         upper: double upper bounds
         threshold: minimum optim value (ignored)

    dimension of the problem.
    max_evals maximum_evaluations_numbers.
    name_output name of the output file
    run number of evaluations
    replace replace the file
    debug show the debug info if it is true
    PS population size
    """
    assert isinstance(dimension, int), 'dimension should be integer'
    assert (dimension > 0), 'dimension must be positive'

    final, fid = get_experiments_file(name_output, replace)

    if final is not None:
        return final

    for attr in ['lower', 'upper']:
        assert attr in run_info.keys(), "'{}' info not provided for benchmark".format(attr)

    # Added in a array the max evaluations
    if not isinstance(check_evals, list):
        check_evals = [check_evals]

    domain = (run_info['lower'], run_info['upper'])
    maxEval = check_evals.pop()

    if population is None:
        population = random_population(domain, dimension, popsize)
    else:
        popsize = population.shape[0]

    if initial_solution is not None:
        population[0] = initial_solution

    if population_fitness is None:
        population_fitness = np.array([fun(ind) for ind in population])
        currentEval = popsize
    else:
        currentEval = 0

    if fid is not None:
        # Find best fitness index
        bIndex = np.argmin(population_fitness)
        # Find best fitness
        bFit = population_fitness[bIndex]
        # Find best solution
        bSol = population[bIndex]

        fid.write(f"[INITIAL] Fitness: {bFit}\n"
                  f"[INITIAL] Solution: {shadeils._maybe_convert_to_list(bSol)}\n"
                  f"[INITIAL] FEs: {currentEval}\n")
        fid.flush()

    # Init memory with population
    memory = population.tolist()

    # Init F and CR
    memorySize = popsize*2

    if MemF is None:
        MemF = np.ones(H)*0.5

    if MemCR is None:
        MemCR = np.ones(H)*0.5

    k = 0
    pmin = 2.0/popsize

    while currentEval < maxEval:
        SCR = []
        SF = []
        F = np.zeros(popsize)
        CR = np.zeros(popsize)
        u = np.zeros((popsize, dimension))
        best_fitness = np.min(population_fitness)

        for (i, xi) in enumerate(population):
            # Getting F and CR for that solution
            index_H = np.random.randint(0, H)
            meanF = MemF[index_H]
            meanCR = MemCR[index_H]
            Fi = np.random.normal(meanF, 0.1)
            CRi = np.random.normal(meanCR, 0.1)
            p = np.random.rand()*(0.2-pmin)+pmin

            # Get two random values
            r1 = random_indexes(1, popsize, ignore=[i])
            # Get the second from the memory
            r2 = random_indexes(1, len(memory), ignore=[i, r1])
            xr1 = population[r1]
            xr2 = memory[r2]
            # Get one of the p best values
            maxbest = int(p*popsize)
            bests = np.argsort(population_fitness)[:maxbest]
            pbest = np.random.choice(bests)
            xbest = population[pbest]
            # Mutation
            v = xi + Fi*(xbest - xi) + Fi*(xr1-xr2)
            # Special clipping
            v = shade_clip(domain, v, xi)
            # Crossover
            idxchange = np.random.rand(dimension) < CRi
            u[i] = np.copy(xi)
            u[i, idxchange] = v[idxchange]
            F[i] = Fi
            CR[i] = CRi

        # Update population and SF, SCR
        weights = []

        for i, fitness in enumerate(population_fitness):
            fitness_u = fun(u[i])

            assert not math.isnan(fitness_u)

            if fitness_u <= fitness:
                # Add to memory
                if fitness_u < fitness:
                    memory.append(population[i])
                    SF.append(F[i])
                    SCR.append(CR[i])
                    weights.append(fitness-fitness_u)

                if (fitness_u < best_fitness):
                    best_fitness = fitness_u

                population[i] = u[i]
                population_fitness[i] = fitness_u

        currentEval += popsize

        # Check the memory
        memory = limit_memory(memory, memorySize)

        # Check evals
        if fid is not None:
            # Find best fitness index
            bIndex = np.argmin(population_fitness)
            # Find best fitness
            bFit = population_fitness[bIndex]
            # Find best solution
            bSol = population[bIndex]

            bEAResult = EAResult(
                fitness=bFit, solution=bSol, evaluations=currentEval)
            shadeils.check_evals(currentEval, check_evals,
                                 bEAResult, bEAResult, fid)

        # Update MemCR and MemF
        if len(SCR) > 0 and len(SF) > 0:
            Fnew, CRnew = update_FCR(SF, SCR, weights)
            MemF[k] = Fnew
            MemCR[k] = CRnew
            k = (k + 1) % H

    bestIndex = np.argmin(population_fitness)
    bestFitness = population_fitness[bestIndex]
    bestSolution = population[bestIndex]

    if fid is not None:
        fid.write("[SHADE] Mean[F,CR]: ({0:.2f}, {1:.2f})\n".format(
            MemF.mean(), MemCR.mean()))
        fid.write(f"[FINAL] Fitness: {bestFitness}\n"
                  f"[FINAL] Solution: {shadeils._maybe_convert_to_list(bestSolution)}\n"
                  f"[FINAL] FEs: {currentEval}\n")
        fid.close()

    return EAResult(fitness=bestFitness, solution=bestSolution, evaluations=currentEval), bestIndex


def limit_memory(memory, memorySize):
    """
    Limit the memory to  the memorySize
    """
    memory = np.array(memory)

    if len(memory) > memorySize:
        indexes = np.random.permutation(len(memory))[:memorySize]
        memory = memory[indexes]

    return memory.tolist()


def update_FCR(SF, SCR, improvements):
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


def shade_clip(domain, solution, original):
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


def start(fitness: fns.FitnessFunction,
          shade_h: int = 100,
          population: int = 100,
          max_evals: int = int(3e6),
          milestones: typing.List[int] = list(map(int, [1.2e5, 6e5, 3e6])),
          runs: int = 1,
          seed=None,
          output_dir='results',
          fname_prefix='SHADE'):

    if seed is None:
        np.random.seed(None)
        seed = np.random.randint(0, 999999999)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set seed
    np.random.seed(seed)

    fname = fname_prefix + \
        f"_pop{population}_H{shade_h}_{fitness.name}_{seed}r{runs}.txt"
    output = os.path.join(output_dir, fname)

    fn = fitness.fn
    info = fitness.info
    dims = info['dimension']

    for _ in range(runs):
        improve(fun=fn,
                run_info=info,
                dimension=dims,
                check_evals=milestones + [max_evals],
                name_output=output,
                replace=True,
                popsize=population,
                H=shade_h)
