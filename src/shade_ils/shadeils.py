import argparse
import sys
import os
import typing

import numpy as np
import scipy.optimize as scipy_optimize

import shade_ils.de as de
import ea
import shade_ils.fitness_function as fns
import shade_ils.shade as shade
import shade_ils.mts as mts


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
            print("new_method: {}".format(new_method))
            return new_method

        # Complete the ranking
        indexes = np.argsort(self.improvements.values())
        posbest = indexes[-1]
        best = list(self.improvements.keys())[posbest]
        return best


def get_improvement(alg_name, before, after):
    """
    Print the improvement with an algorithm
    """
    if before == 0:
        ratio = 0
    else:
        ratio = (before-after)/before

    return f"[IMPROVEMENT][{alg_name}] {before:.3f} -> {after:.3f} [delta={before-after:.3f}, ratio={ratio:.2f}]\n"


SR_global_MTS = []
SR_MTS = []


def apply_localsearch(name, method, fitness_fun, bounds, current_best, current_best_fitness, maxevals, fid):
    global SR_MTS
    global SR_global_MTS

    lower = bounds[0][0]
    upper = bounds[0][1]

    if method == 'grad':
        sol, fit, info = scipy_optimize.fmin_l_bfgs_b(
            fitness_fun, x0=current_best, approx_grad=1, bounds=bounds, maxfun=maxevals, disp=False)
        funcalls = info['funcalls']
    elif method == 'mts':
        if name.lower() == "global":
            SR = SR_global_MTS
        else:
            SR = SR_MTS

        res, SR_MTS = mts.mtsls(fitness_fun, current_best,
                                current_best_fitness, lower, upper, maxevals, SR)
        sol = res.solution
        fit = res.fitness
        funcalls = maxevals
    else:
        raise NotImplementedError(method)

    if fit <= current_best_fitness:
        fid.write(get_improvement("{0} {1}".format(
            method.upper(), name), current_best_fitness, fit))
        return de.EAResult(solution=np.array(sol), fitness=fit, evaluations=funcalls)
    else:
        return de.EAResult(solution=current_best, fitness=current_best_fitness, evaluations=funcalls)


def random_population(lower, upper, dimension, size):
    return np.random.uniform(lower, upper, dimension*size).reshape((size, dimension))


def applySHADE(crossover, fitness, funinfo, dimension, evals, population, populationFitness, bestId, current_best, fid, H=None):
    if current_best.fitness < populationFitness[bestId]:
        population[bestId, :] = current_best.solution
        populationFitness[bestId] = current_best.fitness

    if H is None:
        H = population.shape[0]

    result, bestId = shade.improve(run_info=funinfo, replace=False, dimension=dimension, name_output=None,
                                   population=population, H=H, population_fitness=populationFitness, fun=fitness, check_evals=evals, initial_solution=current_best.solution, MemF=applySHADE.MemF, MemCR=applySHADE.MemCR)
    fid.write(get_improvement("SHADE partial",
              current_best.fitness, result.fitness))
    return result, bestId


optimo = True


def check_evals(totalevals: int,
                evals: typing.List[int],
                bestFitness: de.EAResult,
                globalBestFitness: de.EAResult,
                fid):
    if not evals:
        return evals
    elif totalevals >= evals[0]:
        best = bestFitness if bestFitness.fitness < globalBestFitness.fitness else globalBestFitness
        fid.write(f"[MILESTONE][{evals[0]}] Fitness: {best.fitness}\n"
                  f"[MILESTONE][{evals[0]}] Solution: {best.solution}\n"
                  f"[MILESTONE][{evals[0]}] FEs: {totalevals}\n")
        fid.flush()
        evals.pop(0)

    return evals


def reset_ls(dim, lower, upper, method='all'):
    global SR_global_MTS
    global SR_MTS

    if method == 'all' or method == 'mts':
        SR_global_MTS = np.ones(dim)*(upper-lower)*0.2
        SR_MTS = SR_global_MTS


def reset_de(popsize, dimension, lower, upper, H, current_best_solution=None):
    population = random_population(lower, upper, dimension, popsize)

    if current_best_solution is not None:
        posrand = np.random.randint(popsize)
        population[posrand] = current_best_solution

    applySHADE.MemF = 0.5*np.ones(H)
    applySHADE.MemCR = 0.5*np.ones(H)
    return population


def set_region_ls():
    global SR_global_MTS
    global SR_MTS

    SR_MTS = np.copy(SR_global_MTS)


def get_ratio_improvement(previous_fitness, new_fitness):
    if previous_fitness == 0:
        improvement = 0
    else:
        improvement = (previous_fitness-new_fitness)/previous_fitness

    return improvement


def ihshadels(fitness: fns.FitnessFunction,
              milestones: typing.List[int],
              max_evals: int,
              fid,
              info_de,
              evals_gs: typing.Optional[int] = None,
              evals_de: typing.Optional[int] = None,
              evals_ls: typing.Optional[int] = None,
              population_size=100,
              debug=False,
              threshold=0.05):
    """
    Implementation of the proposal for CEC2015
    """
    lower = fitness.info['lower']
    upper = fitness.info['upper']
    dims = fitness.info['dimension']
    fitness_fun = fitness.fn
    funinfo = fitness.info
    evals = milestones
    totalevals = 1

    initial_sol = np.ones(dims)*((lower+upper)/2.0)
    current_best_fitness = fitness_fun(initial_sol)

    bounds = list(zip(np.ones(dims)*lower, np.ones(dims)*upper))
    bounds_partial = list(zip(np.ones(dims)*lower, np.ones(dims)*upper))

    population_size = min(population_size, 100)
    population = reset_de(population_size, dims, lower, upper, info_de)
    populationFitness = [fitness_fun(ind) for ind in population]
    bestId = np.argmin(populationFitness)

    initial_sol = np.ones(dims)*(lower+upper)/2.0
    initial_fitness = fitness_fun(initial_sol)

    if initial_fitness < populationFitness[bestId]:
        population[bestId] = initial_sol
        populationFitness[bestId] = initial_fitness
    
    fid.write(f"[INITIAL] Fitness: {populationFitness[bestId]}\n"
              f"[INITIAL] Solution: {population[bestId]}\n")

    current_best = de.EAResult(solution=population[bestId, :],
                               fitness=populationFitness[bestId],
                               evaluations=totalevals)

    crossover = ea.DEcrossover.SADECrossover(2)
    best_global_solution = current_best.solution
    best_global_fitness = current_best.fitness
    current_best_solution = best_global_solution

    apply_de = apply_ls = True
    applyDE = applySHADE

    reset_ls(dims, lower, upper)
    methods = ['mts', 'grad']

    pool_global = PoolLast(methods)
    pool = PoolLast(methods)

    num_worse = 0

    evals_gs = min(50*dims, 25000) if evals_gs is None else evals_gs
    evals_de = min(50*dims, 25000) if evals_de is None else evals_de
    evals_ls = min(10*dims, 5000) if evals_ls is None else evals_ls
    num_restarts = 0

    while totalevals < max_evals:
        method = ""

        if not pool_global.is_empty():
            previous_fitness = current_best.fitness
            method_global = pool_global.get_new()
            current_best = apply_localsearch(
                "Global", method_global, fitness_fun, bounds, current_best_solution, current_best.fitness, evals_gs, fid)
            totalevals += current_best.evaluations
            improvement = get_ratio_improvement(
                previous_fitness, current_best.fitness)

            pool_global.improvement(method_global, improvement, 2)
            evals = check_evals(totalevals,
                                evals,
                                current_best,
                                de.EAResult(fitness=best_global_fitness,
                                            solution=best_global_solution,
                                            evaluations=totalevals),
                                fid)
            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness:
                best_global_solution = np.copy(current_best_solution)
                best_global_fitness = fitness_fun(best_global_solution)

        for i in range(1):
            current_best = de.EAResult(
                solution=current_best_solution, fitness=current_best_fitness, evaluations=0)
            set_region_ls()

            method = pool.get_new()

            if apply_de:
                result, bestInd = applyDE(crossover, fitness_fun, funinfo, dims, evals_de,
                                          population, populationFitness, bestId, current_best, fid, info_de)
                improvement = current_best.fitness - result.fitness
                totalevals += result.evaluations
                evals = check_evals(totalevals,
                                    evals,
                                    result,
                                    de.EAResult(fitness=best_global_fitness,
                                                solution=best_global_solution,
                                                evaluations=totalevals),
                                    fid)
                current_best = result

            if apply_ls:
                result = apply_localsearch("Local", method, fitness_fun, bounds_partial,
                                           current_best.solution, current_best.fitness, evals_ls, fid)
                improvement = get_ratio_improvement(
                    current_best.fitness, result.fitness)
                totalevals += result.evaluations
                evals = check_evals(totalevals,
                                    evals,
                                    result,
                                    de.EAResult(fitness=best_global_fitness,
                                                solution=best_global_solution,
                                                evaluations=totalevals),
                                    fid)
                current_best = result

                pool.improvement(method, improvement, 10, .25)

            current_best_solution = current_best.solution
            current_best_fitness = current_best.fitness

            if current_best_fitness < best_global_fitness:
                best_global_fitness = current_best_fitness
                best_global_solution = np.copy(current_best_solution)

            # Restart if it is not improved
            if (previous_fitness == 0):
                ratio_improvement = 1
            else:
                ratio_improvement = (
                    previous_fitness-result.fitness)/previous_fitness

            fid.write(f"[IMPROVEMENT][TotalImprovement] ratio={int(100*ratio_improvement):d}%\n"
                      f"[IMPROVEMENT][TotalImprovement] fitness: {previous_fitness:.3f} => {result.fitness:.3f}\n"
                      f"[IMPROVEMENT][TotalImprovement] num_worse={num_worse}, restart={num_restarts}\n")

            if ratio_improvement >= threshold:
                num_worse = 0
            else:
                num_worse += 1
                imp_str = ",".join(["{}:{}".format(m, val)
                                   for m, val in pool.improvements.items()])
                fid.write(f"[IMPROVEMENT][POOLS] {imp_str}\n")

                # Random the LS
                reset_ls(dims, lower, upper, method)

            if num_worse >= 3:
                num_worse = 0
                fid.write("[RESTART] fitness={0:.2e} for ration={1:.2f}: with {2:d} evaluations\n".format(
                    current_best.fitness, ratio_improvement, totalevals))
                # Increase a 1% of values
                posi = np.random.choice(population_size)
                new_solution = np.random.uniform(-0.01,
                                                 0.01, dims)*(upper-lower)+population[posi]
                new_solution = np.clip(new_solution, lower, upper)
                current_best = de.EAResult(
                    solution=new_solution, fitness=fitness_fun(new_solution), evaluations=0)
                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                # Init DE
                population = reset_de(
                    population_size, dims, lower, upper, info_de)
                populationFitness = [fitness_fun(ind) for ind in population]
                totalevals += population_size

                totalevals += population_size
                # Random the LS
                pool_global.reset()
                pool.reset()
                reset_ls(dims, lower, upper)
                num_restarts += 1

            fid.write(f"[ITERATION] Best Fitness: {current_best_fitness:.2f}\n" 
            f"[ITERATION] Global Best Fitness: {best_global_fitness:.2f}\n" 
            f"[ITERATION] FEs: {totalevals}\n")
            fid.flush()

            if totalevals >= max_evals:
                break

    fid.write(f"[FINAL] Fitness: {best_global_fitness}\n"
              f"[FINAL] Solution: {best_global_solution}\n"
              f"[FINAL] FEs: {totalevals}\n")
    fid.flush()
    return result


def start(fitness: fns.FitnessFunction,
          shade_h: int = 100,
          population: int = 100,
          threshold: float = 1e-2,
          max_evals: int = int(3e6),
          evals_gs: int = None,  # Number of local search FEs
          evals_ls: int = None,  # Number of local search FEs
          evals_de: int = None,  # Number of differential evolution FEs
          milestones: typing.List[int] = list(map(int, [1.2e5, 6e5, 3e6])),
          runs: int = 1,
          seed=None,
          output_dir='results',
          fname_prefix='SHADE',
          verbose=False):
    global SR_MTS, SR_global_MTS

    if seed is None:
        np.random.seed(None)
        seed = np.random.randint(0, 999999999)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set seed
    np.random.seed(seed)

    configs_str = str(f"Function: {fitness.name}\n"
                      f"Seed: {seed}\n"
                      f"Threshold: {threshold}\n"
                      f"Population: {population}\n"
                      f"SHADE_H: {shade_h}\n"
                      f"MaxEvals: {max_evals}\n"
                      f"EvalsDE: {evals_de}\n"
                      f"EvalsGS: {evals_gs}\n"
                      f"EvalsLS: {evals_ls}\n"
                      f"Runs: {runs}\n"
                      f"Milestones: {milestones}\n")

    fname = fname_prefix + \
        f"_pop{population}_H{shade_h}_t{threshold:.2f}_{fitness.name}_{seed}r{runs}.txt"
    output = os.path.join(output_dir, fname)

    if not verbose:
        fid = open(output, 'w+')
        print(configs_str)
    else:
        fid = sys.stdout

    fid.write(configs_str)

    for _ in range(runs):
        SR_MTS = []
        SR_global_MTS = []
        ihshadels(fitness=fitness,
                  milestones=milestones,
                  max_evals=max_evals,
                  evals_gs=evals_gs,
                  evals_ls=evals_ls,
                  evals_de=evals_de,
                  fid=fid,
                  threshold=threshold,
                  population_size=population,
                  info_de=shade_h)

    fid.close()
