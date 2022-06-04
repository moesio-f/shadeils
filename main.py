from shade_ils import shadeils
from shade_ils import fitness_function as fns

if __name__ == '__main__':
    MAX_EVALS = 1000
    DIMS = 20
    POPULATION = 50

    fn = fns.Ackley(DIMS)
    shadeils.start(fn.as_fitness_function(),
                   population=POPULATION,
                   milestones=list(range(0, MAX_EVALS + 1, MAX_EVALS//10)),
                   max_evals=MAX_EVALS,
                   evals_de=25,
                   evals_gs=15,
                   evals_ls=10,
                   verbose=True)
