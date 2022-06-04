from shade_ils import shadeils
from shade_ils import fitness_function as fns

if __name__ == '__main__':
    fn = fns.Ackley(20)
    shadeils.start(fn.as_fitness_function(),
                   population=10,
                   milestones=list(range(50, 501, 50)),
                   max_evals=500,
                   evals_de=25,
                   evals_gs=15,
                   evals_ls=10,
                   verbose=True)
