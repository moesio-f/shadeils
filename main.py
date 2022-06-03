from shade_ils import shadeils
from shade_ils import fitness_function as fns

if __name__ == '__main__':
    fn = fns.Ackley(100)
    shadeils.start(fn.as_fitness_function(),
                   population=100,
                   milestones=list(range(10000)),
                   verbose=True)
