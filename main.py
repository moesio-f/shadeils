from shade_ils import shadeils
from shade_ils import fitness_function as fns

if __name__ == '__main__':
    shadeils.start(fns.ACKLEY, max_evals=1000, milestones=list(range(1001)))