import time

from shade_ils.benchmark import Ackley
from shade_ils.shade_ils import SHADEILSOptimizer

fn = Ackley(50000)
optimizer = SHADEILSOptimizer(fn=fn,
                              population_size=500,
                              max_evaluations=int(3e4),
                              history_size=100,
                              evaluations_gs=int(1e3),
                              evaluations_de=int(1e3),
                              evaluations_ls=int(1e3),
                              seed=42)
start = time.perf_counter()
result = optimizer.optimize()
end = time.perf_counter()

print(f'Duration: {end - start:.2f}')
print(result.fitness)
print(result.evaluations)
