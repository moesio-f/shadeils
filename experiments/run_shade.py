import time

from shade_ils.benchmark import Ackley
from shade_ils.shade import ShadeOptimizer

fn = Ackley(50000)
optimizer = ShadeOptimizer(fn=fn, 
                           population_size=500, 
                           max_evaluations=int(1e4), 
                           seed=42)
start = time.perf_counter()
result = optimizer.optimize()
end = time.perf_counter()

print(f'Duration: {end - start:.2f}')
print(result.fitness)
print(result.evaluations)
