from wsn import wsn
import numpy as np
from pso_mbo import pso_mbo
from benchmark import (
    sphere,
    rosenbrock,
    rastrigin,
    schwefel_1_2,
    schwefel_2_21,
    schwefel_2_22,
    step,
    quartic,
    alpine
)

print("1. Coverage")
print("2. Benchmark")
val = int(input("Enter a values: "))
if val == 1:
    sensor = 50
    area = 50
    radius = 7
    pop_size = 20
    max_iter = 100
    p = 5.0 / 12.0
    peri = 1.2
    bar=5/12
    w=0.9
    c1=2.0
    c2=2.0

    wsn_cov = wsn(sensor, area, radius)

    dim = 2 * sensor
    bounds = [(0, area) for _ in range(dim)]

    optimizer = pso_mbo(
        wsn_cov.coverage_objective,
        bounds,
        pop_size,
        max_iter,
        peri,
        p,
        bar,
        w,
        c1,
        c2
    )

    best_pos, best_val = optimizer.optimize()

    print(f"Best Coverage Rate: {(1 - best_val) * 100:.2f}")

    wsn_cov.visualize(best_pos, title="PSO-MBO")
elif val == 2:
    functions = [
        ('Sphere', sphere, (-100, 100)),
        ('Schwefel 2.22', schwefel_2_22, (-10, 10)),
        ('Schwefel 1.2', schwefel_1_2, (-100, 100)),
        ('Schwefel 2.21', schwefel_2_21, (-100, 100)),
        ('Rosenbrock', rosenbrock, (-10, 10)),
        ('Step', step, (-100, 100)),
        ('Quartic', quartic, (-1.28, 1.28)),
        ('Alpine', alpine, (-10, 100)),
    ]
    dim = 30
    runs = 30
    pop_size = 30
    max_iter = 100
    results = []

    for name, func, bound_vals in functions:
        print(f"\n Benchmark name {name}")
        bounds = [bound_vals for _ in range(dim)]
        results = []

        for r in range(runs):
            optimizer = pso_mbo(
                func,
                bounds,
                pop_size,
                max_iter
            )
            _, best_fit = optimizer.optimize()
            results.append(best_fit)

        mean_val = np.mean(results)
        std_val = np.std(results)
        min_val = np.max(results)
        max_val = np.min(results)

        print(f"Mean Values: {mean_val:.4e}")
        print(f"Std Values: {std_val:.4e}")
        print(f"Min Values: {min_val:.4e}")
        print(f"Max Values: {max_val:.4e}")