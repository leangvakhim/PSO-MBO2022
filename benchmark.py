import numpy as np
from pso_mbo import pso_mbo

# Sphere Function
def sphere(x):
    return np.sum(x**2)

# Rosenbrock Function
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Rastrigin Function
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Schwefel's Problem 2.22.
def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# Schwefel's Problem 1.2
def schwefel_1_2(x):
    return np.sum(np.cumsum(x)**2)

# Schwefel's Problem 2.21.
def schwefel_2_21(x):
    return np.max(np.abs(x))

# Step Function.
def step(x):
    return np.sum(np.floor(x + 0.5)**2)

# Quartic Function with Noise.
def quartic(x):
    d = len(x)
    indices = np.arange(1, d + 1)
    return np.sum(indices * (x**4)) + np.random.random()

# Alpine Function.
def alpine(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))