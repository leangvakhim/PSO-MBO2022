import numpy as np
import copy
import random
from tqdm import tqdm

class pso_mbo:
    def __init__(self, obj_func, bounds, pop_size, max_iter,
                 peri=1.2, p=5/12, bar=5/12, w=0.9, c1=2.0, c2=2.0):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter

        # mbo parameters
        self.peri = peri
        self.p = p
        self.bar = bar

        # pso parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.population = np.zeros((self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)

        # pso memory
        self.pbest_pos = np.zeros((self.pop_size, self.dim))
        self.pbest_val = np.full(self.pop_size, float('inf'))
        self.gbest_pos = np.zeros(self.dim)
        self.gbest_val = float('inf')

        self.initialize_population()

    def initialize_population(self):
        for i in range(self.pop_size):
            for d in range(self.dim):
                self.population[i, d] = np.random.uniform(self.bounds[d, 0], self.bounds[d, 1])
                self.velocities[i, d] = np.random.uniform(-1, 1)

            self.fitness[i] = self.obj_func(self.population[i])

            self.pbest_pos[i] = self.population[i].copy()
            self.pbest_val[i] = self.fitness[i]

            if self.fitness[i] < self.gbest_val:
                self.gbest_val = self.fitness[i]
                self.gbest_pos = self.population[i].copy()

    def update_pso_memory(self, idx, position, fit_val):
        # update personal best
        if fit_val < self.pbest_val[idx]:
            self.pbest_val[idx] = fit_val
            self.pbest_pos[idx] = position.copy()

            # update global best
            if fit_val < self.gbest_val:
                self.gbest_val = fit_val
                self.gbest_pos = position.copy()

    def optimize(self):
        n1 = int(np.ceil(self.p * self.pop_size))
        n2 = self.pop_size - n1

        for t in tqdm(range(self.max_iter), desc="PSO-MBO Progress: "):

            sorted_indices = np.argsort(self.fitness)
            sorted_pop = self.population[sorted_indices]

            new_pop = np.zeros_like(self.population)

            current_w = self.w - (self.w - 0.4) * (t / self.max_iter)

            for i in range(self.pop_size):
                original_idx = sorted_indices[i]
                r1, r2 = np.random.rand(), np.random.rand()
                velo_new = (current_w * self.velocities[original_idx] +
                            self.c1 * r1 * (self.pbest_pos[original_idx] - self.population[original_idx]) +
                            self.c2 * r2 * (self.gbest_pos - self.population[original_idx]))

                self.velocities[original_idx] = velo_new

                x_new_mbo = np.zeros(self.dim)

                if i < n1:
                    # migration operator
                    for k in range(self.dim):
                        r = np.random.rand() * self.peri
                        if r <= self.p:
                            r_idx = np.random.randint(0, n1)
                            x_new_mbo[k] = sorted_pop[r_idx, k]
                        else:
                            r_idx = np.random.randint(n1, self.pop_size)
                            x_new_mbo[k] = sorted_pop[r_idx, k]
                else:
                    # butterfly adjusting operator (bao)
                    for k in range(self.dim):
                        rand_val = np.random.rand()
                        if rand_val <= self.p:
                            x_new_mbo[k] = self.gbest_pos[k]
                        else:
                            r_idx = np.random.randint(n1, self.pop_size)
                            x_new_mbo[k] = sorted_pop[r_idx, k]

                        # levy flight
                        if rand_val > self.bar:
                            dx = 0.05
                            x_new_mbo[k] = x_new_mbo[k] + (dx * (np.random.rand() - 0.05))

                x_new_pso = self.population[original_idx] + self.velocities[original_idx]

                x_new_mbo = np.clip(x_new_mbo, self.bounds[:, 0], self.bounds[:, 1])
                x_new_pso = np.clip(x_new_pso, self.bounds[:, 0], self.bounds[:, 1])

                fit_mbo = self.obj_func(x_new_mbo)
                fit_pso = self.obj_func(x_new_pso)

                if fit_mbo < fit_pso:
                    chosen_pos = x_new_mbo
                    chosen_fit = fit_mbo
                else:
                    chosen_pos = x_new_pso
                    chosen_fit = fit_pso

                if chosen_fit < self.fitness[original_idx]:
                    new_pop[original_idx] = chosen_pos
                    self.fitness[original_idx] = chosen_fit
                else:
                    new_pop[original_idx] = self.population[original_idx]

                self.update_pso_memory(original_idx, chosen_pos, chosen_fit)

            self.population = new_pop

        return self.gbest_pos, self.gbest_val