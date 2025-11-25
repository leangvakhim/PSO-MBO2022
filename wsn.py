import numpy as np
import matplotlib.pyplot as plt

class wsn:
    def __init__(self, n_sensor=50, area_size=100, sensing_radius=10, w1=1.0, w2=0.0, w3=0.0):
        self.n_sensor = n_sensor
        self.area_size = area_size
        self.sensing_radius = sensing_radius

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.grid_res = 1.0
        self.grid_w = int(area_size / self.grid_res)
        self.grid_h = int(area_size / self.grid_res)

        x = np.linspace(0, area_size, self.grid_w)
        y = np.linspace(0, area_size, self.grid_h)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))

        self.node_energy = np.ones(self.n_sensor)

    def coverage_objective(self, flat_position):
        sensors = flat_position.reshape((self.n_sensor, 2))

        is_covered = np.zeros(len(self.grid_points), dtype=bool)

        for sensor in sensors:
            dx = self.grid_points[:, 0] - sensor[0]
            dy = self.grid_points[:, 1] - sensor[1]
            dist_sq = dx**2 + dy**2

            mask = dist_sq <= self.sensing_radius**2
            is_covered |= mask

        lambda_cov = np.sum(is_covered) / len(self.grid_points)

        S1 = self.n_sensor
        S2 = self.n_sensor
        theta = S2 / S1

        max_e = np.max(self.node_energy)
        min_e = np.min(self.node_energy)
        avg_e = np.mean(self.node_energy)

        if avg_e > 0:
            eta = (max_e - min_e) / avg_e
        else:
            eta = 0.0

        fitness = (self.w1 * lambda_cov) + (self.w2 * theta) + (self.w3 * eta)

        return 1.0 - fitness

    def visualize(self, best_solution, title="WSN Coverage"):
        final_loss = self.coverage_objective(best_solution)
        final_fitness = 1.0 - final_loss

        sensors = best_solution.reshape((self.n_sensor, 2))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_aspect('equal')

        ax.scatter(sensors[:, 0], sensors[:, 1], c='red', marker='.', zorder=10, label='Sensors')

        for sensor in sensors:
            circle = plt.Circle((sensor[0], sensor[1]), self.sensing_radius, color='blue', alpha=0.15, zorder=1)
            ax.add_artist(circle)
            circle_outline = plt.Circle((sensor[0], sensor[1]), self.sensing_radius, color='blue', fill=False, alpha=0.5, zorder=2)
            ax.add_artist(circle_outline)

        plt.title(f"{title}\nTotal Coverage: {final_fitness:.4f}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()