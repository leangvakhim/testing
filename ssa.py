import numpy as np
from tqdm import tqdm
from benchmark import benchmark
from coverage import coverage

class ssa():
    def __init__(self, lb, ub, dim, n, max_iter, params, func_name):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n = n
        self.max_iter = max_iter
        self.params = params
        self.func_name = func_name

    def initialization(self):
        X = self.lb + np.random.rand(self.n, self.dim) * (self.ub - self.lb)
        return X

    def obj_func(self, val):
        if self.func_name == 'coverage_optimization':
            w = self.params['w']
            h = self.params['h']
            num_nodes = self.params['num_nodes']
            sensing_radius = self.params['sensing_radius']
            r_error = self.params['r_error']
            pos_reshaped = val.reshape(num_nodes, 2)
            cov = coverage(w, h, num_nodes, sensing_radius, r_error, pos_reshaped)
            coverage_rate = cov.calculate_probabilistics_coverage()
            return 1.0 - coverage_rate
        else:
            obj_func = benchmark(self.lb, self.ub, self.dim)
            method_name = f"{self.func_name}_function"
            func_to_call = getattr(obj_func, method_name)
            f_name = func_to_call(val)
        return f_name

    def update_producers(self, c_pos, iter_max, pd_count):
        R2 = np.random.rand()
        L = np.ones(self.dim)
        Q = np.random.normal()
        st = self.params['st']
        alpha = np.random.rand()

        for i in range(pd_count, self.n):
            if R2 < st:
                exponent = - (i + 1) / (alpha * iter_max)
                c_pos[i, :] = c_pos[i, :] * np.exp(exponent)
            else:
                c_pos[i, :] = c_pos[i, :] + Q * L
        return c_pos

    def update_scroungers(self, c_pos, pd_count, global_best_position, global_worst_position):
        L = np.ones((1, self.dim))
        for i in range(pd_count, self.n):
            if i > self.n / 2:
                Q = np.random.randn()
                exponent_denominator = i ** 2
                exponent_numerator = global_worst_position - c_pos[i, :]
                exponent = exponent_numerator / exponent_denominator
                c_pos[i, :] = Q * np.exp(exponent)
            else:
                A = np.ones((1, self.dim))
                rand_indices = np.random.rand(self.dim) < 0.5
                A[0, rand_indices] = -1
                diff = np.abs(c_pos[i, :] - global_best_position)
                C = np.sum(diff * A) / self.dim
                step_simplified = C * L
                c_pos[i, :] = global_best_position + step_simplified
        return c_pos

    def danger_aware(self, c_pos, fitness_value, sd_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
        epsilon = self.params['epsilon']

        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i] # Current sparrow's fitness
            X_i = c_pos[i, :] # Current sparrow's position

            # Sparrow is at the edge
            if f_i > global_best_fitness:
                beta = np.random.randn()
                c_pos[i, :] = global_best_position + beta * np.abs(X_i - global_best_position)
            # Sparrow is at the middle (the best)
            elif f_i == global_best_fitness:
                K = np.random.uniform(-1, 1)
                numerator = np.abs(X_i - global_worst_position)
                denominator = (f_i - global_worst_fitness) + epsilon
                c_pos[i, :] = X_i + K * (numerator / denominator)

        return c_pos

    def run(self):
        pd_percent = 0.2
        pd_count = int(self.n * pd_percent)
        sd_percent = 0.1
        sd_count = int(self.n * sd_percent)
        convergence_curve = []

        current_pos = self.initialization()
        list_fitness = []
        for i in range(0, self.n):
            fitness = self.obj_func(current_pos[i])
            list_fitness.append(fitness)

        prev_best_fitness = np.min(list_fitness)
        prev_best_pos = current_pos[np.argmin(list_fitness)].copy()
        current_best_pos = prev_best_pos.copy()

        for t in tqdm(range(self.max_iter), desc="SSA Progress: "):
            current_best = np.min(list_fitness)
            current_best_index = np.argmin(list_fitness)
            current_worst = np.max(list_fitness)
            current_worst_index = np.argmax(list_fitness)

            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_best_pos.copy()
            else:
                # Keep tracking the global best
                prev_best_pos = current_pos[np.argmin(list_fitness)].copy()

            sorted_indices = np.argsort(list_fitness)
            current_pos = current_pos[sorted_indices]
            list_fitness = np.array(list_fitness)
            list_fitness = list_fitness[sorted_indices]

            current_global_best_position = current_pos[current_best_index, :].copy()
            current_global_best_fitness = list_fitness[current_best_index]
            current_global_worst_position = current_pos[current_worst_index, :].copy()
            current_global_worst_fitness = list_fitness[current_worst_index]

            global_worst_position = current_pos[current_worst_index, :].copy()

            # Equation 3 - Update producers
            current_pos = self.update_producers(current_pos, self.max_iter, pd_count)
            # current_pos = self.update_producers(current_pos, current_best_index, self.max_iter, st, pd_count, self.dim)
            for i in range(pd_count, self.n):
                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

            # Equation 4 - Update scroungers
            # update_scroungers(self, c_pos, pd_count, global_best_position, global_worst_position):
            current_pos = self.update_scroungers(current_pos, pd_count, current_global_best_position, global_worst_position)
            # current_pos = self.update_scroungers(current_pos, current_best_index, pd_count, self.max_iter, self.n, self.dim, current_global_best_position, global_worst_position)
            for i in range(pd_count, self.n):
                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

            current_best = np.min(list_fitness)
            current_best_idx = np.argmin(list_fitness)
            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_pos[current_best_idx].copy()

            # Equation 5 - Danger aware sparrows
            current_pos = self.danger_aware(current_pos, list_fitness, sd_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)
            for i in range(0, self.n):
                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

            current_best_index = np.argmin(list_fitness)
            if list_fitness[current_best_index] < current_best:
                current_best = list_fitness[current_best_index]
                current_best_pos = current_pos[current_best_index, :].copy()

            convergence_curve.append(current_best)

        # return prev_best_pos, prev_best_fitness, convergence_curve
        return current_best, current_best_pos, convergence_curve