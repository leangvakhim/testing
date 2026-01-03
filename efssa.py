import numpy as np
from tqdm import tqdm
from benchmark import benchmark
import copy

class efssa():
    def __init__(self, lb, ub, dim, n, max_iter, params, func_name):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n = n
        self.max_iter = max_iter
        self.params = params
        self.func_name = func_name

        self.beta0 = 1.0
        self.gamma = 1.0
        self.alpha = 0.2

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

    def elite_reverse_strategy(self):
        # eq 13
        k = np.random.rand(self.n, self.dim)
        reverse_population = k * (self.lb + self.ub) - self.population
        reverse_population = np.clip(reverse_population, self.lb, self.ub)
        reverse_fitness = np.array([self.obj_func(x) for x in reverse_population])

        # eq 12
        combined_pop = np.vstack((self.population, reverse_population))
        combined_fit = np.concatenate((self.fitness, reverse_fitness))
        sorted_indices = np.argsort(combined_fit)[::-1]
        self.population = combined_pop[sorted_indices[:self.n]]
        self.fitness = combined_fit[sorted_indices][:self.n]

        if self.fitness[0] > self.global_best_fit:
            self.global_best_fit = self.fitness[0]
            self.global_best_pos = copy.deepcopy(self.population[0])

    def firefly_move(self, current, target):
        current_pos = self.population[current]
        target_pos = self.population[target]
        # eq 15
        distance = np.linalg.norm(current_pos - target_pos)
        # eq 14
        attraction_beta = self.beta0 * np.exp(-self.gamma * (distance**2))
        # eq 17
        random_walk = self.alpha * (np.random.rand(self.dim) - 0.5)
        new_pos = current_pos + attraction_beta * (target_pos - current_pos) + random_walk
        return np.clip(new_pos, self.lb, self.ub)

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

    # def update_producers(self, c_pos, iter_max, pd_count, current_iter):
    #     R2 = np.random.rand()
    #     L = np.ones(self.dim)
    #     Q = np.random.normal()
    #     st = self.params['st']
    #     alpha = np.random.rand()

    #     for i in range(pd_count):
    #         if R2 < st:
    #             exponent = - (current_iter + 1) / (alpha * iter_max)
    #             c_pos[i, :] = c_pos[i, :] * np.exp(exponent)
    #         else:
    #             c_pos[i, :] = c_pos[i, :] + Q * L
    #     return c_pos

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

        best_index = np.argmin(fitness_value)

        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i] # Current sparrow's fitness
            X_i = c_pos[i, :] # Current sparrow's position

            # Sparrow is at the edge
            if f_i > global_best_fitness:
                c_pos[i, :] = self.firefly_move(current=i, target=best_index)
            # Sparrow is at the middle (the best)
            else:
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

        # 1. Initialize population (Uncommented and assigned to self.population)
        self.population = self.initialization()

        # 2. Calculate initial fitness for the population
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # 3. Initialize global bests (Required for elite_reverse_strategy)
        self.global_best_fit = np.min(self.fitness)
        self.global_best_pos = self.population[np.argmin(self.fitness)].copy()

        prev_best_fitness = self.global_best_fit
        prev_best_pos = self.global_best_pos.copy()
        current_best_pos = self.global_best_pos.copy()

        # 4. Now it is safe to call elite_reverse_strategy
        self.elite_reverse_strategy()

        # Re-evaluate best after elite strategy
        current_best_fit_after_elite = np.min(self.fitness)
        if current_best_fit_after_elite < prev_best_fitness:
            self.global_best_fit = current_best_fit_after_elite
            self.global_best_pos = self.population[np.argmin(self.fitness)].copy()
            prev_best_fitness = self.global_best_fit
            prev_best_pos = self.global_best_pos.copy()

        # Main Loop
        for t in tqdm(range(self.max_iter), desc="EFSSA Progress: "):
            current_best = np.min(self.fitness)
            current_best_index = np.argmin(self.fitness)
            current_worst = np.max(self.fitness)
            current_worst_index = np.argmax(self.fitness)

            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_best_pos.copy()
            else:
                prev_best_pos = self.population[np.argmin(self.fitness)].copy()

            # Sort population
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            current_global_best_position = self.population[0, :].copy() # Best is at index 0 after sort
            current_global_best_fitness = self.fitness[0]
            current_global_worst_position = self.population[-1, :].copy()
            current_global_worst_fitness = self.fitness[-1]

            global_worst_position = self.population[current_worst_index, :].copy()

            # Equation 3 - Update producers
            # Using the return value version (see part 2 below)
            self.population = self.update_producers(self.population, self.max_iter, pd_count)
            # self.population = self.update_producers(self.population, self.max_iter, pd_count, t)

            for i in range(pd_count, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # Equation 4 - Update scroungers
            self.population = self.update_scroungers(self.population, pd_count, current_global_best_position, global_worst_position)

            for i in range(pd_count, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            current_best = np.min(self.fitness)
            current_best_idx = np.argmin(self.fitness)
            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = self.population[current_best_idx].copy()

            # Equation 5 - Danger aware sparrows
            # Note: danger_aware usually modifies in place, but can be updated to return
            self.danger_aware(self.population, self.fitness, sd_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)

            for i in range(0, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            current_best_index = np.argmin(self.fitness)
            if self.fitness[current_best_index] < current_best:
                current_best = self.fitness[current_best_index]
                current_best_pos = self.population[current_best_index, :].copy()

            convergence_curve.append(current_best)

        return current_best, current_best_pos, convergence_curve


