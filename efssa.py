import numpy as np
from tqdm import tqdm
from benchmark import benchmark
import copy
from coverage import coverage

class efssa():
    def __init__(self, lb, ub, dim, n, max_iter, params, func_name):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n = n
        self.max_iter = max_iter
        self.params = params
        self.func_name = func_name

        # Firefly parameters
        self.beta0 = 1.0
        self.gamma = 1.0
        self.alpha = 0.2

    def initialization(self):
        # Initialize within bounds
        X = self.lb + np.random.rand(self.n, self.dim) * (self.ub - self.lb)
        return X

    def obj_func(self, val):
        # Clip value to bounds before evaluation to ensure validity
        val = np.clip(val, self.lb, self.ub)

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
        # Calculate dynamic bounds a_j (min) and b_j (max) for each dimension
        # Note: The paper suggests using the "Elite" group (top p%) for bounds.
        # Using the whole population is a valid simplification but consider
        # filtering self.population[:elite_count] if strictly following text.
        a_j = np.min(self.population, axis=0)
        b_j = np.max(self.population, axis=0)

        # Eq 13: X* = k(a + b) - X
        k = np.random.rand(self.n, self.dim)
        reverse_population = k * (a_j + b_j) - self.population

        # Check boundaries
        reverse_population = np.clip(reverse_population, self.lb, self.ub)

        # Calculate fitness
        reverse_fitness = np.array([self.obj_func(x) for x in reverse_population])

        # Eq 12: Select best N from 2N
        combined_pop = np.vstack((self.population, reverse_population))
        combined_fit = np.concatenate((self.fitness, reverse_fitness))

        # Sort ascending (minimization problem)
        sorted_indices = np.argsort(combined_fit)

        self.population = combined_pop[sorted_indices[:self.n]]
        self.fitness = combined_fit[sorted_indices[:self.n]]

    def firefly_move(self, current, target):
        current_pos = self.population[current]
        target_pos = self.population[target]
        # Eq 15
        distance = np.linalg.norm(current_pos - target_pos)
        # Eq 14
        attraction_beta = self.beta0 * np.exp(-self.gamma * (distance**2))
        # Eq 17
        random_walk = self.alpha * (np.random.rand(self.dim) - 0.5)
        new_pos = current_pos + attraction_beta * (target_pos - current_pos) + random_walk
        return np.clip(new_pos, self.lb, self.ub)

    def update_producers(self, c_pos, iter_max, pd_count, current_iter):
        R2 = np.random.rand()
        L = np.ones(self.dim)
        st = self.params['st']

        for i in range(pd_count):
            alpha = np.random.rand()
            Q = np.random.normal()

            if R2 < st:
                # Producer update: X * exp(-i / (alpha * T))
                exponent = - (current_iter + 1) / (alpha * iter_max)
                c_pos[i, :] = c_pos[i, :] * np.exp(exponent)
            else:
                c_pos[i, :] = c_pos[i, :] + Q * L
        return c_pos

    def update_scroungers(self, c_pos, pd_count, global_best_position, global_worst_position):
        for i in range(pd_count, self.n):
            if i > self.n / 2:
                Q = np.random.randn()
                exponent = (global_worst_position - c_pos[i, :]) / (i**2)
                c_pos[i, :] = Q * np.exp(exponent)
            else:
                diff = np.abs(c_pos[i, :] - global_best_position)

                # FIX 2: Remove (1.0 / self.dim) to allow larger convergence steps
                # Standard SSA typically uses just direction * diff
                direction = np.random.choice([-1, 1], size=self.dim)
                c_pos[i, :] = global_best_position + diff * direction

        return c_pos

    def danger_aware(self, c_pos, fitness_value, sd_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
        epsilon = self.params['epsilon']
        best_index = np.argmin(fitness_value)
        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i]
            X_i = c_pos[i, :].copy()

            if f_i > global_best_fitness:
                # Firefly strategy for edge sparrows (Paper Section 4.2.2)
                new_pos = self.firefly_move(current=i, target=best_index)
                new_fit = self.obj_func(new_pos)
                if new_fit < f_i:
                    c_pos[i, :] = new_pos
                    fitness_value[i] = new_fit

            elif np.abs(f_i - global_best_fitness) < 1e-9:
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

        # 1. Initialize
        self.population = self.initialization()
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # FIX 1: Apply Elite Reverse Strategy ONCE at initialization
        # (Paper Section 4.2.1: "added to the position initialization process")
        self.elite_reverse_strategy()

        # Sort population after Elite Strategy
        sorted_indices = np.argsort(self.fitness)
        self.population = self.population[sorted_indices]
        self.fitness = self.fitness[sorted_indices]

        # Main Loop
        for t in tqdm(range(self.max_iter), desc="EFSSA Progress: "):

            # --- REMOVED elite_reverse_strategy() from inside the loop ---

            current_global_best_position = self.population[0, :].copy()
            current_global_best_fitness = self.fitness[0]
            global_worst_position = self.population[-1, :].copy()

            # 3. Update Producers
            self.population = self.update_producers(self.population, self.max_iter, pd_count, t)

            for i in range(0, pd_count):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # 4. Update Scroungers
            self.population = self.update_scroungers(self.population, pd_count, current_global_best_position, global_worst_position)

            for i in range(pd_count, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # Recalculate bests for danger aware
            current_global_best_fitness = np.min(self.fitness)
            current_global_best_position = self.population[np.argmin(self.fitness)].copy()
            current_global_worst_fitness = np.max(self.fitness)
            current_global_worst_position = self.population[np.argmax(self.fitness)].copy()

            # 5. Danger Aware (Firefly)
            self.danger_aware(self.population, self.fitness, sd_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)

            # Final Fitness Update
            for i in range(self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            current_min_fit = np.min(self.fitness)
            current_min_idx = np.argmin(self.fitness)

            if current_min_fit > current_global_best_fitness:
                worst_idx = np.argmax(self.fitness)
                self.population[worst_idx, :] = current_global_best_position.copy()
                self.fitness[worst_idx] = current_global_best_fitness
            else:
                current_global_best_fitness = current_min_fit
                current_global_best_position = self.population[current_min_idx, :].copy()

            convergence_curve.append(current_global_best_fitness)

        return current_global_best_fitness, current_global_best_position, convergence_curve