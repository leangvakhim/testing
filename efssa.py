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
        # 1. Define Elite Group (e.g., all current individuals or top percentage)
        # The paper implies using the whole population sorted or a subset.
        # Here we use the whole population as "Elite" source for bounds a_j, b_j

        # Calculate dynamic bounds a_j (min) and b_j (max) for each dimension
        # Shape: (1, dim)
        a_j = np.min(self.population, axis=0)
        b_j = np.max(self.population, axis=0)

        # Eq 13: X* = k(a + b) - X
        k = np.random.rand(self.n, self.dim)
        reverse_population = k * (a_j + b_j) - self.population

        # Check boundaries (clip to problem bounds lb, ub)
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

    # FIX 1: Pass current_iter and use it in the formula
    def update_producers(self, c_pos, iter_max, pd_count, current_iter):
        R2 = np.random.rand()
        L = np.ones(self.dim)
        st = self.params['st']

        for i in range(pd_count):
            alpha = np.random.rand() # Alpha should be random per update
            Q = np.random.normal() # Q is random normal

            if R2 < st:
                # FIX: Use current_iter instead of loop index i
                exponent = - (current_iter + 1) / (alpha * iter_max)
                c_pos[i, :] = c_pos[i, :] * np.exp(exponent)
            else:
                c_pos[i, :] = c_pos[i, :] + Q * L
        return c_pos

    def update_scroungers(self, c_pos, pd_count, global_best_position, global_worst_position):
        L = np.ones((1, self.dim))
        for i in range(pd_count, self.n):
            if i > self.n / 2:
                Q = np.random.randn()
                exponent_denominator = (i - pd_count + 1) ** 2 # Avoid div by zero or huge scaling? Paper says i^2
                # Standard SSA implementation usually uses just i^2 or similar
                exponent_numerator = global_worst_position - c_pos[i, :]
                exponent = exponent_numerator / (i**2)
                c_pos[i, :] = Q * np.exp(exponent)
            else:
                # Eq 10 part 2
                A = np.ones((1, self.dim))
                rand_indices = np.random.rand(self.dim) < 0.5
                A[0, rand_indices] = -1

                # A+ = A.T * (A * A.T)^(-1). Since A is 1xD with +/-1, A*A.T = dim.
                # So A+ = A.T / dim
                # Term is: |X - Xbest| * A+ * L
                # Dimensions: (1xD) * (Dx1) * (1xD)

                diff = np.abs(c_pos[i, :] - global_best_position)
                # Dot product (1xD) * (Dx1) results in scalar
                term1 = np.dot(diff, A.T) / self.dim

                step_simplified = term1 * L # Scalar * Vector of ones
                c_pos[i, :] = global_best_position + step_simplified
        return c_pos

    def danger_aware(self, c_pos, fitness_value, sd_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
        epsilon = self.params['epsilon']
        best_index = np.argmin(fitness_value)

        # Randomly select sparrows to be aware of danger
        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i]
            X_i = c_pos[i, :]

            if f_i > global_best_fitness:
                c_pos[i, :] = self.firefly_move(current=i, target=best_index)

            # Middle sparrow (is the best) -> Moves away from itself (Exploration)
            # Use floating point tolerance for equality check
            elif np.abs(f_i - global_best_fitness) < 1e-9:
                K = np.random.uniform(-1, 1)
                numerator = np.abs(X_i - global_worst_position)
                denominator = (f_i - global_worst_fitness) + epsilon
                c_pos[i, :] = X_i + K * (numerator / denominator)
        return c_pos

    def run(self):
        pd_percent = 0.2
        pd_count = int(self.n * pd_percent)
        sd_percent = 0.1 # Usually 0.1 or 0.2
        sd_count = int(self.n * sd_percent)
        convergence_curve = []

        # 1. Initialize
        self.population = self.initialization()

        # Initial Fitness
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # Apply Elite Reverse Strategy on Initialization (As per some papers)
        # OR applied iteratively. The paper Algorithm 1 places it INSIDE the loop.
        # We will keep it inside the loop as per your original structure.

        # Main Loop
        for t in tqdm(range(self.max_iter), desc="EFSSA Progress: "):

            # 2. Elite Reverse Strategy (Applied every iteration in Alg 1)
            self.elite_reverse_strategy()

            # Sort population after Elite Strategy
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            # Get Bests/Worsts
            current_global_best_position = self.population[0, :].copy()
            current_global_best_fitness = self.fitness[0]
            global_worst_position = self.population[-1, :].copy()
            current_global_worst_fitness = self.fitness[-1]

            # 3. Update Producers
            # FIX: Pass 't' (current iteration)
            self.population = self.update_producers(self.population, self.max_iter, pd_count, t)

            # FIX 2: Update Fitness for PRODUCERS immediately after moving
            for i in range(0, pd_count):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # 4. Update Scroungers
            self.population = self.update_scroungers(self.population, pd_count, current_global_best_position, global_worst_position)

            # Update Fitness for Scroungers
            for i in range(pd_count, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # Recalculate bests needed for danger aware?
            # Usually we use the values from the start of the iter, but updating them might help
            current_global_best_fitness = np.min(self.fitness)
            current_global_best_position = self.population[np.argmin(self.fitness)].copy()
            current_global_worst_fitness = np.max(self.fitness)
            current_global_worst_position = self.population[np.argmax(self.fitness)].copy()

            # 5. Danger Aware (Firefly)
            self.danger_aware(self.population, self.fitness, sd_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)

            # Final Fitness Update for everyone (since danger aware moved random ones)
            for i in range(self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # Record Best
            current_best = np.min(self.fitness)
            current_best_pos = self.population[np.argmin(self.fitness)].copy()
            convergence_curve.append(current_best)

        return current_best, current_best_pos, convergence_curve