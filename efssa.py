import numpy as np
from tqdm import tqdm
from benchmark import benchmark
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

        # Firefly parameters (Eq 16 & 17)
        self.beta0 = 1.0
        self.gamma = 1.0
        self.alpha = 0.2

    def initialization(self):
        # Initialize within bounds
        X = self.lb + np.random.rand(self.n, self.dim) * (self.ub - self.lb)
        return X

    def obj_func(self, val):
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

            # Maximizing Coverage
            return 1.0 - coverage_rate

        else:
            obj_func = benchmark(self.lb, self.ub, self.dim)
            method_name = f"{self.func_name}_function"
            try:
                func_to_call = getattr(obj_func, method_name)
                f_name = func_to_call(val)
            except AttributeError:
                f_name = np.sum(val**2)
            return f_name

    def elite_reverse_strategy(self):
        # Corresponds to Eq 12 and 13 in the paper

        # Calculate dynamic bounds a_j (min) and b_j (max) for each dimension
        a_j = np.min(self.population, axis=0)
        b_j = np.max(self.population, axis=0)

        # Eq 13: X* = k(a + b) - X
        k = np.random.rand(self.n, self.dim)
        reverse_population = k * (a_j + b_j) - self.population

        # Check boundaries
        reverse_population = np.clip(reverse_population, self.lb, self.ub)

        # Calculate fitness for reverse population
        reverse_fitness = np.array([self.obj_func(x) for x in reverse_population])

        # Eq 12: Select best N from 2N (Elitism)
        combined_pop = np.vstack((self.population, reverse_population))
        combined_fit = np.concatenate((self.fitness, reverse_fitness))

        # Sort ascending (minimization problem)
        sorted_indices = np.argsort(combined_fit)

        self.population = combined_pop[sorted_indices[:self.n]]
        self.fitness = combined_fit[sorted_indices[:self.n]]

    def firefly_move(self, current, target):
        current_pos = self.population[current]
        target_pos = self.population[target]

        # Eq 15: Distance
        distance = np.linalg.norm(current_pos - target_pos)

        # Eq 14 & 16: Attraction
        attraction_beta = self.beta0 * np.exp(-self.gamma * (distance**2))

        # Eq 17: Movement with random walk
        random_walk = self.alpha * (np.random.rand(self.dim) - 0.5)
        new_pos = current_pos + attraction_beta * (target_pos - current_pos) + random_walk

        return np.clip(new_pos, self.lb, self.ub)

    def update_producers(self, c_pos, fitness, iter_max, pd_count, current_iter):
        # Updates position of the Finders (Eq 8)
        st = self.params['st'] # Warning threshold (0.8)

        new_c_pos = c_pos.copy()
        new_fitness = fitness.copy()

        for i in range(pd_count):
            R2 = np.random.rand()
            current_p = c_pos[i, :].copy()

            if R2 < st:
                # Safe mode: extensive search
                alpha = np.random.rand()
                exponent = - (current_iter + 1) / (alpha * iter_max)
                candidate_pos = current_p * np.exp(exponent)
            else:
                # Danger mode: Random walk / move to safe area
                Q = np.random.normal()
                L = np.ones(self.dim)
                candidate_pos = current_p + Q * L

            # Bound check
            candidate_pos = np.clip(candidate_pos, self.lb, self.ub)

            # Only update if the new position is better
            candidate_fit = self.obj_func(candidate_pos)

            if candidate_fit < fitness[i]:
                new_c_pos[i, :] = candidate_pos
                new_fitness[i] = candidate_fit
            else:
                # Keep old position
                new_c_pos[i, :] = current_p
                new_fitness[i] = fitness[i]

        return new_c_pos, new_fitness

    def update_scroungers(self, c_pos, fitness, pd_count, global_best_position, global_worst_position):
        # Updates position of the Joiners (Eq 10)

        new_c_pos = c_pos.copy()
        new_fitness = fitness.copy()

        for i in range(pd_count, self.n):
            current_p = c_pos[i, :].copy()

            if i > self.n / 2:
                Q = np.random.randn()
                numerator = global_worst_position - current_p
                exponent = numerator / ((i**2) + 1e-10)
                candidate_pos = Q * np.exp(exponent)
            else:
                diff = np.abs(current_p - global_best_position)
                direction = np.random.choice([-1, 1], size=self.dim)
                candidate_pos = global_best_position + diff * direction * (1.0 / self.dim)

            # Bound check
            candidate_pos = np.clip(candidate_pos, self.lb, self.ub)

            # GREEDY SELECTION
            candidate_fit = self.obj_func(candidate_pos)

            if candidate_fit < fitness[i]:
                new_c_pos[i, :] = candidate_pos
                new_fitness[i] = candidate_fit
            else:
                new_c_pos[i, :] = current_p
                new_fitness[i] = fitness[i]

        return new_c_pos, new_fitness

    def danger_aware(self, c_pos, fitness_value, sd_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
        # Implements "Detection and early warning" with Firefly Strategy
        epsilon = self.params['epsilon']
        best_index = np.argmin(fitness_value)

        # Randomly select sparrows to be aware of danger
        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i]
            X_i = c_pos[i, :].copy()
            candidate_pos = X_i.copy()

            if f_i > global_best_fitness:
                # Use Firefly move to jump out of local optimum
                candidate_pos = self.firefly_move(current=i, target=best_index)

            elif np.abs(f_i - global_best_fitness) < 1e-9:
                # Move away from worst
                K = np.random.uniform(-1, 1)
                numerator = np.abs(X_i - global_worst_position)
                denominator = (f_i - global_worst_fitness) + epsilon
                candidate_pos = X_i + K * (numerator / denominator)

            # Bound check
            candidate_pos = np.clip(candidate_pos, self.lb, self.ub)

            # Greedy Check
            new_fit = self.obj_func(candidate_pos)
            if new_fit < f_i:
                c_pos[i, :] = candidate_pos
                fitness_value[i] = new_fit

        return c_pos

    def run(self):
        pd_percent = self.params.get('pd_percent', 0.2)
        sd_percent = self.params.get('sd_percent', 0.15)
        pd_count = int(self.n * pd_percent)
        sd_count = int(self.n * sd_percent)
        convergence_curve = []

        # 1. Initialize
        self.population = self.initialization()
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # Track Global Best History (Strictly Monotonic)
        self.global_best_fitness = np.min(self.fitness)
        self.global_best_position = self.population[np.argmin(self.fitness)].copy()

        for t in tqdm(range(self.max_iter), desc=f"EFSSA Progress"):

            # 2. Elite Reverse Strategy
            self.elite_reverse_strategy()

            # Sort population
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            # Update global best if the elite strategy found something better
            if self.fitness[0] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[0]
                self.global_best_position = self.population[0].copy()

            # Get iteration worst for equations
            global_worst_position = self.population[-1, :].copy()

            # 3. Update Producers (Finders)
            self.population, self.fitness = self.update_producers(self.population, self.fitness, self.max_iter, pd_count, t)

            # 4. Update Scroungers (Joiners)
            self.population, self.fitness = self.update_scroungers(self.population, self.fitness, pd_count, self.global_best_position, global_worst_position)

            # Update stats before warning phase
            current_iter_best = np.min(self.fitness)
            if current_iter_best < self.global_best_fitness:
                self.global_best_fitness = current_iter_best
                self.global_best_position = self.population[np.argmin(self.fitness)].copy()

            current_global_worst_fitness = np.max(self.fitness)
            current_global_worst_position = self.population[np.argmax(self.fitness)].copy()

            # 5. Danger Aware (Firefly Strategy)
            self.population = self.danger_aware(self.population, self.fitness, sd_count, self.global_best_fitness, self.global_best_position, current_global_worst_fitness, current_global_worst_position)

            # Final check for this iteration
            current_iter_best = np.min(self.fitness)
            if current_iter_best < self.global_best_fitness:
                self.global_best_fitness = current_iter_best
                self.global_best_position = self.population[np.argmin(self.fitness)].copy()

            convergence_curve.append(self.global_best_fitness)

        return self.global_best_fitness, self.global_best_position, convergence_curve