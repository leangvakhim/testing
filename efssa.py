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

        # Firefly parameters (from Algorithm 1 description & Eq 16, 17)
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

            # Maximizing Coverage = Minimizing (1 - Coverage)
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

    def update_producers(self, c_pos, iter_max, pd_count, current_iter):
        # Updates position of the "Finders" (Eq 8)
        st = self.params['st'] # Warning threshold (usually 0.8)

        for i in range(pd_count):
            R2 = np.random.rand()
            if R2 < st:
                # Safe mode: extensive search
                alpha = np.random.rand()
                exponent = - (current_iter + 1) / (alpha * iter_max)
                c_pos[i, :] = c_pos[i, :] * np.exp(exponent)
            else:
                # Danger mode: Random walk / move to safe area
                Q = np.random.normal()
                L = np.ones(self.dim)
                c_pos[i, :] = c_pos[i, :] + Q * L
        return c_pos

    def update_scroungers(self, c_pos, pd_count, global_best_position, global_worst_position):
        # Updates position of the "Joiners" (Eq 10)
        for i in range(pd_count, self.n):
            if i > self.n / 2:
                # Hungry/Worst case: Move towards optimum logarithmically
                Q = np.random.randn()
                exponent = (global_worst_position - c_pos[i, :]) / (i**2)
                c_pos[i, :] = Q * np.exp(exponent)
            else:
                # Follower case: Move near the best position
                # Implements: X_best + |X - X_best| * A+ * L
                diff = np.abs(c_pos[i, :] - global_best_position)
                direction = np.random.choice([-1, 1], size=self.dim) # Simulates A+
                c_pos[i, :] = global_best_position + diff * direction * (1.0 / self.dim)
        return c_pos

    def danger_aware(self, c_pos, fitness_value, sd_count, global_best_fitness, global_best_position, global_worst_fitness, global_worst_position):
        # Implements "Detection and early warning" with Firefly Strategy (Eq 11 modification + Firefly)
        epsilon = self.params['epsilon']
        best_index = np.argmin(fitness_value)

        # Randomly select sparrows to be aware of danger
        danger_indices = np.random.choice(self.n, sd_count, replace=False)

        for i in danger_indices:
            f_i = fitness_value[i]
            X_i = c_pos[i, :].copy()

            if f_i > global_best_fitness:
                # Use Firefly move to jump out of local optimum
                new_pos = self.firefly_move(current=i, target=best_index)

                # Greedy selection: only update if better
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
        pd_percent = self.params.get('pd_percent', 0.2)
        sd_percent = self.params.get('sd_percent', 0.15)
        pd_count = int(self.n * pd_percent)
        sd_count = int(self.n * sd_percent)
        convergence_curve = []

        # 1. Initialize
        self.population = self.initialization()
        self.fitness = np.array([self.obj_func(x) for x in self.population])

        # Main Loop (Alg 2: while t < Imax)
        for t in tqdm(range(self.max_iter), desc=f"EFSSA Progress"):

            # 2. Elite Reverse Strategy
            self.elite_reverse_strategy()

            # Sort population
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            # Get Bests/Worsts
            current_global_best_position = self.population[0, :].copy()
            current_global_best_fitness = self.fitness[0]
            global_worst_position = self.population[-1, :].copy()

            # 3. Update Producers (Finders)
            self.population = self.update_producers(self.population, self.max_iter, pd_count, t)

            # Recalculate fitness for updated producers
            for i in range(pd_count):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # 4. Update Scroungers (Joiners)
            self.population = self.update_scroungers(self.population, pd_count, current_global_best_position, global_worst_position)

            # Recalculate fitness for updated scroungers
            for i in range(pd_count, self.n):
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
                self.fitness[i] = self.obj_func(self.population[i])

            # Update global stats before warning phase
            current_global_best_fitness = np.min(self.fitness)
            current_global_best_position = self.population[np.argmin(self.fitness)].copy()
            current_global_worst_fitness = np.max(self.fitness)
            current_global_worst_position = self.population[np.argmax(self.fitness)].copy()

            # 5. Danger Aware (Firefly Strategy)
            self.population = self.danger_aware(self.population, self.fitness, sd_count, current_global_best_fitness, current_global_best_position, current_global_worst_fitness, current_global_worst_position)

            current_iter_best_fit = np.min(self.fitness)

            # Elitism Check: If iteration best is worse than global best, restore global best
            if current_iter_best_fit > current_global_best_fitness:
                 # Ensure we carry forward the absolute best found so far
                 worst_idx = np.argmax(self.fitness)
                 self.population[worst_idx, :] = current_global_best_position
                 self.fitness[worst_idx] = current_global_best_fitness
            else:
                 current_global_best_fitness = current_iter_best_fit
                 current_global_best_position = self.population[np.argmin(self.fitness)].copy()

            convergence_curve.append(current_global_best_fitness)

        return current_global_best_fitness, current_global_best_position, convergence_curve