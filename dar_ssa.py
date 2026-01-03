import numpy as np
from benchmark import benchmark
from tqdm import tqdm
from coverage import coverage

class darssa():
    def __init__(self, lb, ub, dim, n, max_iter, params, func_name):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n = n
        self.max_iter = max_iter
        self.params = params
        self.func_name = func_name

    def initialize(self):
        x = self.lb + np.random.rand(self.n, self.dim) * (self.ub - self.lb)
        return x

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
            # func_to_call = getattr(obj_func, self.func_name)
            method_name = f"{self.func_name}_function"
            func_to_call = getattr(obj_func, method_name)
            f_name = func_to_call(val)
        return f_name

    def adaptive_role(self, t):
        r_end = self.params['r_end']
        r_start = self.params['r_start']
        dynamic_role_lambda = self.params['dynamic_role_lambda']
        current_role = r_end + (r_start - r_end) * (1 - ((t / self.max_iter) ** dynamic_role_lambda))
        return current_role

    def producer_update(self, c_pos, i):
        r_2 = np.random.rand()
        alpha = np.random.rand()
        L = np.ones(self.dim)
        Q = np.random.normal()
        st = self.params['st']

        if r_2 < st:
            c_pos = c_pos * np.exp(-i / alpha * self.max_iter)
        else:
            c_pos = c_pos + Q * L
        return c_pos

    def update_scroungers(self, current_pos, pd_count, num_sensor, dim, global_best_position, global_worst_position):
        L = np.ones((1, dim))
        for i in range(pd_count, num_sensor):
            if i > num_sensor / 2:
                Q = np.random.randn()
                exponent_denominator = i ** 2
                exponent_numerator = global_worst_position - current_pos[i, :]
                exponent = exponent_numerator / exponent_denominator
                current_pos[i, :] = Q * np.exp(exponent)
            else:
                A = np.ones((1, dim))
                rand_indices = np.random.rand(dim) < 0.5
                A[0, rand_indices] = -1
                diff = np.abs(current_pos[i, :] - global_best_position)
                C = np.sum(diff * A) / dim
                step_simplified = C * L
                current_pos[i, :] = global_best_position + step_simplified
        return current_pos

    def calculate_density_and_repulsion(self, particles, sensing_radius):
        epsilon = self.params['epsilon']

        # Crowding Threshold
        D_th = 2.0 * sensing_radius

        # Compute pairwise distance matrix (N x N)
        # diff[i, j] = particles[i] - particles[j]
        diff = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        # 1. Calculate Density (Equation 3.0.1)
        # Mask for neighbors within threshold (exclude self-loop dist=0)
        neighbor_mask = (dists < D_th) & (dists > 0)

        # Kernel function: 1 - (d/D_th)^2 for neighbors, 0 otherwise
        kernels = np.zeros_like(dists)
        kernels[neighbor_mask] = 1.0 - (dists[neighbor_mask] / D_th) ** 2
        densities = np.sum(kernels, axis=1)

        # 2. Calculate Repulsive Forces (Equation 3.0.2)
        # Force magnitude: exp(-d/Rs)
        force_mags = np.zeros_like(dists)
        force_mags[neighbor_mask] = np.exp(-dists[neighbor_mask] / sensing_radius)

        # Avoid division by zero
        safe_dists = dists.copy()
        safe_dists[safe_dists == 0] = 1.0
        normalized_dirs = diff / safe_dists[:, :, np.newaxis]

        # Force on i from j
        force_contributions = normalized_dirs * force_mags[:, :, np.newaxis]
        total_forces = np.sum(force_contributions, axis=1)

        # 3. Normalize to Unit Vectors (Equation 3.0.3)
        force_norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        nonzero_force = force_norms > 0

        repulsion_vecs = np.zeros_like(total_forces)
        repulsion_vecs = np.divide(total_forces, force_norms + epsilon, where=nonzero_force)
        return densities, repulsion_vecs

    def generate_da_r_fbs_candidate(self, particles, densities, repulsion_vecs, base_amplitude):
        gamma = self.params['gamma']  # Density Gain
        omega = self.params['omega']  # Repulsion Weight

        # 1. Calculate Adaptive Amplitude (Eq 3.2)
        # A'_i = A_base * (1 + gamma * rho_i)
        amplitudes = base_amplitude * (1.0 + gamma * densities)
        amplitudes = amplitudes[:, np.newaxis] # Reshape for broadcasting

        # 2. Generate Hybrid Direction (Eq 3.3)
        r_rand = np.random.uniform(-1, 1, particles.shape)
        d_hybrid = (1.0 - omega) * r_rand + omega * repulsion_vecs

        # 3. Apply Displacement
        scale_factor = (self.ub - self.lb)
        displacement = amplitudes * d_hybrid * scale_factor
        new_particles = particles + displacement

        return new_particles

    def density_aware_repulsive_dar(self, current_pos, list_fitness, prev_best_fitness, prev_best_pos):
        epsilon = self.params['epsilon']
        s_min = self.params['s_min']
        s_max = self.params['s_max']
        a_min = self.params['a_min']
        a_max = self.params['a_max']
        danger_p = self.params['danger_p']

        num_danger = int(danger_p * self.n)

        # danger_indices = np.arange(self.n - num_danger, self.n)
        danger_indices = np.random.choice(self.n, num_danger, replace=False)
        # danger_indices = np.random.choice(self.n, num_danger, replace=False) # Alternative: Random

        fitness_best_danger = np.min(list_fitness)
        fitness_worst_danger = np.max(list_fitness)
        fit_range = fitness_worst_danger - fitness_best_danger + epsilon

        if 'num_nodes' in self.params:
            num_nodes = self.params['num_nodes']
            Rs = self.params['sensing_radius']

            for i in danger_indices:
                # 1. Extract Particles (Sensors)
                particles = current_pos[i].reshape(num_nodes, 2)

                # 2. Calculate Physics (Internal Density)
                rho, v_rep = self.calculate_density_and_repulsion(particles, Rs)

                # 3. FBS Logic
                normalized_fitness = (list_fitness[i] - fitness_best_danger) / fit_range
                spark_count = int(s_min + np.round((s_max - s_min) * normalized_fitness))
                base_amplitude = a_min + (a_max - a_min) * normalized_fitness

                local_best_spark_fitness = np.inf
                local_best_spark_pos = None

                for k in range(spark_count):
                    # Generate candidate
                    new_particles = self.generate_da_r_fbs_candidate(particles, rho, v_rep, base_amplitude)
                    candidate_pos = new_particles.flatten()

                    # Boundary Check & Evaluate
                    candidate_pos = np.clip(candidate_pos, self.lb, self.ub)
                    candidate_fitness = self.obj_func(candidate_pos)

                    if candidate_fitness < local_best_spark_fitness:
                        local_best_spark_fitness = candidate_fitness
                        local_best_spark_pos = candidate_pos.copy()

                # Update Sparrow
                if local_best_spark_fitness < list_fitness[i]:
                    list_fitness[i] = local_best_spark_fitness
                    current_pos[i] = local_best_spark_pos
                    if local_best_spark_fitness < prev_best_fitness:
                        prev_best_fitness = local_best_spark_fitness
                        prev_best_pos = local_best_spark_pos.copy()

        else:
            # Dynamic Interaction Radius
            space_diag = np.linalg.norm(self.ub - self.lb) if isinstance(self.lb, np.ndarray) else (self.ub - self.lb) * np.sqrt(self.dim)
            Rs = 0.05 * space_diag

            # Calculate Physics (Population Density)
            rho_pop, v_rep_pop = self.calculate_density_and_repulsion(current_pos, Rs)

            for i in danger_indices:
                # FBS Logic
                normalized_fitness = (list_fitness[i] - fitness_best_danger) / fit_range
                spark_count = int(s_min + np.round((s_max - s_min) * normalized_fitness))
                base_amplitude = a_min + (a_max - a_min) * normalized_fitness

                local_best_spark_fitness = np.inf
                local_best_spark_pos = None

                # Get this sparrow's physics data
                p_i = current_pos[i].reshape(1, self.dim)
                rho_i = np.array([rho_pop[i]])
                v_rep_i = v_rep_pop[i].reshape(1, self.dim)

                for k in range(spark_count):
                    # Generate candidate
                    new_p_i = self.generate_da_r_fbs_candidate(p_i, rho_i, v_rep_i, base_amplitude)
                    candidate_pos = new_p_i.flatten()

                    # Boundary Check & Evaluate
                    candidate_pos = np.clip(candidate_pos, self.lb, self.ub)
                    candidate_fitness = self.obj_func(candidate_pos)

                    if candidate_fitness < local_best_spark_fitness:
                        local_best_spark_fitness = candidate_fitness
                        local_best_spark_pos = candidate_pos.copy()

                # Update Sparrow
                if local_best_spark_fitness < list_fitness[i]:
                    list_fitness[i] = local_best_spark_fitness
                    current_pos[i] = local_best_spark_pos
                    if local_best_spark_fitness < prev_best_fitness:
                        prev_best_fitness = local_best_spark_fitness
                        prev_best_pos = local_best_spark_pos.copy()

        return prev_best_fitness, prev_best_pos

    def run(self):
        list_fitness = []
        convergence_curve = []
        current_pos = self.initialize()
        velocities = np.zeros((self.n, self.dim))

        for i in range(0, self.n):
            fitness = self.obj_func(current_pos[i])
            list_fitness.append(fitness)

        prev_best_fitness = np.min(list_fitness)
        start_best_index = np.argmin(list_fitness)
        prev_best_pos = current_pos[start_best_index].copy()
        current_best_pos = prev_best_pos.copy()

        for t in tqdm(range(0, self.max_iter), desc="DAR-SSA Progress: "):
            current_best = np.min(list_fitness)
            current_best_index = np.argmin(list_fitness)

            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_best_pos.copy()
            else:
                 # Keep tracking the global best
                 prev_best_pos = current_pos[np.argmin(list_fitness)].copy()

            r_current = self.adaptive_role(t)
            producer_count = int(r_current * self.n)

            # Sort population
            sorted_indices = np.argsort(list_fitness)
            current_pos = current_pos[sorted_indices]
            velocities = velocities[sorted_indices]
            list_fitness = np.array(list_fitness)
            list_fitness = list_fitness[sorted_indices]

            # 1. Update Producers
            for i in range(producer_count):
                current_pos[i] = self.producer_update(current_pos[i], i)
                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

            # 2. Update Scroungers
            current_pos = self.update_scroungers(current_pos, producer_count, self.n, self.dim, prev_best_pos, current_pos[-1])

            for i in range(producer_count, self.n):
                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

            # Update Global Best before DAR
            current_best = np.min(list_fitness)
            current_best_idx = np.argmin(list_fitness)
            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_pos[current_best_idx].copy()

            # 3. Density-Aware Repulsive DAR
            current_best, current_best_pos = self.density_aware_repulsive_dar(current_pos, list_fitness, prev_best_fitness, prev_best_pos)

            convergence_curve.append(current_best)

        return current_best, current_best_pos, convergence_curve