import numpy as np
from benchmark import benchmark
from tqdm import tqdm
from coverage import coverage
from scipy.spatial import Voronoi, Delaunay
import math

class ssapm():
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

    # Main
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
            func_to_call = getattr(obj_func, self.func_name)
            f_name = func_to_call(val)
        return f_name

    # def chaotic_initialization(self, map_type='tent'):

    #     population = np.zeros((self.n, self.dim))

    #     # 1. Generate Chaotic Sequence matrix (normalized 0-1)
    #     chaos_matrix = np.zeros((self.n, self.dim))

    #     # Initial chaotic value (random start, but usually not 0, 0.5, etc.)
    #     # We create a random start vector for the first row
    #     x = np.random.rand(self.dim)

    #     # Iterate to fill the columns (or rows depending on strategy)
    #     # Here we run independent chaotic maps for each dimension to decorrelate them
    #     for i in range(self.n):
    #         for j in range(self.dim):
    #             if map_type == 'logistic':
    #                 # Logistic Map: x_new = 4 * x * (1 - x)
    #                 # Avoid fixed points by adding tiny jitter if x hits 0.5 or 0
    #                 if x[j] in [0, 0.25, 0.5, 0.75, 1.0]:
    #                     x[j] += 1e-5
    #                 val = 4.0 * x[j] * (1.0 - x[j])

    #             elif map_type == 'tent':
    #                 # Tent Map
    #                 if x[j] < 0.5:
    #                     val = 2.0 * x[j]
    #                 else:
    #                     val = 2.0 * (1.0 - x[j])

    #             x[j] = val # Update state
    #             chaos_matrix[i, j] = val

    #     # 2. Map Chaotic Sequence from [0, 1] to [lb, ub]
    #     population = self.lb + chaos_matrix * (self.ub - self.lb)

    #     return population

    def levy_flight_jump(self):
        small_sigma_mu_numerator = math.gamma(1 + self.params['beta_levy_flight']) * (math.sin(math.pi * self.params['beta_levy_flight'] / 2))
        small_sigma_mu_denominator = math.gamma((1 + self.params['beta_levy_flight']) / 2) * self.params['beta_levy_flight'] * (2 ** ((self.params['beta_levy_flight'] - 1) / 2))
        small_sigma_mu = (small_sigma_mu_numerator / small_sigma_mu_denominator) ** (1 / self.params['beta_levy_flight'])
        random_mu_val = np.random.normal(0, small_sigma_mu ** 2, size=self.dim)
        random_ve_val = np.random.normal(0, self.params['small_sigma_ve'] ** 2, size=self.dim)
        step_size = random_mu_val / (np.abs(random_ve_val) ** (1 / self.params['beta_levy_flight']))

        return step_size

    def chaotic_rebirth(self):
        z = np.random.rand(self.dim)
        for k in range(100):
            z = self.params['chaotic_rebirth_mu'] * z * (1 - z)
        new_pos = self.lb + z * (self.ub - self.lb)
        return new_pos

    # def flare_burst_search(self, current_pos, list_fitness, prev_best_fitness, prev_best_pos):
    #     epsilon = self.params['epsilon']
    #     s_min = self.params['s_min']
    #     s_max = self.params['s_max']
    #     a_min = self.params['a_min']
    #     a_max = self.params['a_max']
    #     num_danger = int(self.params['danger_p'] * self.n)
    #     danger_indices = np.arange(self.n - num_danger, self.n)

    #     fitness_best_danger = list_fitness[self.n - num_danger]
    #     fitness_worst_danger = list_fitness[-1]

    #     for i in danger_indices:
    #         # Calculate Spark Parameters (Si and Ai)
    #         normalized_fitness = (list_fitness[i] - fitness_best_danger) / (fitness_worst_danger - fitness_best_danger + epsilon)

    #         spark_count = int(s_min + np.round((s_max - s_min) * normalized_fitness))
    #         explosion_amplitude = a_min + (a_max - a_min) * normalized_fitness

    #         local_best_spark_fitness = np.inf
    #         local_best_spark_pos = None

    #         # Generate Sparks (The Burst)
    #         for k in range(spark_count):
    #             # Random Direction Vector (-1 to 1)
    #             random_vector = np.random.uniform(-1, 1, self.dim)

    #             candidate_pos = current_pos[i] + explosion_amplitude * random_vector
    #             candidate_pos = np.clip(candidate_pos, self.lb, self.ub)
    #             candidate_fitness = self.obj_func(candidate_pos)

    #             if candidate_fitness < local_best_spark_fitness:
    #                 local_best_spark_fitness = candidate_fitness
    #                 local_best_spark_pos = candidate_pos.copy()

    #         if local_best_spark_fitness < list_fitness[i]:
    #             list_fitness[i] = local_best_spark_fitness
    #             current_pos[i] = local_best_spark_pos

    #             if local_best_spark_fitness < prev_best_fitness:
    #                 prev_best_fitness = local_best_spark_fitness
    #                 prev_best_pos = local_best_spark_pos.copy()

    #     return prev_best_fitness, prev_best_pos

    def flare_burst_search(self, current_pos, list_fitness, prev_best_fitness, prev_best_pos):
        epsilon = self.params.get('epsilon', 1e-10) # Use safe default if missing
        s_min = self.params['s_min']
        s_max = self.params['s_max']
        a_min = self.params['a_min']
        a_max = self.params['a_max']

        num_danger = int(self.params['danger_p'] * self.n)
        # Ensure at least 1 danger sparrow if danger_p > 0
        num_danger = max(1, num_danger)

        danger_indices = np.arange(self.n - num_danger, self.n)

        fitness_best_danger = list_fitness[self.n - num_danger]
        fitness_worst_danger = list_fitness[-1]

        for i in danger_indices:
            # Calculate Spark Parameters (Si and Ai)
            numerator = list_fitness[i] - fitness_best_danger
            denominator = fitness_worst_danger - fitness_best_danger + epsilon

            # [Fix] Clip the ratio to [0, 1] to prevent explosion
            normalized_fitness = np.clip(numerator / denominator, 0, 1)

            spark_count = int(s_min + np.round((s_max - s_min) * normalized_fitness))
            explosion_amplitude = a_min + (a_max - a_min) * normalized_fitness

            local_best_spark_fitness = np.inf
            local_best_spark_pos = None

            # Generate Sparks (The Burst)
            for k in range(spark_count):
                # Random Direction Vector (-1 to 1)
                random_vector = np.random.uniform(-1, 1, self.dim)

                candidate_pos = current_pos[i] + explosion_amplitude * random_vector
                candidate_pos = np.clip(candidate_pos, self.lb, self.ub)

                # Check bounds or use valid positions
                candidate_fitness = self.obj_func(candidate_pos)

                if candidate_fitness < local_best_spark_fitness:
                    local_best_spark_fitness = candidate_fitness
                    local_best_spark_pos = candidate_pos.copy()

            if local_best_spark_fitness < list_fitness[i]:
                list_fitness[i] = local_best_spark_fitness
                current_pos[i] = local_best_spark_pos

                if local_best_spark_fitness < prev_best_fitness:
                    prev_best_fitness = local_best_spark_fitness
                    prev_best_pos = local_best_spark_pos.copy()

        return prev_best_fitness, prev_best_pos

    def adaptive_role(self, t):
        r_end = self.params['r_end']
        r_start = self.params['r_start']
        dynamic_role_lambda = self.params['dynamic_role_lambda']
        current_role = r_end + (r_start - r_end) * (1 - ((t / self.max_iter) ** dynamic_role_lambda))
        return current_role

    def thermal_attraction(self, f_worst, f_current, f_best, t, c_pos, c_best_pos, c_velocities):
        epsilon = self.params['epsilon']
        g_0 = self.params['g_0']
        alpha_gsa = self.params['alpha_gsa']
        # m_val = (f_worst - f_current) / (f_worst - f_best + epsilon)
        # m = np.array([m_val])
        # M = m / (sum(m) + epsilon)
        # print(f"M[0] is: {M[0]}")
        # calculate adaptive attraction coefficient
        G = g_0 * np.exp(-alpha_gsa * t)/self.max_iter
        # calculate euclidean distance
        R = np.linalg.norm(c_pos - c_best_pos)
        # calculate the acceleration
        # acceleration = G * M[0] * (c_best_pos - c_pos) / (R ** 2 + epsilon)
        acceleration = G * (c_best_pos - c_pos) / (R + epsilon)
        # calculate velocity of each sparrow
        velocity = np.random.rand() * c_velocities + acceleration
        # position update
        att_position = c_pos + velocity
        return att_position, velocity

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

    def thermal_repulsion(self, c_pos, att_pos, c_best_pos, r_heat, t_current):
        fitness_att = self.obj_func(att_pos)
        fitness_best = self.obj_func(c_best_pos)
        if r_heat > np.linalg.norm(att_pos - c_best_pos):
            delta_fitness = fitness_att - fitness_best
            # calculate the probability of being repelled
            if delta_fitness >= 0:
                p_repel = 1
            else:
                p_repel = np.exp(-delta_fitness / t_current)

            # when the sparrow is burned and repelled
            if np.random.rand() < p_repel:
                random_kick = np.random.rand(self.dim)
                c_pos = att_pos - random_kick * (att_pos - c_pos)
            else:
                c_pos = att_pos

        return c_pos

    def calculate_anti_gravity_force(self, i, current_pos, neighbors_indices):
        """
        [Equation 2.1.3] Invert the Attraction (The anti-gravity)
        New Proposed to replace the old 2.1.3.
        Calculates a repulsion force from neighbors to disperse nodes instead of clustering them.

        Args:
            i (int): Index of the current sparrow (scrounger).
            current_pos (np.array): All sparrow positions.
            neighbors_indices (list): Indices of neighbors within 2 * Rs.

        Returns:
            np.array: The calculated repulsion force vector for sparrow i.
        """
        # Coefficients
        k_repel = self.params.get('k_repel', 1.0)  # Repulsion gain coefficient
        rs = self.params['sensing_radius']

        force = np.zeros(self.dim)
        pos_i = current_pos[i]

        for j in neighbors_indices:
            if i == j:
                continue

            pos_j = current_pos[j]
            dist_vec = pos_i - pos_j
            dist = np.linalg.norm(dist_vec)

            # Neighbors are technically defined as nodes within 2*Rs
            # We add a small epsilon to avoid division by zero
            if dist < 2 * rs and dist > 0:
                # u_ij is the unit vector pointing from Node j to Node i
                u_ij = dist_vec / dist

                # Formula: F = Sum( k_repel * (1/d^2) * u_ij )
                force_magnitude = k_repel * (1.0 / (dist**2 + 1e-10))
                force += force_magnitude * u_ij

        return force

    def deterministic_hard_expansion(self, i, current_pos, current_velocity):
        """
        [Equation 2.2.4] Deterministic Hard Expansion
        New Proposed to replace the old Probabilistic Repulsion (Burn).
        Ensures that any time two nodes come too close, they are forcefully separated.

        Args:
            i (int): Index of the current sparrow.
            current_pos (np.array): All sparrow positions.
            current_velocity (np.array): Current velocity of sparrow i.

        Returns:
            np.array: The updated velocity vector after applying hard expansion.
        """
        rs = self.params['sensing_radius']
        # d_thresh is approx sqrt(3) * Rs
        d_thresh = np.sqrt(3) * rs

        # Gamma (push coefficient)
        gamma = self.params.get('gamma', 0.1)

        v_push_total = np.zeros(self.dim)
        pos_i = current_pos[i]

        # Iterate through all other nodes to check for collisions
        for j in range(self.n):
            if i == j:
                continue

            pos_j = current_pos[j]
            dist = np.linalg.norm(pos_i - pos_j)

            # IF d_ij < d_thresh, apply push
            if dist < d_thresh:
                # V_push = gamma * (x_i - x_j)
                # The vector (x_i - x_j) points away from j, pushing i away.
                v_push = gamma * (pos_i - pos_j)
                v_push_total += v_push

        # Update Velocity: v_i(t+1) = v_i(t) + V_push
        new_velocity = current_velocity + v_push_total

        return new_velocity

    def calculate_modified_fitness(self, coverage_rate, current_pos):
        """
        [Equation 4.2] Modifying the fitness function
        Replaces the single objective function.
        Equation: Fitness = w1 * CoverageRate - w2 * OverlapArea + w3 * UniformityMetric

        Args:
            coverage_rate (float): The calculated probabilistic coverage (0.0 to 1.0).
            current_pos (np.array): Positions of all nodes (shape: num_nodes x 2).

        Returns:
            float: The combined fitness value to be minimized.
        """
        # Weights (default values if not in params)
        w1 = self.params.get('w1', 1.0)
        w2 = self.params.get('w2', 0.5)
        w3 = self.params.get('w3', 0.5)

        # 1. Coverage Rate is passed in directly

        # 2. Calculate Overlap Area (Approximation)
        r = self.params['sensing_radius']
        w_area = self.params['w']
        h_area = self.params['h']

        # Use len(current_pos) to get the number of sensors, not self.n
        num_nodes_local = len(current_pos)

        total_potential_area = num_nodes_local * np.pi * (r**2)
        actual_covered_area = coverage_rate * w_area * h_area
        overlap_area = max(0, total_potential_area - actual_covered_area)

        # 3. Calculate Uniformity Metric (Standard Deviation of inter-node distances)
        dists = []
        # Loop over the sensors (rows in current_pos), NOT the sparrow population
        for i in range(num_nodes_local):
            for j in range(i + 1, num_nodes_local):
                d = np.linalg.norm(current_pos[i] - current_pos[j])
                dists.append(d)

        if len(dists) > 0:
            uniformity_metric = np.std(dists)
        else:
            uniformity_metric = 0

        # The Final Equation: Fitness = w1 * Coverage - w2 * Overlap + w3 * Uniformity
        final_fitness_score = (w1 * coverage_rate) - (w2 * overlap_area) + (w3 * uniformity_metric)

        # Return negative because the optimizer minimizes the function
        return -final_fitness_score

    # def calculate_virtual_force(self, current_pos_flat):

    #     if 'num_nodes' not in self.params:
    #         return np.zeros_like(current_pos_flat)

    #     num_nodes = self.params['num_nodes']
    #     Rs = self.params['sensing_radius']
    #     k_rep = 2.0

    #     nodes = current_pos_flat.reshape(num_nodes, 2)
    #     force_vec = np.zeros_like(nodes)

    #     for i in range(num_nodes):
    #         # 1. Inter-node repulsion
    #         for j in range(num_nodes):
    #             if i == j:
    #                 continue
    #             dist_vec = nodes[i] - nodes[j]
    #             dist = np.linalg.norm(dist_vec)

    #             if dist < 2 * Rs and dist > 0:
    #                 f_mag = k_rep / (dist ** 2 + 1e-5)
    #                 force_vec[i] += f_mag * (dist_vec / dist)

    #         # 2. Wall Repulsion
    #         # Left Wall (x=0)
    #         if nodes[i, 0] < Rs: force_vec[i, 0] += k_rep / (nodes[i, 0]**2 + 1e-5)
    #         # Right Wall (x=W)
    #         if nodes[i, 0] > self.params['w'] - Rs: force_vec[i, 0] -= k_rep / ((self.params['w'] - nodes[i, 0])**2 + 1e-5)
    #         # Bottom Wall (y=0)
    #         if nodes[i, 1] < Rs: force_vec[i, 1] += k_rep / (nodes[i, 1]**2 + 1e-5)
    #         # Top Wall (y=H)
    #         if nodes[i, 1] > self.params['h'] - Rs: force_vec[i, 1] -= k_rep / ((self.params['h'] - nodes[i, 1])**2 + 1e-5)

    #     return force_vec.flatten()

    # def voronoi_spark(self, best_pos_flat):
    #     num_nodes = self.params['num_nodes']
    #     nodes = best_pos_flat.reshape(num_nodes, 2)

    #     # Compute Voronoi
    #     try:
    #         vor = Voronoi(nodes)
    #     except:
    #         return best_pos_flat

    #     # Find the largest hole
    #     max_dist = -1
    #     target_pos = None

    #     def in_bounds(pos):
    #         return 0 <= pos[0] <= self.params['w'] and 0 <= pos[1] <= self.params['h']

    #     for v in vor.vertices:
    #         if not in_bounds(v):
    #             continue
    #         d = np.min(np.linalg.norm(nodes - v, axis=1))
    #         if d > max_dist:
    #             max_dist = d
    #             target_pos = v

    #     if target_pos is None:
    #         return best_pos_flat

    #     # Find the most useless node (highest overlap)
    #     overlaps = np.zeros(num_nodes)
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             if i != j and np.linalg.norm(nodes[i] - nodes[j]) < 2 * self.params['sensing_radius']:
    #                 overlaps[i] += 1

    #     worst_node_idx = np.argmax(overlaps)

    #     # Create Spark
    #     new_solution = best_pos_flat.copy()
    #     # Update the x, y of the worst node
    #     new_solution[worst_node_idx * 2] = target_pos[0]
    #     new_solution[worst_node_idx * 2 + 1] = target_pos[1]

    #     return new_solution

    # def calculate_vfa_forces(nodes, sensing_radius, width, height):
    #     forces = np.zeros(shape=(20, 2))
    #     d_th = sensing_radius * np.sqrt(3)  # 12.12m

    #     # 1. Inter-node Repulsion
    #     for i in range(20):
    #         for j in range(i + 1, 20):
    #             dist_vec = nodes[i] - nodes[j]
    #             dist = np.linalg.norm(dist_vec)

    #             # Apply force only if overlap exists (dist < d_th)
    #             if dist < d_th and dist > 0:
    #                 # Coulomb force: k / dist^2
    #                 # Or Linear Spring: k * (d_th - dist)
    #                 force_mag = 10.0 * (d_th - dist) # Linear is often more stable
    #                 force_vec = (dist_vec / dist) * force_mag

    #                 forces[i] += force_vec
    #                 forces[j] -= force_vec

    #     # 2. Boundary Repulsion (Wall Force)
    #     for i in range(20):
    #         # Left Wall
    #         if nodes[i][0] < sensing_radius:
    #             forces[i][0] += 5.0 * (sensing_radius - nodes[i][0])
    #         # Right Wall
    #         if nodes[i][0] > width - sensing_radius:
    #             forces[i][0] -= 5.0 * (nodes[i][0] - (width - sensing_radius))
    #         # Top/Bottom similar...

    #     return forces

    # def delaunay_repair(self, best_pos_flat):
    #     num_nodes = self.params['num_nodes']
    #     w = self.params['w']
    #     h = self.params['h']
    #     sensing_radius = self.params['sensing_radius']

    #     # Reshape to (N, 2)
    #     nodes = best_pos_flat.reshape(num_nodes, 2)

    #     # 1. Compute Delaunay Triangulation
    #     try:
    #         tri = Delaunay(nodes)
    #     except:
    #         return best_pos_flat

    #     # 2. Find the Largest "Empty" Triangle
    #     max_area = -1
    #     target_pos = None

    #     for simplex in tri.simplices:
    #         pts = nodes[simplex]

    #         # Calculate Centroid of the triangle
    #         centroid = np.mean(pts, axis=0)

    #         # Check bounds
    #         if not (0 <= centroid[0] <= w and 0 <= centroid[1] <= h):
    #             continue

    #         # Check if this triangle is actually a "Hole"
    #         # (Distance from centroid to nearest node > sensing_radius)
    #         dists = np.linalg.norm(pts - centroid, axis=1)
    #         if np.min(dists) > sensing_radius * 0.9: # 0.9 factor for safety

    #             # Calculate Area
    #             a, b, c = pts[0], pts[1], pts[2]
    #             area = 0.5 * np.abs(np.cross(b-a, c-a))

    #             if area > max_area:
    #                 max_area = area
    #                 target_pos = centroid

    #     # If no significant hole found, return original
    #     if target_pos is None:
    #         return best_pos_flat

    #     try:
    #         # Add qhull_options="QJ" here as well
    #         tri = Delaunay(nodes, qhull_options="QJ")
    #     except:
    #         # If it fails, just return the original position without repair
    #         return best_pos_flat

    #     # 3. Find the "Worst" Node (Most Clustered/Redundant)
    #     # We find the node that is closest to its neighbors
    #     overlaps = np.zeros(num_nodes)
    #     for i in range(num_nodes):
    #         dist_sum = 0
    #         count = 0
    #         for j in range(num_nodes):
    #             if i == j: continue
    #             d = np.linalg.norm(nodes[i] - nodes[j])
    #             if d < 2 * sensing_radius: # If overlapping
    #                 overlaps[i] += 1
    #                 dist_sum += d
    #                 count += 1
    #         # Penalize nodes with many close neighbors
    #         if count > 0:
    #             overlaps[i] = overlaps[i] + (1.0 / (dist_sum/count + 1e-5))

    #     worst_node_idx = np.argmax(overlaps)

    #     # 4. Move Worst Node to Target (Hole Center)
    #     new_solution = best_pos_flat.copy()
    #     new_solution[worst_node_idx * 2] = target_pos[0]
    #     new_solution[worst_node_idx * 2 + 1] = target_pos[1]

    #     return new_solution

    def run(self):
        list_fitness = []
        # stagnate_count = 0
        heat_lambda = self.params['heat_lambda']
        alpha_cool = self.params['alpha_sa']
        r_base = self.params['r_base']
        t_0 = self.params['t_0']
        convergence_curve = []
        # percentage_to_reset = self.params['tau_stagnate'] * self.max_iter / 100
        current_pos = self.initialize()
        # current_pos = self.chaotic_initialization()
        velocities = np.zeros((self.n, self.dim))

        # for i in range(0, self.n):
        #     fitness = self.obj_func(current_pos[i])
        #     list_fitness.append(fitness)
            # print(f"Sparrow {i} Initial Fitness: {fitness}")

        ## **
        for i in range(0, self.n):
            # 1. Get Base Coverage (obj_func returns 1 - coverage)
            base_obj_val = self.obj_func(current_pos[i])
            coverage_rate = 1.0 - base_obj_val

            # 2. Reshape for fitness function (N_nodes x 2)
            # Assuming dim = num_nodes * 2
            num_nodes = self.params['num_nodes']
            pos_nodes = current_pos[i].reshape(num_nodes, 2)

            # 3. Calculate Modified Fitness
            fitness = self.calculate_modified_fitness(coverage_rate, pos_nodes)
            list_fitness.append(fitness)
        ## **

        prev_best_fitness = np.min(list_fitness)
        start_best_index = np.argmin(list_fitness)
        prev_best_pos = current_pos[start_best_index].copy()
        current_best_pos = prev_best_pos.copy()
        # print(f"List fitness: {list_fitness}")
        # print(f"previous best before loop: {prev_best_fitness:.4e}  ")

        for t in tqdm(range(0, self.max_iter), desc="Progress: "):
            current_best = np.min(list_fitness)
            current_best_index = np.argmin(list_fitness)
            current_worst_index = np.argmax(list_fitness)

            if current_best <= prev_best_fitness:
                # stagnate_count += 1
                # continue
            # else:
                # stagnate_count = 0
                prev_best_fitness = current_best
                current_best_pos = current_pos[current_best_index].copy()
                prev_best_pos = current_best_pos.copy()

            # self.params['flag_stagnate'] = False

            # # if stagnate_count >= self.params['tau_stagnate']:
            # if stagnate_count >= percentage_to_reset:

            #     old_fitness_best = list_fitness[current_best_index]

            #     # levy flight
            #     step_val = self.levy_flight_jump()
            #     new_fitness_pos = current_best_pos + step_val * self.params['alpha_levy_flight'] * (self.ub - self.lb)
            #     new_fitness = self.obj_func(new_fitness_pos)

            #     if new_fitness < old_fitness_best:
            #         current_pos[current_best_index] = new_fitness_pos
            #         list_fitness[current_best_index] = new_fitness
            #         # print(f"{t} Phoenix Jump: Old Best {old_fitness_best:.4f} -> New Best {new_fitness:.4f}")

            #     # old_fitness_worst = list_fitness[current_worst_index]

            #     # chaotic rebirth
            #     new_ashes = self.chaotic_rebirth()
            #     current_pos[current_worst_index] = new_ashes
            #     new_fitness_ashes = self.obj_func(new_ashes)
            #     list_fitness[current_worst_index] = new_fitness_ashes

            #     # print(f"{t} Ashes Rebirth: Old Worst {old_fitness_worst:.4f} -> New Random {new_fitness_ashes:.4f}")

            #     stagnate_count = 0
            #     # print(f"Reset at {t}")
            #     self.params['flag_stagnate'] = True

            # Adaptive role allocation
            # r_current = self.params['r_start']
            r_current = self.adaptive_role(t)
            producer_count = int(r_current * self.n)
            scrounger_count = self.n - producer_count
            sorted_indices = np.argsort(list_fitness)
            current_pos = current_pos[sorted_indices]
            velocities = velocities[sorted_indices]
            list_fitness = np.array(list_fitness)
            list_fitness = list_fitness[sorted_indices]

            fitness_best = list_fitness[0]
            fitness_worst = list_fitness[-1]
            fitness_current = list_fitness[scrounger_count]

            # calculate mean of population
            mean_pos = np.mean(current_pos, axis=0)
            # calculate diagonal length
            d_length = np.sqrt(((self.ub - self.lb) ** 2) * self.dim)
            # calculate distance from mean
            d_from_m = np.sqrt(np.sum((current_pos - mean_pos) ** 2, axis=1))
            # calculate total sum of distances
            total_sum_d = np.sum(d_from_m)
            # calculate diversity
            diversity = total_sum_d / (self.n * d_length)
            # calculate heat radius
            r_heat = r_base * ((1 - diversity) ** heat_lambda)
            # define the cooling schedule
            t_current = t_0 * (alpha_cool ** t)

            for i in range(self.n):
                # producer update
                if i < producer_count:
                    # check if already jump via Levy Flight
                    # if i != 0:
                    # if i == 0 and self.params['flag_stagnate']:
                    #     # print("Trigger")
                    #     continue
                    # else:
                    current_pos[i] = self.producer_update(current_pos[i], i)

                    # current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                    # list_fitness[i] = self.obj_func(current_pos[i])

                    # if list_fitness[i] < current_best:
                    #     current_best = list_fitness[i]
                    #     current_best_pos = current_pos[i].copy()

                # scrounger update
                else:
                    # gravitational attraction
                    # att_pos_from_gsa, velocities[i] = self.thermal_attraction(fitness_worst, fitness_current, fitness_best, t, current_pos[i], current_best_pos, velocities[i])

                    # # repulsion
                    # rep_pos_from_sa = self.thermal_repulsion(current_pos[i], att_pos_from_gsa, current_best_pos, r_heat, t_current)
                    # current_pos[i] = rep_pos_from_sa

                    # Calculate virtual force (repulsion)
                    # v_force = self.calculate_virtual_force(current_pos[i])

                    # # Elastic Attraction to Global Best
                    # elastic_attraction = np.random.rand() * (current_best_pos - current_pos[i])

                    # # Update velocities & Position
                    # velocities[i] = 0.5 * velocities[i] + v_force
                    # current_pos[i] = current_pos[i] + velocities[i]

                    ## **
                    neighbors = []
                    for j in range(self.n):
                        if i == j: continue
                        dist = np.linalg.norm(current_pos[i] - current_pos[j])
                        if dist < 2 * self.params['sensing_radius']:
                            neighbors.append(j)

                    # Calculate Anti-Gravity Force
                    force_repel = self.calculate_anti_gravity_force(i, current_pos, neighbors)

                    # Calculate Acceleration (F = ma, assuming m=1)
                    acceleration = force_repel

                    # Update Velocity (v = v + a) - Simplified integration
                    # You can add a random factor or inertia weight here if desired
                    velocities[i] = velocities[i] + acceleration

                    # Update Position
                    current_pos[i] = current_pos[i] + velocities[i]

                    # --- [Eq 2.2.4] Deterministic Hard Expansion (Replaces Thermal Repulsion) ---
                    # Check for collisions and apply hard push
                    velocities[i] = self.deterministic_hard_expansion(i, current_pos, velocities[i])
                    # Re-apply velocity to position after hard expansion adjustment
                    current_pos[i] = current_pos[i] + velocities[i]

                    ## **

                # vfa_forces = self.calculate_vfa_forces(current_pos)
                # current_pos[i] = current_pos[i] + (vfa_forces[i] * learning_rate)

                # current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                # list_fitness[i] = self.obj_func(current_pos[i])

                ## **
                # ---------------- FITNESS UPDATE [Eq 4.2] ----------------
                base_obj_val = self.obj_func(current_pos[i])
                coverage_rate = 1.0 - base_obj_val

                num_nodes = self.params['num_nodes']
                pos_nodes = current_pos[i].reshape(num_nodes, 2)

                list_fitness[i] = self.calculate_modified_fitness(coverage_rate, pos_nodes)

                ## **
                if list_fitness[i] < current_best:
                    current_best = list_fitness[i]
                    current_best_pos = current_pos[i].copy()

            if current_best < prev_best_fitness:
                prev_best_fitness = current_best
                prev_best_pos = current_best_pos.copy()

            current_best, current_best_pos = self.flare_burst_search(current_pos, list_fitness, prev_best_fitness, prev_best_pos)

            # spark_pos = self.delaunay_repair(current_best_pos)
            # spark_fitness = self.obj_func(spark_pos)

            # # If the repair improved the solution, keep it
            # if spark_fitness < current_best:
            #     # print(f"Delaunay Repair Improved: {current_best:.4f} -> {spark_fitness:.4f}")
            #     current_best = spark_fitness
            #     current_best_pos = spark_pos.copy()
            #     current_pos[current_best_index] = spark_pos.copy()
            #     list_fitness[current_best_index] = spark_fitness

            # Voronoi Spark (Hole Targeting)
            # spark_pos = self.voronoi_spark(current_best_pos)
            # spark_fitness = self.obj_func(spark_pos)

            # if spark_fitness < current_best:
            #     current_best = spark_fitness
            #     current_best_pos = spark_pos.copy()
            #     current_pos[current_best_index] = spark_pos.copy()

            convergence_curve.append(current_best)

        # print(f"prev_best_fitness: {prev_best_fitness:.4e}")
        # print(f"convergence_curve: {convergence_curve}")
        # print(f"prev_best_pos: {prev_best_pos}")

        # print(f"current_best at the bottom: {current_best:.4e}")

        # return prev_best_fitness, prev_best_pos, convergence_curve
        return current_best, current_best_pos, convergence_curve
