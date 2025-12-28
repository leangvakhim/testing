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

    def flare_burst_search(self, current_pos, list_fitness, prev_best_fitness, prev_best_pos):
        epsilon = self.params['epsilon']
        s_min = self.params['s_min']
        s_max = self.params['s_max']
        a_min = self.params['a_min']
        a_max = self.params['a_max']
        num_danger = int(self.params['danger_p'] * self.n)
        danger_indices = np.arange(self.n - num_danger, self.n)

        fitness_best_danger = list_fitness[self.n - num_danger]
        fitness_worst_danger = list_fitness[-1]

        for i in danger_indices:
            # Calculate Spark Parameters (Si and Ai)
            normalized_fitness = (list_fitness[i] - fitness_best_danger) / (fitness_worst_danger - fitness_best_danger + epsilon)

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

    def update_scroungers(self, current_pos, pd_count, num_sensor, dim, global_best_position, global_worst_position):
        L = np.ones((1, dim))

        for i in range(pd_count, num_sensor):

            # i > n/2 (Starving scroungers)
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
        # heat_lambda = self.params['heat_lambda']
        # alpha_cool = self.params['alpha_sa']
        # r_base = self.params['r_base']
        # t_0 = self.params['t_0']
        convergence_curve = []
        # percentage_to_reset = self.params['tau_stagnate'] * self.max_iter / 100
        current_pos = self.initialize()
        # current_pos = self.chaotic_initialization()
        velocities = np.zeros((self.n, self.dim))
        for i in range(0, self.n):
            fitness = self.obj_func(current_pos[i])
            list_fitness.append(fitness)
            # print(f"Sparrow {i} Initial Fitness: {fitness}")

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
            # mean_pos = np.mean(current_pos, axis=0)
            # calculate diagonal length
            # d_length = np.sqrt(((self.ub - self.lb) ** 2) * self.dim)
            # calculate distance from mean
            # d_from_m = np.sqrt(np.sum((current_pos - mean_pos) ** 2, axis=1))
            # calculate total sum of distances
            # total_sum_d = np.sum(d_from_m)
            # calculate diversity
            # diversity = total_sum_d / (self.n * d_length)
            # calculate heat radius
            # r_heat = r_base * ((1 - diversity) ** heat_lambda)
            # define the cooling schedule
            # t_current = t_0 * (alpha_cool ** t)

            for i in range(producer_count):
            # for i in range(self.n):
            #     # producer update
            #     if i < producer_count:
                    # check if already jump via Levy Flight
                    # if i != 0:
                    # if i == 0 and self.params['flag_stagnate']:
                    #     # print("Trigger")
                    #     continue
                    # else:
                current_pos[i] = self.producer_update(current_pos[i], i)

                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

                # if list_fitness[i] < current_best:
                #     current_best = list_fitness[i]
                #     current_best_pos = current_pos[i].copy()

                # scrounger update
                # else:
                current_pos = self.update_scroungers(current_pos, producer_count, self.n, self.dim, current_best_pos, current_pos[-1])
                    # gravitational attraction
                    # * new
                    # att_pos_from_gsa, velocities[i] = self.thermal_attraction(fitness_worst, fitness_current, fitness_best, t, current_pos[i], current_best_pos, velocities[i])
                    # current_pos[i] = att_pos_from_gsa
                    # * old
                    # att_pos_from_gsa, velocities[i] = self.thermal_attraction(fitness_worst, fitness_current, fitness_best, t, current_pos[i], current_best_pos, velocities[i])
                    # print(f"att pos from gsa: {att_pos_from_gsa}")
                    # print(f"velocities: {velocities[i]}")
                    # print(f"current pos: {current_pos[i]}")
                    # print(f"fitness current: {fitness_current}")
                    # repulsion
                    # * new
                    # rep_pos_from_sa = self.thermal_repulsion(current_pos[i], current_best_pos, current_best_pos, r_heat, t_current)
                    # * old
                    # rep_pos_from_sa = self.thermal_repulsion(current_pos[i], att_pos_from_gsa, current_best_pos, r_heat, t_current)
                    # current_pos[i] = rep_pos_from_sa

                    # Calculate virtual force (repulsion)
                    # v_force = self.calculate_virtual_force(current_pos[i])

                    # # Elastic Attraction to Global Best
                    # elastic_attraction = np.random.rand() * (current_best_pos - current_pos[i])

                    # # Update velocities & Position
                    # velocities[i] = 0.5 * velocities[i] + v_force
                    # current_pos[i] = current_pos[i] + velocities[i]

                # vfa_forces = self.calculate_vfa_forces(current_pos)
                # current_pos[i] = current_pos[i] + (vfa_forces[i] * learning_rate)

                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

                # if list_fitness[i] < current_best:
                #     current_best = list_fitness[i]
                #     current_best_pos = current_pos[i].copy()

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
