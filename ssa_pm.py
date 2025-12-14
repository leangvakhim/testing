import numpy as np
from benchmark import benchmark
from tqdm import tqdm
import math

class ssapm():
    def __init__(self, lb, ub, dim, n, max_iter, params, obj_func):
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n = n
        self.max_iter = max_iter
        self.params = params
        self.obj_func = obj_func

    def initialize(self):
        x = self.lb + np.random.rand(self.n, self.dim) * (self.ub - self.lb)
        return x

    # def obj_func(self, val):
    #     obj_func = benchmark(self.lb, self.ub, self.dim)
    #     return obj_func.F15_function(val)

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
        num_danger = int(self.params['danger_p'] * self.n)
        danger_indices = np.arange(self.n - num_danger, self.n)
        # danger_indices = np.random.choice(self.n, num_danger, replace=False)
        # print(f"list fitness: {list_fitness}")
        # print(f"danger indices: {danger_indices}")
        # print(f"danger values: {list_fitness[danger_indices]}")

        fitness_best_danger = list_fitness[0]
        fitness_worst_danger = list_fitness[-1]

        for i in danger_indices:
            # Calculate Spark Parameters (Si and Ai)
            normalized_fitness = (list_fitness[i] - fitness_best_danger) / (fitness_worst_danger - fitness_best_danger + self.params['epsilon'])

            spark_count = int(self.params['s_min'] + np.round((self.params['s_max'] - self.params['s_min']) * normalized_fitness))
            explosion_amplitude = self.params['a_min'] + (self.params['a_max'] - self.params['a_min']) * normalized_fitness

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

            # Greedy Selection (Update if Better)
            if local_best_spark_fitness < list_fitness[i]:
                current_pos[i] = local_best_spark_pos
                list_fitness[i] = local_best_spark_fitness

                # Update Global Best if found
                if local_best_spark_fitness < prev_best_fitness:
                    prev_best_fitness = local_best_spark_fitness
                    prev_best_pos = local_best_spark_pos.copy()

        return prev_best_fitness, prev_best_pos

    def run(self):
        list_fitness = []
        stagnate_count = 0
        repel = 0
        convergence_curve = []
        current_pos = self.initialize()
        for i in range(0, self.n):
            fitness = self.obj_func(current_pos[i])
            list_fitness.append(fitness)
            # print(f"Sparrow {i} Initial Fitness: {fitness}")

        prev_best_fitness = np.min(list_fitness)
        start_best_index = np.argmin(list_fitness)
        prev_best_pos = current_pos[start_best_index].copy()

        for t in tqdm(range(0, self.max_iter), desc="Progress: "):
            current_best = np.min(list_fitness)
            current_best_index = np.argmin(list_fitness)
            current_worst_index = np.argmax(list_fitness)

            if current_best >= prev_best_fitness:
                stagnate_count += 1
            else:
                stagnate_count = 0
                prev_best_fitness = current_best
                prev_best_pos = current_pos[current_best_index].copy()

            self.params['flag_stagnate'] = False

            if stagnate_count >= self.params['tau_stagnate']:

                old_fitness_best = list_fitness[current_best_index]

                # levy flight
                step_val = self.levy_flight_jump()
                new_fitness_pos = prev_best_pos + step_val * self.params['alpha_levy_flight'] * (self.ub - self.lb)
                new_fitness = self.obj_func(new_fitness_pos)

                if new_fitness < old_fitness_best:
                    current_pos[current_best_index] = new_fitness_pos
                    list_fitness[current_best_index] = new_fitness
                    # print(f"{t} Phoenix Jump: Old Best {old_fitness_best:.4f} -> New Best {new_fitness:.4f}")

                # old_fitness_worst = list_fitness[current_worst_index]

                # chaotic rebirth
                new_ashes = self.chaotic_rebirth()
                current_pos[current_worst_index] = new_ashes
                new_fitness_ashes = self.obj_func(new_ashes)
                list_fitness[current_worst_index] = new_fitness_ashes

                # print(f"{t} Ashes Rebirth: Old Worst {old_fitness_worst:.4f} -> New Random {new_fitness_ashes:.4f}")

                stagnate_count = 0
                self.params['flag_stagnate'] = True

            r_current = self.params['r_end'] + (self.params['r_start'] - self.params['r_end']) * (1 - ((t / self.max_iter) ** self.params['dynamic_role_lambda']))
            producer_count = int(r_current * self.n)
            scrounger_count = self.n - producer_count
            sorted_indices = np.argsort(list_fitness)
            current_pos = current_pos[sorted_indices]
            list_fitness = np.array(list_fitness)
            list_fitness = list_fitness[sorted_indices]

            fitness_best = list_fitness[0]
            fitness_worst = list_fitness[-1]

            m = (fitness_worst - list_fitness) / (fitness_worst - fitness_best + self.params['epsilon'])
            M = m / np.sum(m)
            g = self.params['g_0'] * np.exp(-self.params['alpha_gsa'] * t / self.max_iter)
            temperature_current = self.params['t_0'] * self.params['alpha_sa'] ** t
            mean_pos = np.mean(current_pos, axis=0)
            diagonal_length = np.sqrt(((self.ub - self.lb) ** 2) * self.dim)
            dist_from_mean = np.sqrt(np.sum((current_pos - mean_pos) ** 2, axis=1))
            total_sum_distances = np.sum(dist_from_mean)
            diversity = total_sum_distances / (self.n * diagonal_length)
            r_heat = self.params['r_base'] * ((1 - diversity) ** self.params['heat_lambda'])

            for i in range(self.n):
                # producer update
                if i < producer_count:
                    # check if already jump via Levy Flight
                    if i == 0 and self.params['flag_stagnate']:
                        continue
                    else:
                        # old_pos = current_pos[i].copy()
                        # old_fitness = list_fitness[i]
                        r_2 = np.random.rand()
                        alpha = np.random.rand()
                        L = np.ones(self.dim)

                        if r_2 < self.params['st']:
                            current_pos[i] = current_pos[i] * np.exp(-i / (alpha * self.max_iter))
                        else:
                            current_pos[i] = current_pos[i] + np.random.normal() * L

                    current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                    list_fitness[i] = self.obj_func(current_pos[i])

                    if list_fitness[i] < prev_best_fitness:
                        prev_best_fitness = list_fitness[i]
                        prev_best_pos = current_pos[i].copy()

                # scrounger update
                else:
                    # gravitational attraction
                    current_best_pos = current_pos[0]
                    distance_best_to_each = np.linalg.norm(current_best_pos - current_pos[i])
                    acceleration = g * M[0] * (current_best_pos - current_pos[i]) / (distance_best_to_each + self.params['epsilon'])
                    fitness_temp = current_pos[i] + acceleration
                    # repulsion
                    distance_temp_best_fitness = np.linalg.norm(fitness_temp - current_best_pos)

                    if distance_temp_best_fitness >= r_heat:
                        current_pos[i] = fitness_temp
                    else:
                        fitness_new = self.obj_func(fitness_temp)
                        delta_fitness = fitness_new - list_fitness[0]

                        if delta_fitness >= 0:
                            repel = 1
                        else:
                            repel = np.exp(delta_fitness / temperature_current)

                        if repel > np.random.rand():
                            random_kick = np.random.rand(self.dim)
                            current_pos[i] = fitness_temp - random_kick * (fitness_temp - current_pos[i])
                        else:
                            current_pos[i] = fitness_temp

                current_pos[i] = np.clip(current_pos[i], self.lb, self.ub)
                list_fitness[i] = self.obj_func(current_pos[i])

                if list_fitness[i] < prev_best_fitness:
                    prev_best_fitness = list_fitness[i]
                    prev_best_pos = current_pos[i].copy()

            prev_best_fitness, prev_best_pos = self.flare_burst_search(current_pos, list_fitness, prev_best_fitness, prev_best_pos)

            # current_best = prev_best_fitness

            convergence_curve.append(prev_best_fitness)

        # print(f"prev_best_fitness: {prev_best_fitness:.4e}")
        # print(f"convergence_curve: {convergence_curve}")
        # print(f"prev_best_pos: {prev_best_pos}")
        return prev_best_fitness, prev_best_pos, convergence_curve
