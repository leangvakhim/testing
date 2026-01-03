import numpy as np

class benchmark():
    def __init__(self, lb, ub, dim):
        self.lb = lb
        self.ub = ub
        self.dim = dim

    def get_function(func_name):
        if func_name == "sphere" or func_name == "f1":
            return benchmark.sphere_function, -100, 100, 30, 0.0
        elif func_name == "schwefel_2_21" or func_name == "f2":
            return benchmark.schwefel_2_21_function, -10, 10, 30, 0.0
        elif func_name == "schwefel_2_22" or func_name == "f3":
            return benchmark.schwefel_2_22_function, -100, 100, 30, 0.0
        elif func_name == "max" or func_name == "f4":
            return benchmark.max_function, -100, 100, 30, 0.0
        elif func_name == "rosenbrock" or func_name == "f5":
            return benchmark.rosenbrock_function, -30, 30, 30, 0.0
        elif func_name == "step" or func_name == "f6":
            return benchmark.step_function, -100, 100, 30, 0.0
        elif func_name == "quartic" or func_name == "f7":
            return benchmark.quartic_function, -1.28, 1.28, 30, 0.0
        elif func_name == "schwefel_2_26" or func_name == "f8":
            return benchmark.schwefel_2_26_function, -500, 500, 30, -418.9829
        elif func_name == "rastrigin" or func_name == "f9":
            return benchmark.rastrigin_function, -5.12, 5.12, 30, 0.0
        elif func_name == "ackley" or func_name == "f10":
            return benchmark.ackley_function, -32, 32, 30, 0.0
        elif func_name == "griewank" or func_name == "f11":
            return benchmark.griewank_function, -600, 600, 30, 0.0
        # elif func_name == "levy" or func_name == "f12":
        #     return benchmark.levy_function, -50, 50, 30, 0.0
            # fixed dimension
        # elif func_name == "six-hump-camel" or func_name == "f13":
        #     return benchmark.F13_function, -5, 5, 2, -1.0316
        # elif func_name == "kowalik" or func_name == "f17":
        #     return benchmark.F17_function, -5, 5, 4, 0.000307
        # elif func_name == "shekel" or func_name == "f18":
        #     return benchmark.F18_function, 0, 1, 3, -3.86
        else:
            raise ValueError(f"Function '{func_name}' not found. Options: Sphere, Schwefel-1, Rosenbrock, Ackley, etc.")


    # Sphere function
    def sphere_function(self, pos):
        return np.sum(pos ** 2)

    # Schwefel's function 2.21
    def schwefel_2_21_function(self, pos):
        return np.sum(np.abs(pos)) + np.prod(np.abs(pos))

    # Schwefel's Problem 1.2 or 2.22
    def schwefel_2_22_function(self, pos):
        inner_sums = np.cumsum(pos)
        squared_sums = inner_sums ** 2
        return np.sum(squared_sums)

    # Max function
    def max_function(self, pos):
        return np.max(np.abs(pos))

    # Rosenbrock's function
    def rosenbrock_function(self, pos):
        pos_i = pos[:-1]
        pos_i_plus_1 = pos[1:]
        term1 = 100 * (pos_i_plus_1 - pos_i**2)**2
        term2 = (pos_i - 1)**2
        total_sum = np.sum(term1 + term2)
        return total_sum

    # Step function
    def step_function(self, pos):
        x_plus_0_5 = pos + 0.5
        # floored_x = np.floor(x_plus_0_5)
        # squared_values = floored_x**2
        squared_values = x_plus_0_5**2
        total_sum = np.sum(squared_values)
        return total_sum

    # Quartic function
    def quartic_function(self, pos):
        dimension = self.dim
        i_vector = np.arange(1, dimension + 1)
        x_pow_4 = pos**4
        axis = 1 if pos.ndim == 2 else None
        sum_part = np.sum(x_pow_4 * i_vector, axis=axis)
        if pos.ndim == 1:
            noise_part = np.random.rand()
        else:
            noise_part = np.random.rand(pos.shape[0])
        return sum_part + noise_part

    # Multimodal Test Functions (F8 - F12)
    # Schwefel's function 2.26
    def schwefel_2_26_function(self, pos):
        term = -pos * np.sin(np.sqrt(np.abs(pos)))
        total_sum = np.sum(term)
        return total_sum

    # Rastrigin's function
    def rastrigin_function(self, pos):
        part1 = pos**2
        part2 = 10 * np.cos(2 * np.pi * pos)
        terms = part1 - part2 + 10
        total_sum = np.sum(terms)
        return total_sum

    # Ackley Function
    def ackley_function(self, pos):
        dimension = self.dim
        sum_sq = np.sum(pos**2)
        avg_sum_sq = sum_sq / dimension
        sqrt_avg_sum_sq = np.sqrt(avg_sum_sq)
        part1 = -20 * np.exp(-0.5 * sqrt_avg_sum_sq)
        sum_cos = np.sum(np.cos(2 * np.pi * pos))
        avg_sum_cos = sum_cos / dimension
        part2 = -np.exp(avg_sum_cos)
        total_sum = part1 + part2 + 20 + np.e
        return total_sum

    # Griewank Function
    def griewank_function(self, pos):
        dimension = self.dim
        sum_sq = np.sum(pos**2)
        part1 = sum_sq / 4000
        i = np.arange(1, dimension + 1)
        denominators = np.sqrt(i)
        cos_terms_input = pos / denominators
        cos_terms = np.cos(cos_terms_input)
        part2 = np.prod(cos_terms)
        total_sum = part1 - part2 + 1
        return total_sum

    # F12 helper function
    def _u(self, pos, a, k, m):
        u_vals = np.zeros_like(pos)

        # Condition 1: pos > a
        cond1_mask = pos > a
        u_vals[cond1_mask] = k * (pos[cond1_mask] - a)**m

        # Condition 2: pos < -a
        cond2_mask = pos < -a
        u_vals[cond2_mask] = k * (-pos[cond2_mask] - a)**m
        return u_vals

    # Levy's function
    def levy_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        dimension = self.dim
        u_sum = np.sum(self._u(pos, a=10, k=100, m=4), axis=1)
        y = 1 + (pos + 1) / 4
        part1 = 10 * np.sin(np.pi * y[:, 0])
        part3 = (y[:, -1] - 1)**2
        y_i = y[:, :-1]
        y_i_plus_1 = y[:, 1:]
        term1 = (y_i - 1)**2
        term2 = 1 + 10 * np.sin(np.pi * y_i_plus_1)**2
        part2_sum = np.sum(term1 * term2, axis=1)
        total = (np.pi / dimension) * (part1 + part2_sum + part3) + u_sum
        return total[0] if original_ndim == 1 else total

    # Fixed-Dimension Test Functions (F13 - F19)
    # Six-Hump Camel function
    def F13_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        x1 = pos[:, 0]
        x2 = pos[:, 1]
        term1 = 4 * x1**2
        term2 = -2.1 * x1**4
        term3 = (x1**6) / 3
        term4 = x1 * x2
        term5 = -4 * x2**2
        term6 = 4 * x2**4
        total_sum = term1 + term2 + term3 + term4 + term5 + term6
        return total_sum[0] if original_ndim == 1 else total_sum

    def F14_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        x1 = pos[:, 0]
        x2 = pos[:, 1]
        sinc_term = np.sinc(x1 - 2) * np.sinc(x2 - 2)
        part1 = 1 - np.abs(sinc_term)**5
        part2 = 2 + (x1 - 7)**2 + 2 * (x2 - 7)**2
        total_value = part1 * part2
        return total_value[0] if original_ndim == 1 else total_value

    def F15_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        x1 = pos[:, 0]
        x2 = pos[:, 1]
        term_sqrt = np.sqrt(x1**2 + x2**2) / np.pi
        term_abs1 = np.abs(100 - term_sqrt)
        term_exp = np.exp(term_abs1)
        term_sins = np.sin(x1) * np.sin(x2)
        term_abs2 = np.abs(term_exp * term_sins)
        term_power = (term_abs2 + 1)**(-0.1)
        total_value = -term_power
        return total_value[0] if original_ndim == 1 else total_value

    def F16_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        x1 = pos[:, 0]
        x2 = pos[:, 1]
        beta = 15
        m = 5
        term1_sum = (x1 / beta)**(2 * m) + (x2 / beta)**(2 * m)
        term1 = np.exp(-term1_sum)
        term2_sum = x1**2 + x2**2
        term2 = 2 * np.exp(-term2_sum)
        part1 = term1 - term2
        part2 = (np.cos(x1)**2) * (np.cos(x2)**2)
        total_value = part1 * part2
        return total_value[0] if original_ndim == 1 else total_value

    # Kowalik Function
    def F17_function(self, pos):
        original_ndim = pos.ndim
        if original_ndim == 1:
            pos = pos[np.newaxis, :]
        a = np.array([
            0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
            0.0456, 0.0342, 0.0323, 0.0235, 0.0246
        ])
        b_inv = np.array([
            0.25, 0.5, 1.0, 2.0, 4.0, 6.0,
            8.0, 10.0, 12.0, 14.0, 16.0
        ])
        b = 1.0 / b_inv
        a = a.reshape(1, 11)
        b = b.reshape(1, 11)
        x1 = pos[:, 0:1]
        x2 = pos[:, 1:2]
        x3 = pos[:, 2:3]
        x4 = pos[:, 3:4]
        b_sq = b**2
        numerator = x1 * (b_sq + b * x2)
        denominator = b_sq + b * x3 + x4
        fraction = numerator / denominator
        inner_term = a - fraction
        squared_term = inner_term**2
        total_value = np.sum(squared_term, axis=1)
        return total_value[0] if original_ndim == 1 else total_value

    # Shekel function
    def F18_function(self, pos):
        c = np.array([0.1, 0.2, 0.2, 0.4])
        A = np.array([
            [10.0,  3.0, 17.0],
            [ 3.0, 3.5,  8.0],
            [17.0,  8.0, 17.0],
            [ 8.0, 10.0,  1.0]
        ])
        P = np.array([
            [0.1312, 0.1696, 0.5569],
            [0.2329, 0.4135, 0.8307],
            [0.2348, 0.1415, 0.3522],
            [0.4047, 0.8828, 0.8732]
        ])
        n_pop = pos.shape[0]
        pos_reshaped = pos.reshape(n_pop, 1, 3)
        P_reshaped = P.reshape(1, 4, 3)
        A_reshaped = A.reshape(1, 4, 3)
        diff = pos_reshaped - P_reshaped
        diff_sq = diff**2
        inner_term = A_reshaped * diff_sq
        inner_sum = np.sum(inner_term, axis=2)
        exp_term = np.exp(-inner_sum)
        c_reshaped = c.reshape(1, 4)
        term_to_sum = c_reshaped * exp_term
        outer_sum = np.sum(term_to_sum, axis=1)
        total_value = -outer_sum
        return total_value

    def F19_function(self, pos):
        c = np.array([0.1, 0.2, 0.2, 0.4])
        A = np.array([
            [10.0,  3.0, 17.0, 3.50, 1.70,  8.0],
            [ 3.0, 3.5,  1.7, 10.0, 17.0,  8.0],
            [17.0,  8.0, 10.0,  3.0,  1.7, 3.5],
            [ 8.0, 10.0,  3.5, 17.0,  8.0, 17.0]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569,  124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1415, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091,  381]
        ])
        n_pop = pos.shape[0]
        pos_reshaped = pos.reshape(n_pop, 1, 6)
        P_reshaped = P.reshape(1, 4, 6)
        A_reshaped = A.reshape(1, 4, 6)
        diff = pos_reshaped - P_reshaped
        diff_sq = diff**2
        inner_term = A_reshaped * diff_sq
        inner_sum = np.sum(inner_term, axis=2)
        exp_term = np.exp(-inner_sum)
        c_reshaped = c.reshape(1, 4)
        term_to_sum = c_reshaped * exp_term
        outer_sum = np.sum(term_to_sum, axis=1)
        total_value = -outer_sum
        return total_value