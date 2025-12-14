import numpy as np

class benchmark():
    def __init__(self, lb, ub, dim):
        self.lb = lb
        self.ub = ub
        self.dim = dim

    def sphere(self, pos):
        return np.sum(pos**2)

    # Sphere function
    def F1_function(self, pos):
        return np.sum(pos ** 2)

    # Schwefel's function 2.21
    def F2_function(self, pos):
        return np.sum(np.abs(pos)) + np.prod(np.abs(pos))

    # Schwefel's Problem 1.2 or 2.22
    def F3_function(self, pos):
        inner_sums = np.cumsum(pos)
        squared_sums = inner_sums ** 2
        return np.sum(squared_sums)

    # Max function
    def F4_function(self, pos):
        return np.max(np.abs(pos))

    # Rosenbrock's function
    def F5_function(self, pos):
        pos_i = pos[:-1]
        pos_i_plus_1 = pos[1:]
        term1 = 100 * (pos_i_plus_1 - pos_i**2)**2
        term2 = (pos_i - 1)**2
        total_sum = np.sum(term1 + term2)
        return total_sum