import matplotlib.pyplot as plt
import numpy as np
# from ssa_pm import ssapm

class coverage():
    def __init__(self, w, h, num_nodes, sensing_radius, r_error, pos):
        self.w = w
        self.h = h
        self.num_nodes = num_nodes
        self.sensing_radius = sensing_radius
        self.r_error = r_error
        self.nodes_pos = pos
        self.lambda_param = 0.5
        self.beta_param = 1

        x_range = np.arange(0, self.w + 1, 1)
        y_range = np.arange(0, self.h + 1, 1)

        # x_range = np.arange(0, self.w, 1)
        # y_range = np.arange(0, self.h, 1)

        X, Y = np.meshgrid(x_range, y_range)
        self.grid_points = np.column_stack((X.ravel(), Y.ravel()))
        # print(f"Grid points: {self.grid_points.shape}")

        # print(f"pos is: {pos}")
        # print(f"num of pos: {len(pos)}")
        # self.nodes = np.random.rand(self.num_nodes, 2)
        # self.nodes[:, 0] *= self.w
        # self.nodes[:, 1] *= self.h

        # if isinstance(num_nodes, int):
        #     self.num_nodes = num_nodes
        #     self.nodes = np.random.rand(self.num_nodes, 2)
        #     self.nodes[:, 0] *= self.w
        #     self.nodes[:, 1] *= self.h
        # else:
        #     self.nodes = num_nodes
        #     self.num_nodes = len(num_nodes)

    def calculate_probabilistics_coverage(self):
        # euclidean distance apply
        # grid points (M, 1, 2)
        # sensor nodes (1, N, 2)
        diff = self.grid_points[:, np.newaxis, :] - self.nodes_pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        r_c = self.sensing_radius
        r_e = self.r_error
        # certainty limit
        c1 = r_c - r_e
        # max range (uncertainty limit)
        c2 = r_c + r_e
        probs = np.zeros_like(dists)
        mask1 = dists <= c1
        probs[mask1] = 1.0

        mask2 = (dists > c1) & (dists < c2)
        target_dists = dists[mask2]
        alpha = target_dists - c1
        calculated_values = np.exp(-self.lambda_param * (alpha ** self.beta_param))
        probs[mask2] = calculated_values

        mask3 = dists >= c2
        probs[mask3] = 0.0

        prob_not_detected = np.prod(1 - probs, axis=1)
        p_total = 1 - prob_not_detected

        cov = np.mean(p_total)

        # print(f"diff shape is: {diff.shape}")
        # print(f"dists shape is: {dists.shape}")
        # print(f"probs is: {probs[1][1]}")
        # print(f"Coverage is: {cov * 100:.2f}%")

        return cov

    def plot_coverage(self):
        fig, ax = plt.subplots(figsize=(6,6))

        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect('equal')
        ax.set_title('Coverage')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        for node in self.nodes_pos:
            circle = plt.Circle(node, self.sensing_radius, color='green', fill=True, alpha=0.1)
            ax.add_artist(circle)
            ax.plot(node[0], node[1], 'r.', markersize=5)

        ax.grid(True, linestyle='--')
        # ax.legend()
        all_x = [s[0] for s in self.nodes_pos]
        all_y = [s[1] for s in self.nodes_pos]
        ax.set_xlim(min(all_x) - self.sensing_radius, max(all_x) + self.sensing_radius)
        ax.set_ylim(min(all_y) - self.sensing_radius, max(all_y) + self.sensing_radius)

        plt.show()

