import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

        # x_range = np.arange(0, self.w + 1, 1)
        # y_range = np.arange(0, self.h + 1, 1)

        x_range = np.arange(0, self.w, 1)
        y_range = np.arange(0, self.h, 1)

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

        # coverage = (cov * self.w * self.h) / (self.num_nodes * (self.sensing_radius ** np.pi))

        # print(f"diff shape is: {diff.shape}")
        # print(f"dists shape is: {dists.shape}")
        # print(f"probs is: {probs[1][1]}")
        # print(f"Coverage is: {coverage * 100:.2f}%")

        return cov

    def plot_iterative_coverage(self, convergence_curve):
        # Convert fitness (1 - coverage) back to coverage percentage
        coverage_history = [(1 - fitness) * 100 for fitness in convergence_curve]
        iterations = np.arange(1, len(coverage_history) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot the line graph
        ax.plot(iterations, coverage_history, color='b', linewidth=2, label='Max Coverage')

        # Formatting the graph
        ax.set_title('Coverage Rate Evolution')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Coverage Rate (%)')
        ax.set_xlim(0, len(coverage_history))
        ax.set_ylim(min(coverage_history) - 1, 100.5)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.show()


    def plot_coverage(self, best_fitness, node_status=None):
        fig, ax = plt.subplots(figsize=(8,8))

        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect('equal')
        ax.set_title(f'SSA-PM Coverage with percentage: {(1 - best_fitness) * 100:.2f}%')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        r_certain = self.sensing_radius - self.r_error
        r_max = self.sensing_radius + self.r_error

        for i, node in enumerate(self.nodes_pos):
            # 1. Plot Max Range (r_c + r_e)
            circle_max = plt.Circle(node, r_max, color='palegreen', fill=True, alpha=0.2, linewidth=0)
            ax.add_artist(circle_max)

            # 2. Plot Certainty Range (r_c - r_e)
            circle_certain = plt.Circle(node, r_certain, color='forestgreen', fill=True, alpha=0.3, linewidth=0)
            ax.add_artist(circle_certain)

            # 3. Plot Nominal Radius (r_c)
            # circle_nominal = plt.Circle(node, self.sensing_radius, color='black', fill=False, linestyle='--', alpha=0.3)
            # ax.add_artist(circle_nominal)

            # Plot the sensor node center
            ax.plot(node[0], node[1], 'r.', markersize=5)
            # ax.text(node[0] + 0.5, node[1] + 0.5, str(i + 1), fontsize=9, color='black')
            label_text = str(i + 1)
            if node_status is not None and i < len(node_status):
                # Append the role to the text
                label_text += f"\n({node_status[i]})"

            ax.text(node[0] + 0.5, node[1] + 0.5, label_text, fontsize=8, color='black')

        legend_elements = [
            Patch(facecolor='forestgreen', edgecolor='none', alpha=0.3, label='Certainty Range'),
            Patch(facecolor='palegreen', edgecolor='none', alpha=0.2, label='Uncertain Range'),
            Line2D([0], [0], color='black', linestyle='--', alpha=0.5, label='Nominal Radius'),
            Line2D([0], [0], marker='.', color='w', label='Sensor Node', markerfacecolor='r', markersize=10),
        ]

        # ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

        ax.grid(True, linestyle='--')
        all_x = [s[0] for s in self.nodes_pos]
        all_y = [s[1] for s in self.nodes_pos]
        ax.set_xlim(min(all_x) - r_max, max(all_x) + r_max)
        ax.set_ylim(min(all_y) - r_max, max(all_y) + r_max)

        plt.show()

