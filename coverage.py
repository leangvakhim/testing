import matplotlib.pyplot as plt
import numpy as np
from ssa_pm import ssapm

class coverage():
    def __init__(self, w, h, num_nodes, sensing_radius, r_error, pos):
        self.w = w
        self.h = h
        self.num_nodes = num_nodes
        self.sensing_radius = sensing_radius
        self.r_error = r_error
        self.nodes = pos
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


    def plot_coverage(self):
        fig, ax = plt.subplots(figsize=(6,6))

        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.set_aspect('equal')
        ax.set_title('Coverage')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        for node in self.nodes:
            circle = plt.Circle(node, self.sensing_radius, color='green', fill=True, alpha=0.1)
            ax.add_artist(circle)
            ax.plot(node[0], node[1], 'r.', markersize=5)

        ax.grid(True, linestyle='--')
        # ax.legend()
        all_x = [s[0] for s in self.nodes]
        all_y = [s[1] for s in self.nodes]
        ax.set_xlim(min(all_x) - self.sensing_radius, max(all_x) + self.sensing_radius)
        ax.set_ylim(min(all_y) - self.sensing_radius, max(all_y) + self.sensing_radius)

        plt.show()

