import numpy as np


class MeshLine:
    """Class for handling line segment "meshes"."""

    def __init__(self, points, cells):
        self.points = points

        num_cells = len(cells)
        self.cells = np.empty(num_cells, dtype=np.dtype([("nodes", (int, 2))]))
        self.cells["nodes"] = cells

        self.create_cell_volumes()
        self.create_control_volumes()

    def create_cell_volumes(self):
        """Computes the volumes of the "cells" in the mesh."""
        self.cell_volumes = np.array(
            [
                abs(self.points[cell["nodes"]][1] - self.points[cell["nodes"]][0])
                for cell in self.cells
            ]
        )

    def create_control_volumes(self):
        """Compute the control volumes of all nodes in the mesh."""
        self.control_volumes = np.zeros(len(self.points), dtype=float)
        for k, cell in enumerate(self.cells):
            node_ids = cell["nodes"]
            self.control_volumes[node_ids] += 0.5 * self.cell_volumes[k]

        # Sanity checks.
        sum_cv = sum(self.control_volumes)
        sum_cells = sum(self.cell_volumes)
        alpha = sum_cv - sum_cells
        assert abs(alpha) < 1.0e-6

    def show_vertex_function(self, u):
        """"""
        from matplotlib import pyplot as plt

        plt.plot(self.points, u)
