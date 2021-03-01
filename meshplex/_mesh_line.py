from ._mesh import Mesh


class MeshLine(Mesh):
    """Class for handling line segment "meshes"."""

    def __init__(self, points, cells):
        super().__init__(points, cells)

    def show_vertex_function(self, u):
        import matplotlib.pyplot as plt

        plt.plot(self.points, u)
