import numpy as np

import meshplex

from .helpers import assert_mesh_consistency, compute_all_entities


def test():
    points = [
        [-2.1, -3.1],
        [0.0, 0.0],
        [1.0, 0.0],
        [-2.1, -3.1],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    cells = [[1, 2, 4], [1, 4, 5]]
    mesh = meshplex.MeshTri(points, cells)
    compute_all_entities(mesh)

    mesh.remove_dangling_points()

    assert len(mesh.points) == 4
    assert np.all(mesh.cells("points") == np.array([[0, 1, 2], [0, 2, 3]]))
    assert np.all(
        mesh.edges["points"] == np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])
    )
    assert_mesh_consistency(mesh)
