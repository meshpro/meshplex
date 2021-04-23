import meshplex
import numpy as np


def test_degenerate_cell(tol=1.0e-14):
    points = [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
    ]
    cells = [[0, 1, 2]]
    mesh = meshplex.Mesh(points, cells)

    ref = np.array([0.0])
    assert np.all(np.abs(mesh.cell_volumes - ref) < tol * (1.0 + ref))

    ref = np.array([[0.0], [0.0], [0.0]])
    assert np.all(np.abs(mesh.cell_volumes - ref) < tol * (1.0 + ref))

    # those are nan
    assert np.all(np.isnan(mesh.cell_circumradius))
    assert np.all(np.isnan(mesh.cell_circumcenters))
