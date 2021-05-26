import numpy as np

import meshplex


def _is_near_equal(a, b, tol=1.0e-12):
    return np.allclose(a, b, rtol=0.0, atol=tol)


def test_mesh_line():
    pts = [0.0, 1.0, 3.0, 4.0]
    cells = [[0, 1], [1, 2], [2, 3]]
    mesh = meshplex.Mesh(pts, cells)
    assert _is_near_equal(mesh.cell_volumes, [1.0, 2.0, 1.0])
    assert _is_near_equal(mesh.control_volumes, [0.5, 1.5, 1.5, 0.5])

    assert np.all(mesh.is_boundary_point == [True, False, False, True])

    # control volume with some index
    c = mesh.get_control_volume_centroids([False, False, False])
    assert _is_near_equal(c, [0.25, 1.25, 2.75, 3.75])
