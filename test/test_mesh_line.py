import numpy

import meshplex


def _is_near_equal(a, b, tol=1.0e-12):
    return numpy.allclose(a, b, rtol=0.0, atol=tol)


def test_mesh_line():
    pts = numpy.array([0.0, 1.0, 3.0, 4.0])
    cells = [[0, 1], [1, 2], [2, 3]]
    mesh = meshplex.mesh_line.MeshLine(pts, cells)
    assert _is_near_equal(mesh.cell_volumes, [1.0, 2.0, 1.0])
    assert _is_near_equal(mesh.control_volumes, [0.5, 1.5, 1.5, 0.5])
    return
