# -*- coding: utf-8 -*-
#
import numpy
import meshplex

from helpers import download_mesh


def _run(mesh):
    # Create circular vector field 0.5 * (y, -x, 0)
    # which has curl (0, 0, 1).
    A = numpy.array(
        [[-0.5 * coord[1], 0.5 * coord[0], 0.0] for coord in mesh.node_coords]
    )
    # Compute the curl numerically.
    B = mesh.compute_curl(A)

    # mesh.write(
    #     'curl.vtu',
    #     point_data={'A': A},
    #     cell_data={'B': B}
    #     )

    tol = 1.0e-14
    for b in B:
        assert abs(b[0] - 0.0) < tol
        assert abs(b[1] - 0.0) < tol
        assert abs(b[2] - 1.0) < tol
    return


def test_pacman():
    filename = download_mesh("pacman.vtk", "c621cb22f8b87cecd77724c2c0601c36")
    mesh = meshplex.read(filename)
    _run(mesh)
    return


if __name__ == "__main__":
    test_pacman()
