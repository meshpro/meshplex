import pathlib

import numpy as np

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_pacman():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman.vtk")

    # mesh = meshplex.MeshTri(mesh.points[:, :2], mesh.cells("points"))
    # mesh.signed_cell_volumes
    # exit(1)

    # Create circular vector field 0.5 * (y, -x, 0)
    # which has curl (0, 0, 1).
    A = np.array([[-0.5 * coord[1], 0.5 * coord[0], 0.0] for coord in mesh.points])
    # Compute the curl numerically.
    B = mesh.compute_ncurl(A)

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


if __name__ == "__main__":
    test_pacman()
