import pathlib

import meshio
import numpy as np
import pytest

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "points,cells,ref",
    [
        # line
        ([[0.0], [0.35]], [[0, 1]], [0.35]),
        ([[0.0], [0.35]], [[1, 0]], [-0.35]),
        # triangle
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]], [0.5]),
        ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [[0, 1, 2]], [-0.5]),
        (
            [[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0]],
            [[0, 1, 2], [0, 3, 2]],
            [0.5, -0.55],
        ),
        # tetra
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0, 1, 2, 3]],
            [1 / 6],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0, 1, 3, 2]],
            [-1 / 6],
        ),
    ],
)
def test_signed_area(points, cells, ref):
    mesh = meshplex.Mesh(points, cells)
    ref = np.array(ref)
    assert mesh.signed_cell_volumes.shape == ref.shape
    assert np.all(
        np.abs(ref - mesh.signed_cell_volumes) < np.abs(ref) * 1.0e-13 + 1.0e-13
    )


def test_signed_area_pacman():
    mesh = meshio.read(this_dir / "meshes" / "pacman.vtk")
    assert np.all(np.abs(mesh.points[:, 2]) < 1.0e-15)
    X = mesh.points[:, :2]

    mesh = meshplex.Mesh(X, mesh.get_cells_type("triangle"))

    vols = mesh.signed_cell_volumes
    # all cells are positively oriented in this mesh
    assert np.all(mesh.signed_cell_volumes > 0.0)
    assert np.all(abs(abs(vols) - mesh.cell_volumes) < 1.0e-12 * mesh.cell_volumes)
