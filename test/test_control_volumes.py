import numpy as np

import meshplex


def test_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([0.05, 0.2, 0.45, 0.3])
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref))


def test_tri():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref))


def test_tetra():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([18.0, 2.0, 2.0, 1.0, 1.0]) / 72
    print(mesh.control_volumes)
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)
