import numpy as np

import meshplex


def test_control_volume_centroids_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([0.025, 0.15, 0.475, 0.85])

    print(mesh.control_volume_centroids)
    assert mesh.control_volume_centroids.shape == ref.shape
    assert np.all(
        np.abs(ref - mesh.control_volume_centroids) < np.abs(ref) * 1.0e-13 + 1.0e-13
    )


def test_control_volume_centroids_tri():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.Mesh(points, cells)

    ref = np.array([[0.25, 0.25], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])

    print(mesh.control_volume_centroids)
    assert mesh.control_volume_centroids.shape == ref.shape
    assert np.all(
        np.abs(ref - mesh.control_volume_centroids) < np.abs(ref) * 1.0e-13 + 1.0e-13
    )


def test_tetra_simple():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array(
        [
            [0.25, 0.25, 0.25],
            [17 / 24, 1 / 48, 1 / 48],
            [1 / 48, 17 / 24, 1 / 48],
            [1 / 48, 1 / 48, 17 / 24],
        ]
    )
    print(mesh.control_volume_centroids)
    assert np.all(
        np.abs(mesh.control_volume_centroids - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14
    )
