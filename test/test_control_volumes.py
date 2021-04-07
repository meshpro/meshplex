import numpy as np

import meshplex


def test_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)

    vol = np.sum(mesh.cell_volumes)
    assert np.all(np.abs(np.sum(mesh.control_volumes) - vol) < 1.0e-14 * np.abs(vol))

    ref = np.array([0.05, 0.2, 0.45, 0.3])
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref))


def test_tri_degen():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0e-2],
        ]
    )

    cells = np.array([[0, 1, 2]])
    mesh = meshplex.Mesh(X, cells)

    vol = np.sum(mesh.cell_volumes)
    assert np.all(np.abs(np.sum(mesh.control_volumes) - vol) < 1.0e-12 * np.abs(vol))

    ref = np.array([-1.560625, -1.560625, 3.12625])
    print(mesh.control_volumes)
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

    ref = np.array([9.0, 1.0, 1.0, 1.0]) / 72
    mesh.control_volumes
    print()
    print(mesh.control_volumes)
    print(sum(mesh.control_volumes))
    print(ref)
    print(sum(ref))
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


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
    print(ref)
    assert np.all(np.abs(mesh.control_volumes - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)
