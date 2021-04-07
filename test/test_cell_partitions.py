import numpy as np

import meshplex


def test_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array(
        [
            [0.05, 0.15, 0.3],
            [0.05, 0.15, 0.3],
        ]
    )
    assert ref.shape == mesh.cell_partitions.shape
    assert np.all(np.abs(mesh.cell_partitions - ref) < 1.0e-14 * np.abs(ref))


def test_tri_simple():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    cells = np.array([[0, 1, 2]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([[0.125], [0.25], [0.125]])
    print(mesh.cell_partitions)
    assert ref.shape == mesh.cell_partitions.shape
    assert np.all(np.abs(mesh.cell_partitions - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


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

    ref = np.array(
        [
            [0.125, 0.125],
            [0.25, 0.125],
            [0.125, 0.25],
        ]
    )
    print(mesh.cell_partitions)
    assert ref.shape == mesh.cell_partitions.shape
    assert np.all(np.abs(mesh.cell_partitions - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


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

    ref = np.array(
        [
            [1 / 8, 1 / 8],
            [1 / 72, 1 / 72],
            [1 / 72, 1 / 72],
            [1 / 72, 1 / 72],
        ]
    )
    print(mesh.cell_partitions)
    assert ref.shape == mesh.cell_partitions.shape
    assert np.all(np.abs(mesh.cell_partitions - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


if __name__ == "__main__":
    test_tri_simple()
