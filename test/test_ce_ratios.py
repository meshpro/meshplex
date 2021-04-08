import numpy as np

import meshplex


def test_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)

    ref = np.array([10.0, 10.0 / 3.0, 5.0 / 3.0])
    print(mesh.ce_ratios)
    assert ref.shape == mesh.ce_ratios.shape
    assert np.all(np.abs(mesh.ce_ratios - ref) < 1.0e-14 * np.abs(ref))


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

    ref = np.array([[0.5], [0.0], [0.5]])
    print(mesh.ce_ratios)
    assert ref.shape == mesh.ce_ratios.shape
    assert np.all(np.abs(mesh.ce_ratios - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


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

    ref = np.array([[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    print(mesh.ce_ratios)
    assert ref.shape == mesh.ce_ratios.shape
    assert np.all(np.abs(mesh.ce_ratios - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


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
            [
                [-1 / 24, -1 / 24],
                [1 / 8, 1 / 8],
                [1 / 8, 1 / 8],
                [0.0, 0.0],
            ],
            [
                [-1 / 24, -1 / 24],
                [1 / 8, 1 / 8],
                [0.0, 0.0],
                [1 / 8, 1 / 8],
            ],
            [
                [-1 / 24, -1 / 24],
                [0.0, 0.0],
                [1 / 8, 1 / 8],
                [1 / 8, 1 / 8],
            ],
        ],
    )
    print(mesh.ce_ratios)
    assert ref.shape == mesh.ce_ratios.shape
    assert np.all(np.abs(mesh.ce_ratios - ref) < 1.0e-14 * np.abs(ref) + 1.0e-14)


if __name__ == "__main__":
    test_tri_simple()
