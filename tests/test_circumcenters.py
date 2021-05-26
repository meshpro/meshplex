import numpy as np

import meshplex


def test_circumcenters_line():
    pts = [0.0, 1.0, 3.0, 4.0]
    cells = [[0, 1], [1, 2], [2, 3]]
    mesh = meshplex.Mesh(pts, cells)
    print(mesh.cell_circumcenters)
    ref = [0.5, 2.0, 3.5]
    assert np.all(np.abs(mesh.cell_circumcenters - ref) < np.abs(ref) * 1.0e-13)

    print(mesh.cell_circumradius)
    ref = [0.5, 1.0, 0.5]
    assert np.all(np.abs(mesh.cell_circumradius - ref) < np.abs(ref) * 1.0e-13)


def test_circumcenters_tri():
    # two triangles in 5D
    points = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    cells = [[0, 1, 2], [0, 3, 2]]
    mesh = meshplex.Mesh(points, cells)

    ref = [[0.5, 0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]]
    assert np.all(
        np.abs(mesh.cell_circumcenters - ref) < np.abs(ref) * 1.0e-13 + 1.0e-13
    )

    print(mesh.cell_circumradius)
    ref = [np.sqrt(2) / 2, np.sqrt(2) / 2]
    assert np.all(np.abs(mesh.cell_circumradius - ref) < np.abs(ref) * 1.0e-13)


def test_circumcenters_tetra():
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    cells = [[0, 1, 2, 3]]
    mesh = meshplex.Mesh(points, cells)

    ref = [[0.5, 0.5, 0.5]]
    print(mesh.cell_circumcenters)
    assert np.all(np.abs(mesh.cell_circumcenters - ref) < np.abs(ref) * 1.0e-13)

    print(mesh.cell_circumradius)
    ref = [np.sqrt(3) / 2]
    assert np.all(np.abs(mesh.cell_circumradius - ref) < np.abs(ref) * 1.0e-13)


def test_circumcenters_simplex5():
    points = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cells = [[0, 1, 2, 3, 4]]
    mesh = meshplex.Mesh(points, cells)

    ref = [[0.5, 0.5, 0.5, 0.5]]
    print(mesh.cell_circumcenters)
    assert np.all(np.abs(mesh.cell_circumcenters - ref) < np.abs(ref) * 1.0e-13)

    print(mesh.cell_circumradius)
    ref = [1.0]
    assert np.all(np.abs(mesh.cell_circumradius - ref) < np.abs(ref) * 1.0e-13)


if __name__ == "__main__":
    # test_circumcenters_tri()
    test_circumcenters_tetra()
