import meshplex


def test_gh130():
    """https://github.com/nschloe/meshplex/issues/130"""
    points = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.9, 1.0],
        [1.0, 1.1, 1.0],
        [1.0, 0.7, 1.0],
        [1.0, 0.7, 0.0],
    ]
    cells = [[2, 0, 1], [2, 3, 0], [4, 0, 3], [4, 3, 5], [2, 5, 3]]

    mesh = meshplex.MeshTri(points, cells)
    mesh.flip_until_delaunay(max_steps=1)

    # make sure we can build a new mesh from the points/cells
    mesh = meshplex.MeshTri(mesh.points, mesh.cells("points"))
