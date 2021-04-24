import meshplex


def test_gh126():
    """https://github.com/nschloe/meshplex/issues/126"""
    cells = [[1, 2, 0], [2, 3, 0], [3, 1, 0]]
    points = [
        [4.3000283, 8.57424769, 0.010431],
        [4.2568793, 8.41702649, 0.0],
        [4.2864766, 8.56681008, 0.0],
        [4.2897457, 8.58639357, 0.0],
    ]
    mesh = meshplex.MeshTri(points, cells)

    mesh.flip_until_delaunay()

    # this failed in a previous version because there were edges with more than two
    # cells after the flip
    mesh.create_facets()


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

    # this failed in a previous version because there were edges with more than two
    # cells after the flip
    mesh.create_facets()
