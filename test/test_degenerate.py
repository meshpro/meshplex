import meshplex
import numpy as np


def test_degenerate_cell(tol=1.0e-14):
    points = [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
    ]
    cells = [[0, 1, 2]]
    mesh = meshplex.Mesh(points, cells)

    ref = np.array([0.0])
    assert np.all(np.abs(mesh.cell_volumes - ref) < tol * (1.0 + ref))

    ref = np.array([[0.0], [0.0], [0.0]])
    assert np.all(np.abs(mesh.cell_volumes - ref) < tol * (1.0 + ref))

    ref = np.array([[0.0], [0.0], [0.0]])
    assert np.all(np.abs(mesh.cell_heights - ref) < tol * (1.0 + ref))

    # inf, not nan
    assert np.all(mesh.cell_circumradius == [np.inf])
    assert np.all(mesh.circumcenter_facet_distances == [[np.inf], [-np.inf], [np.inf]])
    assert np.all(
        mesh.cell_partitions
        == [[[np.inf], [-np.inf], [np.inf]], [[np.inf], [-np.inf], [np.inf]]]
    )

    # those are nan
    assert np.all(np.isnan(mesh.cell_circumcenters))


def test_degenerate_flip():
    # almost degenerate
    points = [
        [0.0, 0.0],
        [0.5, -1.0e-5],
        [1.0, 0.0],
        [0.5, 0.5],
    ]
    cells = [[0, 2, 1], [0, 2, 3]]
    mesh = meshplex.MeshTri(points, cells)
    num_flips = mesh.flip_until_delaunay()
    assert num_flips == 1
    ref = np.array([[1, 0, 3], [1, 3, 2]])
    assert np.all(mesh.cells("points") == ref)

    # make sure the same thing happens if the cell is exactly degenerate
    points = [
        [0.0, 0.0],
        [0.5, 0.0],
        [1.0, 0.0],
        [0.5, 0.5],
    ]
    cells = [[0, 2, 1], [0, 2, 3]]
    mesh = meshplex.MeshTri(points, cells)
    num_flips = mesh.flip_until_delaunay()
    assert num_flips == 1
    ref = np.array([[1, 0, 3], [1, 3, 2]])
    assert np.all(mesh.cells("points") == ref)
