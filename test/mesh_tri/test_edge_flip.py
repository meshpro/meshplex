import pathlib

import meshio
import numpy
from helpers import near_equal

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_flip_delaunay():
    mesh = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")

    numpy.random.seed(123)
    mesh.points[:, :2] += 5.0e-2 * numpy.random.rand(*mesh.points[:, :2].shape)

    mesh = meshplex.MeshTri(mesh.points, mesh.get_cells_type("triangle"))

    assert mesh.num_delaunay_violations() == 16

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0

    # Assert edges_cells integrity
    for cell_gid, edge_gids in enumerate(mesh.cells["edges"]):
        for edge_gid in edge_gids:
            num_adj_cells, edge_id = mesh._edge_gid_to_edge_list[edge_gid]
            assert cell_gid in mesh._edges_cells[num_adj_cells][edge_id]

    new_cells = mesh.cells["points"].copy()
    new_coords = mesh.points.copy()

    # Assert that some key values are updated properly
    mesh2 = meshplex.MeshTri(new_coords, new_cells)
    assert numpy.all(mesh.idx_hierarchy == mesh2.idx_hierarchy)
    tol = 1.0e-15
    assert near_equal(mesh.half_edge_coords, mesh2.half_edge_coords, tol)
    assert near_equal(mesh.cell_volumes, mesh2.cell_volumes, tol)
    assert near_equal(mesh.ei_dot_ej, mesh2.ei_dot_ej, tol)


def test_flip_delaunay_near_boundary():
    points = numpy.array(
        [[0.0, +0.0, 0.0], [0.5, -0.1, 0.0], [1.0, +0.0, 0.0], [0.5, +0.1, 0.0]]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1
    assert numpy.array_equal(mesh.cells["points"], [[0, 1, 2], [0, 2, 3]])
    assert numpy.array_equal(mesh.cells["edges"], [[3, 1, 0], [4, 2, 1]])

    mesh.flip_until_delaunay()

    assert mesh.num_delaunay_violations() == 0
    assert numpy.array_equal(mesh.cells["points"], [[1, 2, 3], [1, 3, 0]])
    assert numpy.array_equal(mesh.cells["edges"], [[4, 1, 3], [2, 0, 1]])


def test_flip_same_edge_twice():
    points = numpy.array(
        [[0.0, +0.0, 0.0], [0.5, -0.1, 0.0], [1.0, +0.0, 0.0], [0.5, +0.1, 0.0]]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.num_delaunay_violations() == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0

    # Assert edges_cells integrity
    for cell_gid, edge_gids in enumerate(mesh.cells["edges"]):
        for edge_gid in edge_gids:
            num_adj_cells, edge_id = mesh._edge_gid_to_edge_list[edge_gid]
            assert cell_gid in mesh._edges_cells[num_adj_cells][edge_id]

    new_points = numpy.array(
        [[0.0, +0.0, 0.0], [0.1, -0.5, 0.0], [0.2, +0.0, 0.0], [0.1, +0.5, 0.0]]
    )
    mesh.points = new_points
    assert mesh.num_delaunay_violations() == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0
    # mesh.show()
    mesh.plot()


def test_flip_two_edges():
    alpha = numpy.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]) / 6.0 * numpy.pi
    R = [0.9, 1.0, 0.9, 1.0, 1.2, 1.0]
    points = numpy.array(
        [[r * numpy.cos(a), r * numpy.sin(a), 0.0] for a, r in zip(alpha, R)]
    )
    cells = numpy.array([[1, 3, 5], [0, 1, 5], [1, 2, 3], [3, 4, 5]])
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.num_delaunay_violations() == 2

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0

    mesh.show(show_point_numbers=True)
    assert numpy.array_equal(
        mesh.cells["points"], [[5, 0, 2], [0, 1, 2], [5, 2, 3], [3, 4, 5]]
    )


def test_flip_delaunay_near_boundary_preserve_boundary_count():
    # This test is to make sure meshplex preserves the boundary point count.
    points = numpy.array(
        [
            [+0.0, +0.0, 0.0],
            [+0.5, -0.5, 0.0],
            [+0.5, +0.5, 0.0],
            [+0.0, +0.6, 0.0],
            [-0.5, +0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [2, 3, 4]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1

    mesh.mark_boundary()
    is_boundary_point_ref = [False, True, True, True, True, True]
    assert numpy.array_equal(mesh.is_boundary_point, is_boundary_point_ref)

    mesh.flip_until_delaunay()

    mesh.mark_boundary()
    assert numpy.array_equal(mesh.is_boundary_point, is_boundary_point_ref)


def test_flip_orientation():
    points = numpy.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])

    # preserve positive orientation
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)
    assert numpy.all(mesh.signed_cell_areas > 0.0)
    mesh.flip_until_delaunay()
    assert numpy.all(mesh.signed_cell_areas > 0.0)

    # also preserve negative orientation
    cells = numpy.array([[0, 2, 1], [0, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)
    assert numpy.all(mesh.signed_cell_areas < 0.0)
    mesh.flip_until_delaunay()
    assert numpy.all(mesh.signed_cell_areas < 0.0)


def test_flip_infinite():
    """In rare cases, it can happen that the ce-ratio of an edge is negative (up to
    machine precision, -2.13e-15 or something like that), an edge flip is done, and the
    ce-ratio of the resulting edge is again negative. The flip_until_delaunay() method
    would continue indefinitely. This test replicates such an edge case."""
    a = 3.9375644347017862e02
    points = numpy.array([[205.0, a], [185.0, a], [330.0, 380.0], [60.0, 380.0]])
    cells = [[0, 1, 2], [1, 2, 3]]

    mesh = meshplex.MeshTri(points, cells)
    num_flips = mesh.flip_until_delaunay(tol=1.0e-13)
    assert num_flips == 0
