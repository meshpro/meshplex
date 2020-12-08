import pathlib

import meshio
import numpy

import meshplex

from .helpers import assert_mesh_consistency, assert_mesh_equality, compute_all_entities

this_dir = pathlib.Path(__file__).resolve().parent


def test_flip_simple():
    #        3                   3
    #        A                   A
    #       /|\                 / \
    #     1/ | \4             1/ 1 \4
    #     /  |  \             /     \
    #   0/ 0 3   \2   ==>   0/___3___\2
    #    \   | 1 /           \       /
    #     \  |  /             \     /
    #     0\ | /2             0\ 0 /2
    #       \|/                 \ /
    #        V                   V
    #        1                   1
    #
    points = numpy.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = numpy.array([[0, 1, 3], [1, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["edges"], [[3, 1, 0], [4, 3, 2]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations() == 0
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["points"], [[0, 1, 2], [0, 2, 3]])
    assert numpy.array_equal(mesh.cells["edges"], [[2, 3, 0], [4, 1, 3]])


def test_flip_simple_negative_orientation():
    #        3                   3
    #        A                   A
    #       /|\                 / \
    #     1/ | \4             1/ 1 \4
    #     /  |  \             /     \
    #   0/ 0 3   \2   ==>   0/___3___\2
    #    \   | 1 /           \       /
    #     \  |  /             \     /
    #     0\ | /2             0\ 0 /2
    #       \|/                 \ /
    #        V                   V
    #        1                   1
    #
    points = numpy.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = numpy.array([[0, 3, 1], [1, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["edges"], [[3, 0, 1], [4, 2, 3]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations() == 0
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["points"], [[0, 3, 2], [0, 2, 1]])
    assert numpy.array_equal(mesh.cells["edges"], [[4, 3, 1], [2, 0, 3]])


def test_flip_simple_opposite_orientation():
    #        3                   3
    #        A                   A
    #       /|\                 / \
    #     1/ | \4             1/ 1 \4
    #     /  |  \             /     \
    #   0/ 0 3   \2   ==>   0/___3___\2
    #    \   | 1 /           \       /
    #     \  |  /             \     /
    #     0\ | /2             0\ 0 /2
    #       \|/                 \ /
    #        V                   V
    #        1                   1
    #
    points = numpy.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = numpy.array([[0, 1, 3], [1, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["edges"], [[3, 1, 0], [4, 2, 3]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations() == 0
    assert numpy.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert numpy.array_equal(mesh.cells["points"], [[0, 1, 2], [0, 2, 3]])
    assert numpy.array_equal(mesh.cells["edges"], [[2, 3, 0], [4, 1, 3]])


def test_flip_delaunay_near_boundary():
    points = numpy.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1
    assert numpy.array_equal(mesh.cells["points"], [[0, 1, 2], [0, 2, 3]])
    assert numpy.array_equal(mesh.cells["edges"], [[3, 1, 0], [4, 2, 1]])

    mesh.flip_until_delaunay()

    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations() == 0
    assert numpy.array_equal(mesh.cells["points"], [[1, 2, 3], [1, 3, 0]])
    assert numpy.array_equal(mesh.cells["edges"], [[4, 1, 3], [2, 0, 1]])


def test_flip_same_edge_twice():
    points = numpy.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.num_delaunay_violations() == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0

    mesh.show(
        mark_cells=mesh.is_boundary_cell,
        show_point_numbers=True,
        show_edge_numbers=True,
        show_cell_numbers=True,
    )
    assert_mesh_consistency(mesh)

    new_points = numpy.array([[0.0, +0.0], [0.1, -0.5], [0.2, +0.0], [0.1, +0.5]])
    mesh.points = new_points
    assert mesh.num_delaunay_violations() == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations() == 0
    mesh.show()
    # mesh.plot()


def test_flip_two_edges():
    alpha = numpy.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]) / 6.0 * numpy.pi
    # Make the mesh slightly asymmetric to get the same flips on every architecture; see
    # <https://github.com/nschloe/meshplex/issues/78>.
    R = [0.95, 1.0, 0.9, 1.0, 1.2, 1.0]
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
        mesh.cells["points"], [[2, 5, 0], [2, 0, 1], [5, 2, 3], [3, 4, 5]]
    )


def test_flip_delaunay_near_boundary_preserve_boundary_count():
    # This test is to make sure meshplex preserves the boundary point count.
    points = numpy.array(
        [
            [+0.0, +0.0],
            [+0.5, -0.5],
            [+0.5, +0.5],
            [+0.0, +0.6],
            [-0.5, +0.5],
            [-0.5, -0.5],
        ]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [2, 3, 4]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_edges()
    assert mesh.num_delaunay_violations() == 1

    is_boundary_point_ref = [False, True, True, True, True, True]
    assert numpy.array_equal(mesh.is_boundary_point, is_boundary_point_ref)

    mesh.flip_until_delaunay()
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


def test_flip_interior_to_boundary():
    #  __________          __________
    #  |\__      A         |\__      A
    #  |   \__  /|\        |   \__  / \
    #  |      \/ | \  ==>  |      \/___\
    #  |    __/\ | /       |    __/\   /
    #  | __/    \|/        | __/    \ /
    #  |/________V         |/________V
    #
    points = numpy.array(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.5], [1.0, 1.0], [0.0, 1.0], [0.9, 0.5]]
    )
    cells = numpy.array([[0, 1, 5], [1, 3, 5], [1, 2, 3], [3, 4, 5], [0, 5, 4]])

    mesh = meshplex.MeshTri(points, cells)
    compute_all_entities(mesh)
    # mesh.show(mark_cells=mesh.is_boundary_cell)
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    # mesh.show(mark_cells=mesh.is_boundary_cell)
    assert numpy.all(mesh.is_boundary_cell)


def test_flip_delaunay():
    numpy.random.seed(123)
    mesh0 = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")
    mesh0.points[:, :2] += 5.0e-2 * numpy.random.rand(*mesh0.points[:, :2].shape)

    mesh0 = meshplex.MeshTri(mesh0.points[:, :2], mesh0.get_cells_type("triangle"))
    compute_all_entities(mesh0)

    assert mesh0.num_delaunay_violations() == 16

    mesh0.flip_until_delaunay()
    assert mesh0.num_delaunay_violations() == 0

    assert_mesh_consistency(mesh0)

    # mesh0.show(mark_cells=mesh0.is_boundary_cell)

    # We don't need to check for exact equality with a replicated mesh. The order of the
    # edges will be different, for example. Just make sure the mesh is consistent.
    # mesh1 = meshplex.MeshTri(mesh0.points.copy(), mesh0.cells["points"].copy())
    # mesh1.create_edges()
    # assert_mesh_equality(mesh0, mesh1)


if __name__ == "__main__":
    test_flip_same_edge_twice()
