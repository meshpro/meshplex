import pathlib

import meshio
import numpy as np

import meshplex

from .helpers import assert_mesh_consistency, compute_all_entities

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
    points = np.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = np.array([[0, 1, 3], [1, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_facets()
    assert not mesh.is_delaunay
    assert mesh.num_delaunay_violations == 1
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert np.array_equal(mesh.cells("edges"), [[3, 1, 0], [4, 3, 2]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    num_flips = mesh.flip_until_delaunay()
    assert num_flips == 1

    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations == 0
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert np.array_equal(mesh.cells("points"), [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(mesh.cells("edges"), [[2, 3, 0], [4, 1, 3]])


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
    points = np.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = np.array([[0, 3, 1], [1, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_facets()
    assert mesh.num_delaunay_violations == 1
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert np.array_equal(mesh.cells("edges"), [[3, 0, 1], [4, 2, 3]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations == 0
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert np.array_equal(mesh.cells("points"), [[0, 3, 2], [0, 2, 1]])
    assert np.array_equal(mesh.cells("edges"), [[4, 3, 1], [2, 0, 3]])


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
    points = np.array([[-0.1, 0.0], [0.0, -1.0], [0.1, 0.0], [0.0, 1.1]])
    cells = np.array([[0, 1, 3], [1, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_facets()
    assert mesh.num_delaunay_violations == 1
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [1, 3], [2, 3]]
    )
    assert np.array_equal(mesh.cells("edges"), [[3, 1, 0], [4, 2, 3]])
    assert_mesh_consistency(mesh)

    # mesh.show()
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations == 0
    assert np.array_equal(
        mesh.edges["points"], [[0, 1], [0, 3], [1, 2], [0, 2], [2, 3]]
    )
    assert np.array_equal(mesh.cells("points"), [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(mesh.cells("edges"), [[2, 3, 0], [4, 1, 3]])


def test_flip_delaunay_near_boundary():
    points = np.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_facets()
    assert mesh.num_delaunay_violations == 1
    assert np.array_equal(mesh.cells("points"), [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(mesh.cells("edges"), [[3, 1, 0], [4, 2, 1]])

    mesh.flip_until_delaunay()

    assert_mesh_consistency(mesh)
    assert mesh.num_delaunay_violations == 0
    assert np.array_equal(mesh.cells("points"), [[1, 2, 3], [1, 3, 0]])
    assert np.array_equal(mesh.cells("edges"), [[4, 1, 3], [2, 0, 1]])


def test_flip_same_edge_twice():
    points = np.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.num_delaunay_violations == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations == 0

    mesh.show(
        mark_cells=mesh.is_boundary_cell,
        show_point_numbers=True,
        show_edge_numbers=True,
        show_cell_numbers=True,
    )
    assert_mesh_consistency(mesh)

    new_points = np.array([[0.0, +0.0], [0.1, -0.5], [0.2, +0.0], [0.1, +0.5]])
    mesh.points = new_points
    assert mesh.num_delaunay_violations == 1

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations == 0
    mesh.show()
    # mesh.plot()


def test_flip_two_edges():
    alpha = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0]) / 6.0 * np.pi
    # Make the mesh slightly asymmetric to get the same flips on every architecture; see
    # <https://github.com/nschloe/meshplex/issues/78>.
    R = [0.95, 1.0, 0.9, 1.0, 1.2, 1.0]
    points = np.array([[r * np.cos(a), r * np.sin(a), 0.0] for a, r in zip(alpha, R)])
    cells = np.array([[1, 3, 5], [0, 1, 5], [1, 2, 3], [3, 4, 5]])
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.num_delaunay_violations == 2

    mesh.flip_until_delaunay()
    assert mesh.num_delaunay_violations == 0

    mesh.show(show_point_numbers=True)
    assert np.array_equal(
        mesh.cells("points"), [[2, 5, 0], [2, 0, 1], [5, 2, 3], [3, 4, 5]]
    )


def test_flip_delaunay_near_boundary_preserve_boundary_count():
    # This test is to make sure meshplex preserves the boundary point count.
    points = np.array(
        [
            [+0.0, +0.0],
            [+0.5, -0.5],
            [+0.5, +0.5],
            [+0.0, +0.6],
            [-0.5, +0.5],
            [-0.5, -0.5],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1], [2, 3, 4]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.create_facets()
    assert mesh.num_delaunay_violations == 1

    is_boundary_point_ref = [False, True, True, True, True, True]
    assert np.array_equal(mesh.is_boundary_point, is_boundary_point_ref)

    mesh.flip_until_delaunay()
    assert np.array_equal(mesh.is_boundary_point, is_boundary_point_ref)


def test_flip_orientation():
    points = np.array([[0.0, +0.0], [0.5, -0.1], [1.0, +0.0], [0.5, +0.1]])

    # preserve positive orientation
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)
    assert np.all(mesh.signed_cell_volumes > 0.0)
    mesh.flip_until_delaunay()
    assert np.all(mesh.signed_cell_volumes > 0.0)

    # also preserve negative orientation
    cells = np.array([[0, 2, 1], [0, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)
    assert np.all(mesh.signed_cell_volumes < 0.0)
    mesh.flip_until_delaunay()
    assert np.all(mesh.signed_cell_volumes < 0.0)


def test_flip_infinite():
    """In rare cases, it can happen that the ce-ratio of an edge is negative (up to
    machine precision, -2.13e-15 or something like that), an edge flip is done, and the
    ce-ratio of the resulting edge is again negative. The flip_until_delaunay() method
    would continue indefinitely. This test replicates such an edge case."""
    a = 3.9375644347017862e02
    points = np.array([[205.0, a], [185.0, a], [330.0, 380.0], [60.0, 380.0]])
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
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.5], [1.0, 1.0], [0.0, 1.0], [0.9, 0.5]]
    )
    cells = np.array([[0, 1, 5], [1, 3, 5], [1, 2, 3], [3, 4, 5], [0, 5, 4]])

    mesh = meshplex.MeshTri(points, cells)
    compute_all_entities(mesh)
    # mesh.show(mark_cells=mesh.is_boundary_cell)
    mesh.flip_until_delaunay()
    assert_mesh_consistency(mesh)
    # mesh.show(mark_cells=mesh.is_boundary_cell)
    assert np.all(mesh.is_boundary_cell)


def test_flip_delaunay():
    np.random.seed(123)
    mesh0 = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")
    mesh0.points[:, :2] += 5.0e-2 * np.random.rand(*mesh0.points[:, :2].shape)

    mesh0 = meshplex.MeshTri(mesh0.points[:, :2], mesh0.get_cells_type("triangle"))
    compute_all_entities(mesh0)

    assert mesh0.num_delaunay_violations == 16

    mesh0.flip_until_delaunay()
    assert mesh0.num_delaunay_violations == 0

    assert_mesh_consistency(mesh0)

    # mesh0.show(mark_cells=mesh0.is_boundary_cell)

    # We don't need to check for exact equality with a replicated mesh. The order of the
    # edges will be different, for example. Just make sure the mesh is consistent.
    # mesh1 = meshplex.MeshTri(mesh0.points.copy(), mesh0.cells("points").copy())
    # mesh1.create_facets()
    # assert_mesh_equality(mesh0, mesh1)


def test_flip_into_existing_edge():
    """For surface meshes, flips can lead to duplicate cells. For context, see
    <https://github.com/nschloe/optimesh/issues/71#issuecomment-785699560>.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.3, 0.3],
            [0.5, -0.3, 0.3],
        ]
    )
    cells = np.array(
        [
            [1, 2, 0],
            [2, 3, 0],
            [3, 1, 0],
        ]
    )

    mesh = meshplex.MeshTri(points, cells)
    mesh.flip_until_delaunay()
    # Note that we actually have duplicate cells after the flipping. This can happen
    # with manifold meshes. An edge is also duplicated.
    # When reinstating a new mesh from points and cells, the edge generator will fail
    # with a hint on duplicate cells. The simple fix is to remove those first via
    # remove_duplicate_cells().
    ref = np.array([[[2, 0, 3], [2, 3, 0], [2, 3, 1]]])
    # after flip
    assert np.all(mesh.cells("points") == ref)

    mesh2 = meshplex.Mesh(mesh.points, mesh.cells("points"))
    mesh2.remove_duplicate_cells()
    ref = np.array([[[2, 0, 3], [2, 3, 1]]])
    assert np.all(mesh2.cells("points") == ref)


def test_doubled_cell():
    # Two congruent cells. One can think of it as a deflated, coarse ball.
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.4],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)
    mesh.flip_until_delaunay()
    assert np.all(np.isnan(mesh.circumcenter_facet_distances))


def test_negative_after_flip():
    points = [[0.0, 0.0], [3.0, 0.0], [1.14960653, 0.03], [1.85039347, 0.03]]
    cells = [
        [0, 3, 2],
        [0, 1, 3],
    ]
    mesh0 = meshplex.MeshTri(points, cells)
    # mesh.show()
    mesh0.flip_until_delaunay()
