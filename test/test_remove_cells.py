import pathlib

import numpy as np
import pytest

import meshplex

from .mesh_tri.helpers import assert_mesh_consistency, compute_all_entities


def get_mesh0():
    # _____________
    # |   _/ \_   |
    # | _/     \_ |
    # |/_________\|
    # |\_       _/|
    # |  \_   _/  |
    # |___ \_/____|
    #
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.5],
        [0.5, 0.0],
        [1.0, 0.5],
        [0.5, 1.0],
    ]
    cells = [
        [0, 5, 4],
        [5, 1, 6],
        [6, 2, 7],
        [7, 3, 4],
        [5, 6, 4],
        [6, 7, 4],
    ]
    mesh = meshplex.Mesh(points, cells)
    mesh.create_facets()
    return mesh


def get_mesh1():
    # _____________
    # |\_       _/|
    # |  \_   _/  |
    # |    \_/    |
    # |   _/ \_   |
    # | _/     \_ |
    # |/_________\|
    #
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    cells = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 4, 3]]
    return meshplex.Mesh(points, cells)


def get_mesh2():
    this_dir = pathlib.Path(__file__).resolve().parent
    mesh0 = meshplex.read(this_dir / "meshes" / "pacman.vtk")
    return meshplex.Mesh(mesh0.points[:, :2], mesh0.cells("points"))


@pytest.mark.parametrize(
    "remove_idx,expected_num_cells,expected_num_edges",
    [
        # remove corner cell
        [[0], 5, 11],
        # remove corner cell
        [[0, 4], 4, 10],
        # remove interior cells
        [[4, 5], 4, 12],
        # remove no cells at all
        [[], 6, 13],
    ],
)
def test_remove_cells(remove_idx, expected_num_cells, expected_num_edges):
    mesh = get_mesh0()
    assert len(mesh.cells("points")) == 6
    assert len(mesh.edges["points"]) == 13
    # remove a corner cell
    mesh.remove_cells(remove_idx)
    assert len(mesh.cells("points")) == expected_num_cells
    assert len(mesh.edges["points"]) == expected_num_edges
    assert_mesh_consistency(mesh)


def test_remove_cells_boundary():
    mesh = get_mesh1()

    assert np.all(mesh.is_boundary_point == [True, True, True, True, False])
    assert np.all(mesh.is_boundary_facet_local[0] == [False, False, False, False])
    assert np.all(mesh.is_boundary_facet_local[1] == [False, False, False, True])
    assert np.all(mesh.is_boundary_facet_local[2] == [True, True, True, False])
    assert np.all(
        mesh.is_boundary_facet == [True, True, False, True, False, True, False, False]
    )
    assert np.all(mesh.is_boundary_cell)
    assert np.all(mesh.facets_cells_idx == [0, 1, 0, 2, 1, 3, 2, 3])
    # cell id:
    assert np.all(mesh.facets_cells["boundary"][1] == [0, 3, 1, 2])
    # local edge:
    assert np.all(mesh.facets_cells["boundary"][2] == [2, 1, 2, 2])
    # cell id:
    assert np.all(
        mesh.facets_cells["interior"][1:3].T == [[0, 3], [0, 1], [1, 2], [2, 3]]
    )
    # local edge:
    assert np.all(
        mesh.facets_cells["interior"][3:5].T == [[1, 2], [0, 1], [0, 1], [0, 0]]
    )

    # now lets remove some cells
    mesh.remove_cells([0])

    assert_mesh_consistency(mesh)

    assert np.all(mesh.is_boundary_point)
    assert np.all(mesh.is_boundary_facet_local[0] == [False, False, False])
    assert np.all(mesh.is_boundary_facet_local[1] == [True, False, True])
    assert np.all(mesh.is_boundary_facet_local[2] == [True, True, True])
    assert np.all(
        mesh.is_boundary_facet == [True, True, True, True, True, False, False]
    )
    assert np.all(mesh.is_boundary_cell)


def test_remove_boundary_cell():
    mesh = get_mesh0()
    mesh.remove_boundary_cells(
        lambda ibc: np.all(mesh.cell_centroids[ibc] > 0.5, axis=1)
    )
    assert mesh.cells("points").shape[0] == 5


def test_remove_all():
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cells = [[0, 1, 2]]
    mesh = meshplex.Mesh(points, cells)
    assert np.all(mesh.is_point_used)

    mesh.remove_cells([0])
    assert_mesh_consistency(mesh)
    assert not np.any(mesh.is_point_used)


@pytest.mark.parametrize(
    "mesh0, remove_cells",
    [
        (get_mesh0(), [0]),
        (get_mesh1(), [0, 1]),
        (get_mesh2(), [0, 3, 57, 59, 60, 61, 100]),
    ],
)
def test_reference(mesh0, remove_cells):
    # some dummy calls to make sure the respective value are computed before the cell
    # removal and then updated
    compute_all_entities(mesh0)

    # now remove some cells
    mesh0.remove_cells(remove_cells)

    assert_mesh_consistency(mesh0)


def test_remove_duplicate():
    # lines
    points = [0.0, 0.1, 0.7, 1.0]
    cells = [[0, 1], [1, 2], [2, 1], [0, 1], [3, 2]]
    mesh = meshplex.Mesh(points, cells)
    n = mesh.remove_duplicate_cells()
    assert n == 2
    assert np.all(mesh.cells("points") == [[0, 1], [1, 2], [3, 2]])

    # triangle
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 2, 3], [0, 1, 2], [0, 2, 1]]
    mesh = meshplex.Mesh(points, cells)
    n = mesh.remove_duplicate_cells()
    assert n == 1
    assert np.all(mesh.cells("points") == [[0, 2, 3], [0, 1, 2]])

    # tetrahedra
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cells = [[0, 1, 2, 3], [3, 1, 2, 0]]
    mesh = meshplex.Mesh(points, cells)
    n = mesh.remove_duplicate_cells()
    assert n == 1
    assert np.all(mesh.cells("points") == [[0, 1, 2, 3]])


if __name__ == "__main__":
    test_remove_cells_boundary()
