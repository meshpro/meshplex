import numpy
import pytest

import meshplex


def _get_test_mesh():
    points = numpy.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.5],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
        ]
    )
    cells = numpy.array(
        [
            [0, 5, 4],
            [5, 1, 6],
            [6, 2, 7],
            [7, 3, 4],
            [5, 6, 4],
            [6, 7, 4],
        ]
    )
    mesh = meshplex.MeshTri(points, cells)
    mesh.create_edges()
    return mesh


@pytest.mark.parametrize(
    "remove_idx,expected_num_cells,expected_num_edges",
    [
        # remove corner cell
        [[0], 5, 11],
        # remove interior cells
        [[4, 5], 4, 12],
        # remove no cells at all
        [[], 6, 13],
    ],
)
def test_remove_cells(remove_idx, expected_num_cells, expected_num_edges):
    mesh = _get_test_mesh()
    assert len(mesh.cells["points"]) == 6
    assert len(mesh.edges["points"]) == 13
    # remove a corner cell
    mesh.remove_cells(remove_idx)
    assert len(mesh.cells["points"]) == expected_num_cells
    assert len(mesh.edges["points"]) == expected_num_edges


def test_remove_cells_boundary():
    points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    cells = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 4, 3]]
    mesh = meshplex.MeshTri(points, cells)
    assert numpy.all(mesh.is_boundary_point == [True, True, True, True, False])
    assert numpy.all(mesh.is_boundary_edge[0] == [False, False, False, False])
    assert numpy.all(mesh.is_boundary_edge[1] == [False, False, False, True])
    assert numpy.all(mesh.is_boundary_edge[2] == [True, True, True, False])
    assert numpy.all(
        mesh.is_boundary_edge_gid
        == [True, True, False, True, False, True, False, False]
    )
    assert numpy.all(mesh.is_boundary_cell)

    mesh.remove_cells([0])
    assert numpy.all(mesh.is_boundary_point == [True, True, True, True, True])
    assert numpy.all(mesh.is_boundary_edge[0] == [False, False, False])
    assert numpy.all(mesh.is_boundary_edge[1] == [True, False, True])
    assert numpy.all(mesh.is_boundary_edge[2] == [True, True, True])
    assert numpy.all(
        mesh.is_boundary_edge_gid == [True, True, True, True, True, False, False]
    )
    assert numpy.all(mesh.is_boundary_cell)
