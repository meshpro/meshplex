import pathlib

import numpy
import pytest

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def assert_mesh_consistency(mesh):
    assert numpy.all(numpy.logical_xor(mesh.is_boundary_edge, mesh.is_interior_edge))

    bpts = numpy.array(
        [
            mesh.is_boundary_point,
            mesh.is_interior_point,
            ~mesh.is_point_used,
        ]
    )
    assert numpy.all(numpy.sum(bpts, axis=0) == 1)

    # consistency check for edges_cells
    assert numpy.all(mesh.is_boundary_edge[mesh.edges_cells["boundary"][0]])
    assert not numpy.any(mesh.is_boundary_edge[mesh.edges_cells["interior"][0]])

    for edge_id, cell_id, local_edge_id in mesh.edges_cells["boundary"].T:
        assert edge_id == mesh.cells["edges"][cell_id][local_edge_id]

    for edge_id, cell_id0, cell_id1, local_edge_id0, local_edge_id1 in mesh.edges_cells[
        "interior"
    ].T:
        assert edge_id == mesh.cells["edges"][cell_id0][local_edge_id0]
        assert edge_id == mesh.cells["edges"][cell_id1][local_edge_id1]

    # check consistency of edges_cells_idx with edges_cells
    for edge_id, idx in enumerate(mesh.edges_cells_idx):
        if mesh.is_boundary_edge[edge_id]:
            assert edge_id == mesh.edges_cells["boundary"][0, idx]
        else:
            assert edge_id == mesh.edges_cells["interior"][0, idx]

    # TODO add more consistency checks


def compute_all_entities(mesh):
    mesh.is_boundary_point
    mesh.is_interior_point
    mesh.is_boundary_edge_local
    mesh.is_boundary_edge
    mesh.is_boundary_cell
    mesh.cell_volumes
    mesh.ce_ratios
    mesh.signed_cell_areas
    mesh.cell_centroids
    mesh.control_volumes
    mesh.create_edges()
    mesh.edges_cells
    mesh.edges_cells_idx
    mesh.boundary_edges
    mesh.interior_edges
    mesh.cell_circumcenters
    mesh.ce_ratios_per_interior_edge
    mesh.control_volume_centroids

    assert mesh._cv_cell_mask is not None
    assert mesh.edges is not None
    assert mesh.subdomains is not {}
    assert mesh._is_interior_point is not None
    assert mesh._is_boundary_point is not None
    assert mesh._is_boundary_edge_local is not None
    assert mesh._is_boundary_edge is not None
    assert mesh._is_boundary_cell is not None
    assert mesh._edges_cells is not None
    assert mesh._edges_cells_idx is not None
    assert mesh._boundary_edges is not None
    assert mesh._interior_edges is not None
    assert mesh._is_point_used is not None
    assert mesh._half_edge_coords is not None
    assert mesh._ei_dot_ei is not None
    assert mesh._ei_dot_ej is not None
    assert mesh._cell_volumes is not None
    assert mesh._ce_ratios is not None
    assert mesh._cell_circumcenters is not None
    assert mesh._interior_ce_ratios is not None
    assert mesh._control_volumes is not None
    assert mesh._cell_partitions is not None
    assert mesh._cv_centroids is not None
    assert mesh._cvc_cell_mask is not None
    assert mesh._signed_cell_areas is not None
    assert mesh._cell_centroids is not None


def assert_mesh_equality(mesh0, mesh1):
    assert numpy.all(mesh0.cells["points"] == mesh1.cells["points"])
    assert numpy.all(numpy.abs(mesh0.points - mesh1.points) < 1.0e-14)

    assert numpy.all(numpy.abs(mesh0.points - mesh1.points) < 1.0e-14)
    assert numpy.all(mesh0.edges["points"] == mesh1.edges["points"])

    # # Assume that in mesh1, the rows are ordered such that the edge indices [0] are in
    # # order. The mesh0 array isn't order in many cases, e.g., if cells were removed and
    # # rows appened the boundary array. Hence, justsort the array in mesh0 and compare.
    # k = numpy.argsort(mesh0.edges_cells["boundary"][0])
    # assert numpy.all(mesh0.edges_cells["boundary"][:, k] == mesh1.edges_cells["boundary"])

    # # The interior edges_cells are already ordered, even after remove_cells(). (As
    # # opposed to boundary edges, there can be no new interior edges, just some are
    # # removed.)
    # print(mesh0.edges_cells["interior"].T)
    # print()
    # print(mesh1.edges_cells["interior"].T)
    # assert numpy.all(mesh0.edges_cells["interior"] == mesh1.edges_cells["interior"])
    # exit(1)
    # # print()
    # # print(mesh0.edges_cells["interior"])
    # # print()
    # # print(mesh1.edges_cells["interior"])
    #
    # # These should not be equal; see reordering above
    # assert numpy.all(mesh0.edges_cells_idx == mesh1.edges_cells_idx)

    assert numpy.all(mesh0.boundary_edges == mesh1.boundary_edges)
    assert numpy.all(mesh0.interior_edges == mesh1.interior_edges)

    assert numpy.all(mesh0.is_point_used == mesh1.is_point_used)
    assert numpy.all(mesh0.is_boundary_point == mesh1.is_boundary_point)
    assert numpy.all(mesh0.is_interior_point == mesh1.is_interior_point)
    assert numpy.all(mesh0.is_boundary_edge_local == mesh1.is_boundary_edge_local)
    assert numpy.all(mesh0.is_boundary_edge == mesh1.is_boundary_edge)
    assert numpy.all(mesh0.is_boundary_cell == mesh1.is_boundary_cell)

    assert numpy.all(numpy.abs(mesh0.ei_dot_ei - mesh1.ei_dot_ei) < 1.0e-14)
    assert numpy.all(numpy.abs(mesh0.ei_dot_ej - mesh1.ei_dot_ej) < 1.0e-14)
    assert numpy.all(numpy.abs(mesh0.cell_volumes - mesh1.cell_volumes) < 1.0e-14)
    assert numpy.all(numpy.abs(mesh0.ce_ratios - mesh1.ce_ratios) < 1.0e-14)
    assert numpy.all(
        numpy.abs(mesh0.ce_ratios_per_interior_edge - mesh1.ce_ratios_per_interior_edge)
        < 1.0e-14
    )
    assert numpy.all(
        numpy.abs(mesh0.signed_cell_areas - mesh1.signed_cell_areas) < 1.0e-14
    )
    assert numpy.all(numpy.abs(mesh0.cell_centroids - mesh1.cell_centroids) < 1.0e-14)
    assert numpy.all(
        numpy.abs(mesh0.cell_circumcenters - mesh1.cell_circumcenters) < 1.0e-14
    )
    assert numpy.all(numpy.abs(mesh0.control_volumes - mesh1.control_volumes) < 1.0e-14)

    ipu = mesh0.is_point_used
    assert numpy.all(
        numpy.abs(
            mesh0.control_volume_centroids[ipu] - mesh1.control_volume_centroids[ipu]
        )
        < 1.0e-14
    )


def get_mesh0():
    # _____________
    # |   _/ \_   |
    # | _/     \_ |
    # |/_________\|
    # |\_       _/|
    # |  \_   _/  |
    # |___ \_/____|
    #
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
    return meshplex.MeshTri(points, cells)


def get_mesh2():
    mesh0 = meshplex.read(this_dir / ".." / "meshes" / "pacman.vtk")
    return meshplex.MeshTri(mesh0.points[:, :2], mesh0.cells["points"])


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
    assert len(mesh.cells["points"]) == 6
    assert len(mesh.edges["points"]) == 13
    # remove a corner cell
    mesh.remove_cells(remove_idx)
    assert len(mesh.cells["points"]) == expected_num_cells
    assert len(mesh.edges["points"]) == expected_num_edges
    assert_mesh_consistency(mesh)


def test_remove_cells_boundary():
    mesh = get_mesh1()

    assert numpy.all(mesh.is_boundary_point == [True, True, True, True, False])
    assert numpy.all(mesh.is_boundary_edge_local[0] == [False, False, False, False])
    assert numpy.all(mesh.is_boundary_edge_local[1] == [False, False, False, True])
    assert numpy.all(mesh.is_boundary_edge_local[2] == [True, True, True, False])
    assert numpy.all(
        mesh.is_boundary_edge == [True, True, False, True, False, True, False, False]
    )
    assert numpy.all(mesh.is_boundary_cell)
    assert numpy.all(mesh.edges_cells_idx == [0, 1, 0, 2, 1, 3, 2, 3])
    # cell id:
    assert numpy.all(mesh.edges_cells["boundary"][1] == [0, 3, 1, 2])
    # local edge:
    assert numpy.all(mesh.edges_cells["boundary"][2] == [2, 1, 2, 2])
    # cell id:
    assert numpy.all(
        mesh.edges_cells["interior"][1:3].T == [[0, 3], [0, 1], [1, 2], [2, 3]]
    )
    # local edge:
    assert numpy.all(
        mesh.edges_cells["interior"][3:5].T == [[1, 2], [0, 1], [0, 1], [0, 0]]
    )

    # now lets remove some cells
    mesh.remove_cells([0])

    assert_mesh_consistency(mesh)

    assert numpy.all(mesh.is_boundary_point)
    assert numpy.all(mesh.is_boundary_edge_local[0] == [False, False, False])
    assert numpy.all(mesh.is_boundary_edge_local[1] == [True, False, True])
    assert numpy.all(mesh.is_boundary_edge_local[2] == [True, True, True])
    assert numpy.all(
        mesh.is_boundary_edge == [True, True, True, True, True, False, False]
    )
    assert numpy.all(mesh.is_boundary_cell)


def test_remove_all():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = [[0, 1, 2]]
    mesh = meshplex.MeshTri(points, cells)
    assert numpy.all(mesh.is_point_used)

    mesh.remove_cells([0])
    assert_mesh_consistency(mesh)
    assert not numpy.any(mesh.is_point_used)


@pytest.mark.parametrize(
    "mesh0, remove_cells",
    [
        # TODO enable all
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

    # recreate the reduced mesh from scratch
    mesh1 = meshplex.MeshTri(mesh0.points, mesh0.cells["points"])
    mesh1.create_edges()

    # check against the original
    assert_mesh_equality(mesh0, mesh1)


if __name__ == "__main__":
    test_remove_cells_boundary()
