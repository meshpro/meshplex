import numpy


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

    # Assert edges_cells integrity
    for cell_gid, edge_gids in enumerate(mesh.cells["edges"]):
        for edge_gid in edge_gids:
            idx = mesh.edges_cells_idx[edge_gid]
            if mesh.is_boundary_edge[edge_gid]:
                assert cell_gid == mesh.edges_cells["boundary"][1, idx]
            else:
                assert cell_gid in mesh.edges_cells["interior"][1:3, idx]

    # TODO add more consistency checks
