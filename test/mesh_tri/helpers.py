import numpy as np

import meshplex


def assert_mesh_consistency(mesh0, tol=1.0e-14):
    assert np.all(np.logical_xor(mesh0.is_boundary_facet, mesh0.is_interior_facet))

    bpts = np.array(
        [
            mesh0.is_boundary_point,
            mesh0.is_interior_point,
            ~mesh0.is_point_used,
        ]
    )
    assert np.all(np.sum(bpts, axis=0) == 1)

    # consistency check for facets_cells
    assert np.all(mesh0.is_boundary_facet[mesh0.facets_cells["boundary"][0]])
    assert not np.any(mesh0.is_boundary_facet[mesh0.facets_cells["interior"][0]])

    for edge_id, cell_id, local_edge_id in mesh0.facets_cells["boundary"].T:
        assert edge_id == mesh0.cells("facets")[cell_id][local_edge_id]

    for (
        edge_id,
        cell_id0,
        cell_id1,
        local_edge_id0,
        local_edge_id1,
    ) in mesh0.facets_cells["interior"].T:
        assert edge_id == mesh0.cells("facets")[cell_id0][local_edge_id0]
        assert edge_id == mesh0.cells("facets")[cell_id1][local_edge_id1]

    # check consistency of facets_cells_idx with facets_cells
    for edge_id, idx in enumerate(mesh0.facets_cells_idx):
        if mesh0.is_boundary_facet[edge_id]:
            assert edge_id == mesh0.facets_cells["boundary"][0, idx]
        else:
            assert edge_id == mesh0.facets_cells["interior"][0, idx]

    # Assert facets_cells integrity
    for cell_gid, edge_gids in enumerate(mesh0.cells("facets")):
        for edge_gid in edge_gids:
            idx = mesh0.facets_cells_idx[edge_gid]
            if mesh0.is_boundary_facet[edge_gid]:
                assert cell_gid == mesh0.facets_cells["boundary"][1, idx]
            else:
                assert cell_gid in mesh0.facets_cells["interior"][1:3, idx]

    # make sure the edges are opposite of the points
    for cell_gid, (point_ids, edge_ids) in enumerate(
        zip(mesh0.cells("points"), mesh0.cells("facets"))
    ):
        for k in range(len(point_ids)):
            assert set(point_ids) == set(
                [*mesh0.edges["points"][edge_ids][k], point_ids[k]]
            )

    # make sure the is_boundary_point/edge/cell is consistent
    ref_cells = np.any(mesh0.is_boundary_facet_local, axis=0)
    assert np.all(mesh0.is_boundary_cell == ref_cells)
    ref_points = np.zeros(len(mesh0.points), dtype=bool)
    ref_points[mesh0.idx[1][:, mesh0.is_boundary_facet_local]] = True
    assert np.all(mesh0.is_boundary_point == ref_points)

    assert len(mesh0.control_volumes) == len(mesh0.points)
    assert len(mesh0.control_volume_centroids) == len(mesh0.points)

    # TODO add more consistency checks

    # now check the numerical values
    # create a new mesh from the points and cells and compare
    mesh1 = meshplex.Mesh(mesh0.points, mesh0.cells("points"))

    # Can't add those tests since the facet order will be different.
    # TODO bring back
    # if mesh0.facets is None:
    #     mesh0.create_facets()
    # if mesh1.facets is None:
    #     mesh1.create_facets()
    # assert np.all(mesh0.boundary_facets == mesh1.boundary_facets)
    # assert np.all(mesh0.interior_facets == mesh1.interior_facets)
    # assert np.all(mesh0.is_boundary_facet == mesh1.is_boundary_facet)
    # assert np.all(
    #     np.abs(
    #         mesh0.signed_circumcenter_distances - mesh1.signed_circumcenter_distances
    #     )
    #     < tol
    # )

    assert np.all(mesh0.is_point_used == mesh1.is_point_used)
    assert np.all(mesh0.is_boundary_point == mesh1.is_boundary_point)
    assert np.all(mesh0.is_interior_point == mesh1.is_interior_point)
    assert np.all(mesh0.is_boundary_facet_local == mesh1.is_boundary_facet_local)
    assert np.all(mesh0.is_boundary_cell == mesh1.is_boundary_cell)

    assert np.all(np.abs(mesh0.ei_dot_ei - mesh1.ei_dot_ei) < tol)
    assert np.all(np.abs(mesh0.cell_volumes - mesh1.cell_volumes) < tol)
    assert np.all(
        np.abs(mesh0.circumcenter_facet_distances - mesh1.circumcenter_facet_distances)
        < tol
    )

    assert np.all(np.abs(mesh0.signed_cell_volumes - mesh1.signed_cell_volumes) < tol)
    assert np.all(np.abs(mesh0.cell_centroids - mesh1.cell_centroids) < tol)
    assert np.all(np.abs(mesh0.cell_circumcenters - mesh1.cell_circumcenters) < tol)
    assert np.all(np.abs(mesh0.control_volumes - mesh1.control_volumes) < tol)
    assert np.all(np.abs(mesh0.ce_ratios - mesh1.ce_ratios) < tol)

    ipu = mesh0.is_point_used
    assert np.all(
        np.abs(
            mesh0.control_volume_centroids[ipu] - mesh1.control_volume_centroids[ipu]
        )
        < tol
    )


def compute_all_entities(mesh):
    mesh.is_boundary_point
    mesh.is_interior_point
    mesh.is_boundary_facet_local
    mesh.is_boundary_facet
    mesh.is_boundary_cell
    mesh.cell_volumes
    mesh.ce_ratios
    mesh.signed_cell_volumes
    mesh.cell_centroids
    mesh.control_volumes
    mesh.create_facets()
    mesh.facets_cells
    mesh.facets_cells_idx
    mesh.boundary_facets
    mesh.interior_facets
    mesh.cell_circumcenters
    mesh.signed_circumcenter_distances
    mesh.control_volume_centroids

    assert mesh.edges is not None
    assert mesh.subdomains is not {}
    assert mesh._is_interior_point is not None
    assert mesh._is_boundary_point is not None
    assert mesh._is_boundary_facet_local is not None
    assert mesh._is_boundary_facet is not None
    assert mesh._is_boundary_cell is not None
    assert mesh._facets_cells is not None
    assert mesh._facets_cells_idx is not None
    assert mesh._boundary_facets is not None
    assert mesh._interior_facets is not None
    assert mesh._is_point_used is not None
    assert mesh._half_edge_coords is not None
    assert mesh._ei_dot_ei is not None
    assert mesh._volumes is not None
    assert mesh._ce_ratios is not None
    assert mesh._circumcenters is not None
    assert mesh._circumcenter_facet_distances is not None
    assert mesh._signed_circumcenter_distances is not None
    assert mesh._control_volumes is not None
    assert mesh._cell_partitions is not None
    assert mesh._cv_centroids is not None
    assert mesh._signed_cell_volumes is not None
    assert mesh._cell_centroids is not None
