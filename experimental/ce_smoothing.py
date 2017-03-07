#
# This is an attempt to use the sum of the c/e ratios for triangular mesh
# smoothing.
# Works fine for very small meshes, but for larger meshes, it turns out that
# that the objective functional has many -- too many -- local minima that have
# the iteration trapped in a point away from the minimum.  Lloyd's smoothing is
# clearly superior.
#
from . import mesh_tri
import numpy


def _objective(mesh):
    double_vol = 2 * mesh.cell_volumes
    num_cells = len(mesh.cell_volumes)
    return (
        -numpy.sum(numpy.sum(mesh.ei_dot_ej, axis=0) / double_vol)
        / num_cells
        )


def _grad(mesh):
    double_vol = 2 * mesh.cell_volumes

    s = numpy.sum(mesh.ei_dot_ej, axis=0)
    e = mesh.half_edge_coords

    ei_dot_ei = numpy.einsum('ijk, ijk->ij', e, e)
    grad_x0 = (
        + (e[1] - e[2]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[0][..., None] * (e[1] - e[2])
            + (mesh.ei_dot_ej[1] - mesh.ei_dot_ej[2])[..., None] * e[0]
            )
        ) / double_vol[..., None]**3
    grad_x1 = (
        + (e[2] - e[0]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[1][..., None] * (e[2] - e[0])
            + (mesh.ei_dot_ej[2] - mesh.ei_dot_ej[0])[..., None] * e[1]
            )
        ) / double_vol[..., None]**3
    grad_x2 = (
        + (e[0] - e[1]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[2][..., None] * (e[0] - e[1])
            + (mesh.ei_dot_ej[0] - mesh.ei_dot_ej[1])[..., None] * e[2]
            )
        ) / double_vol[..., None]**3

    grad_stack = numpy.array([
        grad_x0, grad_x1, grad_x2
        ])

    # add up all the contributions
    grad = numpy.zeros(mesh.node_coords.shape)
    numpy.add.at(grad, mesh.cells['nodes'].T, grad_stack)

    return grad


def flip_until_delaunay(mesh):
    # We need boundary flat cell correction for flipping. If `full`,
    # all c/e ratios are nonnegative.
    mesh = mesh_tri.MeshTri(
            mesh.node_coords,
            mesh.cells['nodes'],
            flat_cell_correction='boundary'
            )
    ce_ratios = mesh.get_ce_ratios_per_edge()
    while any(ce_ratios < 0.0):
        mesh = flip_edges(mesh, ce_ratios < 0.0)
        ce_ratios = mesh.get_ce_ratios_per_edge()
    return mesh


def flip_edges(mesh, is_flip_edge):
    '''Creates a new mesh by flipping those interior edges which have a
    negative covolume (i.e., a negative covolume-edge length ratio). The
    resulting mesh is Delaunay.
    '''
    is_flip_edge_per_cell = is_flip_edge[mesh.cells['edges']]

    # can only handle the case where each cell has at most one edge to flip
    count = numpy.sum(is_flip_edge_per_cell, axis=1)
    assert all(count <= 1)

    # new cells
    edge_cells = mesh.compute_edge_cells()
    flip_edges = numpy.where(is_flip_edge)[0]
    new_cells = numpy.empty((len(flip_edges), 2, 3), dtype=int)
    for k, flip_edge in enumerate(flip_edges):
        adj_cells = edge_cells[flip_edge]
        assert len(adj_cells) == 2
        # The local edge ids are opposite of the local vertex with the same
        # id.
        cell0_local_edge_id = numpy.where(
            is_flip_edge_per_cell[adj_cells[0]]
            )[0]
        cell1_local_edge_id = numpy.where(
            is_flip_edge_per_cell[adj_cells[1]]
            )[0]

        #     0
        #     /\
        #    /  \
        #   / 0  \
        # 2/______\3
        #  \      /
        #   \  1 /
        #    \  /
        #     \/
        #      1
        verts = [
            mesh.cells['nodes'][adj_cells[0], cell0_local_edge_id],
            mesh.cells['nodes'][adj_cells[1], cell1_local_edge_id],
            mesh.cells['nodes'][adj_cells[0], (cell0_local_edge_id + 1) % 3],
            mesh.cells['nodes'][adj_cells[0], (cell0_local_edge_id + 2) % 3],
            ]
        new_cells[k, 0] = [verts[0], verts[1], verts[2]]
        new_cells[k, 1] = [verts[0], verts[1], verts[3]]

    # find cells that can stay
    is_good_cell = numpy.all(
            numpy.logical_not(is_flip_edge_per_cell),
            axis=1
            )

    mesh.cells['nodes'] = numpy.concatenate([
        mesh.cells['nodes'][is_good_cell],
        new_cells[:, 0, :],
        new_cells[:, 1, :]
        ])

    # Create and return new mesh.
    new_mesh = mesh_tri.MeshTri(
        mesh.node_coords,
        mesh.cells['nodes'],
        # Don't actually need that last bit here.
        flat_cell_correction='boundary'
        )

    return new_mesh


def smooth(mesh):
    # x = numpy.array([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [1.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.1, 0.5, 0.0],
    #     ])
    # cells = numpy.array([
    #     [0, 1, 4],
    #     [1, 2, 4],
    #     [2, 3, 4],
    #     [3, 0, 4],
    #     ])
    # mesh = mesh_tri.MeshTri(x, cells)
    boundary_verts = mesh.get_boundary_vertices()
    x = mesh.node_coords.copy()
    x[:, :2] += 1.0e-1 * (
        numpy.random.rand(len(mesh.node_coords), 2) - 0.5
        )
    x[boundary_verts] = mesh.node_coords[boundary_verts]
    mesh = mesh_tri.MeshTri(x, mesh.cells['nodes'])
    t = 1.0e-3
    for k in range(1000):
        mesh = flip_until_delaunay(mesh)
        x = mesh.node_coords.copy()
        x -= t * _grad(mesh)
        x[boundary_verts] = mesh.node_coords[boundary_verts]
        mesh = mesh_tri.MeshTri(x, mesh.cells['nodes'])
        mesh.write('smoo%04d.vtu' % k)
        print(_objective(mesh))
    return mesh
