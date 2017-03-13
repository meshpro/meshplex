# -*- coding: utf-8 -*-
#
from __future__ import print_function
from .mesh_tri import MeshTri
from matplotlib import pyplot as plt
import numpy


def flip_until_delaunay(mesh):
    fcc_type = mesh.fcc_type
    if fcc_type is not None:
        # No flat_cell_correction when flipping.
        mesh = MeshTri(
                mesh.node_coords,
                mesh.cells['nodes'],
                flat_cell_correction=None
                )
    mesh.create_edges()
    needs_flipping = numpy.logical_and(
        numpy.logical_not(mesh.is_boundary_edge),
        mesh.get_ce_ratios_per_edge() < 0.0
        )
    is_flipped = any(needs_flipping)
    k = 0
    while any(needs_flipping):
        k += 1
        mesh = flip_edges(mesh, needs_flipping)
        #
        mesh.create_edges()
        needs_flipping = numpy.logical_and(
            numpy.logical_not(mesh.is_boundary_edge),
            mesh.get_ce_ratios_per_edge() < 0.0
            )

    # Translate back to input fcc_type.
    if fcc_type is not None:
        mesh = MeshTri(
                mesh.node_coords,
                mesh.cells['nodes'],
                flat_cell_correction=fcc_type
                )
    return mesh, is_flipped


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

    # Create new mesh to make sure that all entities are computed again.
    new_mesh = MeshTri(
        mesh.node_coords,
        mesh.cells['nodes'],
        flat_cell_correction=mesh.fcc_type
        )

    return new_mesh


def _gather_stats(mesh):
    # The cosines of the angles are the negative dot products of
    # the normalized edges adjacent to the angle.
    norms = numpy.sqrt(mesh.ei_dot_ei)
    normalized_ei_dot_ej = numpy.array([
        mesh.ei_dot_ej[0] / norms[1] / norms[2],
        mesh.ei_dot_ej[1] / norms[2] / norms[0],
        mesh.ei_dot_ej[2] / norms[0] / norms[1],
        ])
    angles = numpy.arccos(-normalized_ei_dot_ej) \
        / (2 * numpy.pi) * 360.0

    hist, bin_edges = numpy.histogram(
        angles,
        bins=numpy.linspace(0.0, 180.0, num=19, endpoint=True)
        )
    return hist, bin_edges


def _print_stats(data_list):
    # make sure that all data sets have the same length
    n = len(data_list[0][0])
    for data in data_list:
        assert len(data[0]) == n

    # find largest hist value
    max_val = max([max(data[0]) for data in data_list])
    digits_max_val = len(str(max_val))

    print('  angles (in degrees):\n')
    for i in range(n):
        for data in data_list:
            hist, bin_edges = data
            tple = (bin_edges[i], bin_edges[i+1], hist[i])
            fmt = '         %%3d < angle < %%3d:   %%%dd' % digits_max_val
            print(fmt % tple, end='')
        print('\n', end='')
    return


def _write(mesh, filetype, k):
    if filetype == 'png':
        fig = mesh.plot(
                show_coedges=False,
                show_centroids=False,
                show_axes=False
                )
        fig.suptitle('step %d' % k, fontsize=20)
        plt.savefig('lloyd%04d.png' % k)
        plt.close(fig)
    else:
        mesh.write('lloyd%04d.vtu' % k)


def lloyd(
        mesh,
        tol,
        max_steps=10000,
        fcc_type='full',
        flip_frequency=0,
        verbose=True,
        output_filetype=None
        ):

    # 2D mesh
    assert all(mesh.node_coords[:, 2] == 0.0)
    assert mesh.fcc_type == fcc_type

    boundary_verts = mesh.get_boundary_vertices()

    max_move = tol + 1

    initial_stats = _gather_stats(mesh)

    next_flip_at = 0
    flip_skip = 1
    for k in range(max_steps):
        if max_move < tol:
            break
        if output_filetype:
            _write(mesh, output_filetype, k)

        if k == next_flip_at:
            mesh, is_flipped = flip_until_delaunay(mesh)
            if flip_frequency > 0:
                # fixed flip frequency
                flip_skip = flip_frequency
            else:
                # If the mesh needed flipping, flip again next time. Otherwise
                # double the interval.
                if is_flipped:
                    flip_skip = 1
                else:
                    flip_skip *= 2
            next_flip_at = k + flip_skip

        # move interior points into centroids
        new_points = mesh.get_control_volume_centroids()
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]
        diff = new_points - mesh.node_coords
        max_move = numpy.sqrt(numpy.max(numpy.sum(diff*diff, axis=1)))

        mesh = MeshTri(
                new_points,
                mesh.cells['nodes'],
                flat_cell_correction=fcc_type
                )

        if verbose:
            print('\nstep: %d' % k)
            print('  maximum move: %.15e' % max_move)
            _print_stats([_gather_stats(mesh)])

    # Flip one last time.
    mesh, _ = flip_until_delaunay(mesh)

    if verbose:
        print('\nFinal:' + 35*' ' + 'Initial:')
        _print_stats([
            _gather_stats(mesh),
            initial_stats
            ])

    if output_filetype:
        _write(mesh, output_filetype, k+1)

    return mesh
