# -*- coding: utf-8 -*-
#
import numpy
from voropy.base import \
        _base_mesh, \
        _row_dot, \
        compute_tri_areas_and_ce_ratios, \
        compute_triangle_circumcenters

__all__ = ['MeshTri']


def _column_stack(a, b):
    # http://stackoverflow.com/a/39638773/353337
    return numpy.concatenate([a[:, None], b[:, None]], axis=1)


def lloyd_smoothing(
        mesh,
        tol,
        max_steps=10000,
        fcc_type='full',
        flip_frequency=100,
        verbose=True,
        output_filetype=None
        ):
    from matplotlib import pyplot as plt

    # 2D mesh
    assert all(mesh.node_coords[:, 2] == 0.0)
    assert mesh.fcc is not fcc_type

    boundary_verts = mesh.get_boundary_vertices()

    max_move = tol + 1

    # from matplotlib import pyplot as plt
    # mesh.show()
    # plt.show()

    k = 0
    for k in range(max_steps):
        if max_move < tol:
            break
        if output_filetype:
            if output_filetype == 'png':
                fig = mesh.show(
                        show_ce_ratios=False,
                        show_centroids=False,
                        show_axes=False
                        )
                plt.savefig('lloyd%04d.png' % k)
                plt.close(fig)
            else:
                mesh.write('lloyd%04d.vtu' % k)

        # Flip the edges every so often.
        if k % flip_frequency == 0:
            # We need boundary flat cell correction for flipping. If `full`,
            # all c/e ratios are nonnegative.
            mesh = MeshTri(
                    mesh.node_coords,
                    mesh.cells['nodes'],
                    flat_cell_correction='boundary'
                    )
            ce_ratios = mesh.get_ce_ratios_per_edge()
            while any(ce_ratios < 0.0):
                mesh = flip_edges(mesh, ce_ratios < 0.0)
                ce_ratios = mesh.get_ce_ratios_per_edge()

        # move interior points into centroids
        new_points = mesh.get_control_volume_centroids()
        new_points[boundary_verts] = mesh.node_coords[boundary_verts]
        diff = new_points - mesh.node_coords
        max_move = numpy.sqrt(numpy.max(numpy.sum(diff*diff, axis=1)))

        mesh = MeshTri(
                new_points,
                mesh.cells['nodes'],
                flat_cell_correction='full'
                )

        if verbose:
            print('\nstep: %d' % k)
            print('  maximum move: %.15e' % max_move)

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
            print('  angles (in degrees):\n')
            for i in range(len(hist)):
                print(
                    '         %3d < angle < %3d:   %d'
                    % (bin_edges[i], bin_edges[i+1], hist[i])
                    )

            # av_ce_ratios = numpy.sum(mesh.ce_ratios_per_half_edge.flat) \
            #     / len(mesh.ce_ratios_per_half_edge.flat)
            # max_ce_ratios = numpy.max(mesh.ce_ratios_per_half_edge.flat)
            # print(
            #     '  c/e ratios (min, av, max): %.15e  %.15e  %.15e' %
            #     (min_ce_ratios, av_ce_ratios, max_ce_ratios)
            #     )

    # Flip one last time.
    mesh = MeshTri(
            mesh.node_coords,
            mesh.cells['nodes'],
            flat_cell_correction='boundary'
            )
    ce_ratios = mesh.get_ce_ratios_per_edge()
    while any(ce_ratios < 0.0):
        mesh = flip_edges(mesh, ce_ratios < 0.0)
        ce_ratios = mesh.get_ce_ratios_per_edge()

    if output_filetype:
        if output_filetype == 'png':
            fig = mesh.show(
                    show_ce_ratios=False,
                    show_centroids=False,
                    show_axes=False
                    )
            plt.savefig('lloyd%04d.png' % (k+1))
            plt.close(fig)
        else:
            mesh.write('lloyd%04d.vtu' % (k+1))

    return mesh


def _mirror_point(p0, p1, p2):
    '''For any given triangle p0--p1--p2, this method creates the point p0',
    namely p0 mirrored along the edge p1--p2, and the point q at the
    perpendicular intersection of the mirror.

            p0
          _/| \__
        _/  |    \__
       /    |       \
      p1----|q-------p2
       \_   |     __/
         \_ |  __/
           \| /
           p0'

    '''
    # Create the mirror.
    # q: Intersection point of old and new edge
    # q = p1 + dot(p0-p1, (p2-p1)/||p2-p1||) * (p2-p1)/||p2-p1||
    #   = p1 + dot(p0-p1, p2-p1)/dot(p2-p1, p2-p1) * (p2-p1)
    #
    if len(p0) == 0:
        return numpy.empty(p0.shape), numpy.empty(p0.shape)
    alpha = _row_dot(p0-p1, p2-p1)/_row_dot(p2-p1, p2-p1)
    q = p1 + alpha[:, None] * (p2-p1)
    # p0d = p0 + 2*(q - p0)
    p0d = 2 * q - p0
    return p0d, q


def _isosceles_ce_ratios(p0, p1, p2):
    '''Compute the _two_ covolume-edge length ratios of the isosceles
    triaingle p0, p1, p2; the edges p0---p1 and p0---p2 are assumed to be
    equally long.
               p0
             _/ \_
           _/     \_
         _/         \_
        /             \
       p1-------------p2
    '''
    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0
    assert all(abs(_row_dot(e2, e2) - _row_dot(e1, e1)) < 1.0e-14)

    e_shift1 = numpy.array([e1, e2, e0])
    e_shift2 = numpy.array([e2, e0, e1])
    ei_dot_ej = numpy.einsum('ijk, ijk->ij', e_shift1, e_shift2)

    _, ce_ratios = compute_tri_areas_and_ce_ratios(ei_dot_ej)
    assert all(abs(ce_ratios[1] - ce_ratios[2]) < 1.0e-12)
    return ce_ratios[[0, 1]]


class FlatCellCorrector(object):
    '''Cells can be flat such that the circumcenter is outside the cell, e.g.,

                               p0
                               _^_
                           ___/   \___
                       ___/           \___
                   ___/                   \___
               ___/  \                     /  \___
           ___/       \                   /       \___
          /____________\________________ /____________\
         p1             \       |       /               p2
                         \      |      /
                          \     |     /
                           \    |    /
                            \   |   /
                             \  |  /
                              \ | /
                               \|/
                                V

    This has some funny consequences: The covolume along this edge is negative,
    the area "belonging" to p0 can be overly large etc. While this is mostly of
    cosmetical interest, some applications suffer. For example, Lloyd smoothing
    will fail if a flat edge is on the boundary: p0 will be dragged further
    towards the outside, making the edge even flatter. Also, when building an
    FVM equation system with a negative covolume, the wrong sign might appear
    on the diagonal, rendering the spectrum indefinite.

    Altogether, it sometimes makes sense to adjust a few things: The
    covolume-edge length ratio, the control volumes contributions, the centroid
    contributions etc. Since all of these computations share some data, that we
    could pass around, we might as well do that as part of a class. Enter
    `FlatCellCorrector` to "cut off" the tail.

                               p0
                               _^_
                           ___/   \___
                       ___/           \___
                   ___/                   \___
               ___/  \                     /  \___
           ___/       \                   /       \___
          /____________\________________ /____________\
         p1                                             p2
    '''
    def __init__(self, cells, flat_edge_local_id, node_coords):
        self.cells = cells
        self.flat_edge_local_id = flat_edge_local_id
        self.node_coords = node_coords

        # In each cell, edge k is opposite of vertex k, so p0 is the point
        # opposite of the flat edge.
        self.p0_local_id = self.flat_edge_local_id.copy()
        self.p1_local_id = (self.flat_edge_local_id + 1) % 3
        self.p2_local_id = (self.flat_edge_local_id + 2) % 3

        i = range(len(self.cells))
        self.p0_id = self.cells[i, self.p0_local_id]
        self.p1_id = self.cells[i, self.p1_local_id]
        self.p2_id = self.cells[i, self.p2_local_id]

        self.p0 = self.node_coords[self.p0_id]
        self.p1 = self.node_coords[self.p1_id]
        self.p2 = self.node_coords[self.p2_id]

        ghost, self.q = _mirror_point(self.p0, self.p1, self.p2)

        ce = _isosceles_ce_ratios(
                numpy.concatenate([self.p1, self.p2]),
                numpy.concatenate([self.p0, self.p0]),
                numpy.concatenate([ghost, ghost])
                )

        n = len(self.p0)
        self.ce_ratios1 = ce[:, :n]
        self.ce_ratios2 = ce[:, n:]

        # The ce_ratios should all be greater than 0, but due to round-off
        # errors can be slightly smaller sometimes.
        # assert (self.ce_ratios1 > 0.0).all()
        # assert (self.ce_ratios2 > 0.0).all()

        self.ghostedge_length_2 = _row_dot(ghost - self.p0, ghost - self.p0)
        return

    def get_ce_ratios(self):
        '''Return the covolume-edge length ratios for the flat boundary cells.
        '''
        vals = numpy.empty((len(self.cells), 3), dtype=float)
        i = numpy.arange(len(vals))
        vals[i, self.p0_local_id] = 0.0
        vals[i, self.p1_local_id] = self.ce_ratios2[1]
        vals[i, self.p2_local_id] = self.ce_ratios1[1]
        return vals

    def get_control_volumes(self):
        '''Control volume contributions

                               p0
                               _^_
                           ___/ | \___
                   e2  ___/ /   |   \ \___  e1
                   ___/    /    |    \    \___
               ___/  \ p0 /     |     \ p0 /  \___
           ___/   p1  \  /  p0  | p0   \  /  p2   \___
          /____________\/_______|_______\/____________\
         p1                                           p2
        '''
        e1 = self.p0 - self.p2
        e2 = self.p1 - self.p0

        e1_length2 = _row_dot(e1, e1)
        e2_length2 = _row_dot(e2, e2)

        ids = numpy.stack([
            _column_stack(self.p0_id, self.p0_id),
            _column_stack(self.p0_id, self.p1_id),
            _column_stack(self.p0_id, self.p2_id)
            ], axis=1)

        a = 0.25 * self.ce_ratios1[0] * self.ghostedge_length_2
        b = 0.25 * self.ce_ratios2[0] * self.ghostedge_length_2
        c = 0.25 * self.ce_ratios1[1] * e2_length2
        d = 0.25 * self.ce_ratios2[1] * e1_length2
        vals = numpy.stack([
            _column_stack(a, b),
            _column_stack(c, c),
            _column_stack(d, d)
            ], axis=1)

        return ids, vals

    def surface_areas(self):
        '''In the triangle

                               p0
                               _^_
                           ___/ | \___
                       ___/ /   |   \ \___
                   ___/    /    |    \    \___
               ___/  \    /     |     \    /  \___
           ___/       \  /      |      \  /       \___
          /____________\/__cv1__|__cv2__\/____________\
         p1            q1       q       q2            p2

        associate the lenght dist(p1, q1) with p1, dist(q2, p2) with p2, and
        dist(q1, q2) with p0.
        '''
        ghostedge_length = numpy.sqrt(self.ghostedge_length_2)

        cv1 = self.ce_ratios1[:, 0] * ghostedge_length
        cv2 = self.ce_ratios2[:, 0] * ghostedge_length

        ids0 = _column_stack(self.p0_id, self.p0_id)
        vals0 = _column_stack(cv1, cv2)

        ids1 = _column_stack(self.p1_id, self.p2_id)
        vals1 = _column_stack(
                numpy.linalg.norm(self.q - self.p1) - cv1,
                numpy.linalg.norm(self.q - self.p2) - cv2
                )

        ids = numpy.concatenate([ids0, ids1])
        vals = numpy.concatenate([vals0, vals1])
        return ids, vals

    def integral_x(self):
        '''Computes the integral of x,

          \int_V x,

        over all "atomic" triangles

                               p0
                               _^_
                           ___/ | \___
                       ___/ /   |   \ \___
                   _em2    /    |    \    em1_
               ___/  \    /     |     \    /  \___
           ___/       \  /      |      \  /       \___
          /____________\/_______|_______\/____________\
         p1            q1       q       q2            p2
        '''
        # The long edge is opposite of p0 and has the same local index,
        # likewise for the other edges.
        e0 = self.p2 - self.p1
        e1 = self.p0 - self.p2
        e2 = self.p1 - self.p0

        # The orthogonal projection of the point q1 (and likewise q2) is the
        # midpoint em2 of the edge e2, so
        #
        #     <q1 - p1, (p0 - p1)/||p0 - p1||> = 0.5 * ||p0 - p1||.
        #
        # Setting
        #
        #     q1 = p1 + lambda1 * (p2 - p1)
        #
        # gives
        #
        #     lambda1 = 0.5 * <p0-p1, p0-p1> / <p2-p1, p0-p1>.
        #
        lambda1 = 0.5 * _row_dot(e2, e2) / _row_dot(e0, -e2)
        lambda2 = 0.5 * _row_dot(e1, e1) / _row_dot(e0, -e1)
        q1 = self.p1 + lambda1[:, None] * (self.p2 - self.p1)
        q2 = self.p2 + lambda2[:, None] * (self.p1 - self.p2)

        em1 = 0.5 * (self.p0 + self.p2)
        em2 = 0.5 * (self.p1 + self.p0)

        e1_length2 = _row_dot(e1, e1)
        e2_length2 = _row_dot(e2, e2)

        # triangle areas
        # TODO take from control volume contributions
        area_p0_q_q1 = 0.25 * self.ce_ratios1[0] * self.ghostedge_length_2
        area_p0_q_q2 = 0.25 * self.ce_ratios2[0] * self.ghostedge_length_2
        area_p0_q1_em2 = 0.25 * self.ce_ratios1[1] * e2_length2
        area_p1_q1_em2 = area_p0_q1_em2
        area_p0_q2_em1 = 0.25 * self.ce_ratios2[1] * e1_length2
        area_p2_q2_em1 = area_p0_q2_em1

        # The integral of any linear function over a triangle is the average of
        # the values of the function in each of the three corners, times the
        # area of the triangle.
        ids = numpy.stack([
                _column_stack(self.p0_id, self.p0_id),
                _column_stack(self.p0_id, self.p1_id),
                _column_stack(self.p0_id, self.p2_id)
                ], axis=1)

        vals = numpy.stack([
            _column_stack(
                area_p0_q_q1[:, None] * (self.p0 + self.q + q1) / 3.0,
                area_p0_q_q2[:, None] * (self.p0 + self.q + q2) / 3.0
                ),
            _column_stack(
                area_p0_q1_em2[:, None] * (self.p0 + q1 + em2) / 3.0,
                area_p1_q1_em2[:, None] * (self.p1 + q1 + em2) / 3.0
                ),
            _column_stack(
                area_p0_q2_em1[:, None] * (self.p0 + q2 + em1) / 3.0,
                area_p2_q2_em1[:, None] * (self.p2 + q2 + em1) / 3.0
                ),
            ], axis=1)

        return ids, vals


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
    new_mesh = MeshTri(
        mesh.node_coords,
        mesh.cells['nodes'],
        # Don't actually need that last bit here.
        flat_cell_correction='boundary'
        )

    return new_mesh


class MeshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: MeshTri
    '''
    def __init__(self, nodes, cells, flat_cell_correction=None):
        '''Initialization.
        '''
        super(MeshTri, self).__init__(nodes, cells)

        # Assert that all vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = numpy.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # nodes = nodes[uvertices]
        # ```
        # helps.
        is_used = numpy.zeros(len(nodes), dtype=bool)
        is_used[cells.flat] = True
        assert all(is_used)

        self.cells = {
            'nodes': cells
            }

        self._ce_ratios = None
        self._control_volumes = None
        self._control_volume_contribs = None
        self._cv_centroids = None
        self._surface_areas = None
        self.edges = None
        self.cell_circumcenters = None

        # TODO don't create by default
        self.create_edges()

        # compute data
        # Create the cells->edges->nodes hierarchy. Make sure that the k-th
        # edge is opposite of the k-th point in the triangle.
        nds = self.cells['nodes'].T
        self.node_edge_cells = numpy.stack([
            nds[[1, 2]],
            nds[[2, 0]],
            nds[[0, 1]],
            ], axis=1)

        # Create the corresponding edge coordinates.
        self.half_edge_coords = \
            self.node_coords[self.node_edge_cells[1]] - \
            self.node_coords[self.node_edge_cells[0]]

        e_shift1 = self.half_edge_coords[[1, 2, 0]]
        e_shift2 = self.half_edge_coords[[2, 0, 1]]
        self.ei_dot_ej = numpy.einsum('ijk, ijk->ij', e_shift1, e_shift2)

        e = self.half_edge_coords
        self.ei_dot_ei = numpy.einsum('ijk, ijk->ij', e, e)

        self.cell_volumes, self.ce_ratios_per_half_edge = \
            compute_tri_areas_and_ce_ratios(self.ei_dot_ej)

        if flat_cell_correction is not None:
            assert flat_cell_correction in ['full', 'boundary']
            if flat_cell_correction == 'full':
                # All cells with a negative c/e ratio are redone.
                edge_needs_fcc = self.ce_ratios_per_half_edge < 0.0
            else:  # 'boundary'
                # This best imitates the classical notion of control volumes.
                # Only cells with a negative c/e ratio on the boundary are
                # redone. Of course, this requires identifying boundary edges
                # first.
                if self.edges is None:
                    self.create_edges()
                edge_needs_fcc = numpy.logical_and(
                    self.ce_ratios_per_half_edge < 0.0,
                    self.is_boundary_edge[self.cells['edges']].T
                    )

            fcc_local_edge, self.fcc_cells = numpy.where(edge_needs_fcc)
            self.regular_cells = numpy.where(numpy.logical_not(
                numpy.any(edge_needs_fcc, axis=0)
                ))[0]

            self.fcc = FlatCellCorrector(
                    self.cells['nodes'][self.fcc_cells],
                    fcc_local_edge,
                    self.node_coords
                    )
            self.ce_ratios_per_half_edge[:, self.fcc_cells] = \
                self.fcc.get_ce_ratios().T
        else:
            self.fcc = None
            self.regular_cells = range(len(self.cells['nodes']))

        # TODO don't create by default
        self.mark_default_subdomains()

        return

    def get_boundary_vertices(self):
        self.create_edges()
        self.mark_default_subdomains()
        return self.subdomains['boundary']['vertices']

    def get_ce_ratios(self, cell_ids=None):
        if cell_ids is not None:
            return self.ce_ratios_per_half_edge[cell_ids]
        return self.ce_ratios_per_half_edge

    def get_ce_ratios_per_edge(self):
        if self._ce_ratios is None:
            # sum up from self.ce_ratios_per_half_edge
            cells_edges = self.cells['edges'].T
            self._ce_ratios = numpy.zeros(len(self.edges['nodes']))
            numpy.add.at(
                    self._ce_ratios,
                    cells_edges,
                    self.ce_ratios_per_half_edge
                    )
        return self._ce_ratios

    def get_control_volumes(self):
        if self._control_volumes is None:
            v = self._get_control_volume_contribs()[..., self.regular_cells]
            # Again, make use of the fact that edge k is opposite of node k in
            # every cell. Adding the arrays first makes the work for
            # numpy.add.at lighter.
            ids = self.cells['nodes'][self.regular_cells].T
            vals = numpy.array([
                v[1] + v[2],
                v[2] + v[0],
                v[0] + v[1],
                ])
            control_volume_data = [(ids, vals)]
            if self.fcc is not None:
                control_volume_data.append(self.fcc.get_control_volumes())

            # sum up from self.control_volume_data
            self._control_volumes = numpy.zeros(len(self.node_coords))
            for d in control_volume_data:
                numpy.add.at(self._control_volumes, d[0], d[1])
        return self._control_volumes

    def get_surface_areas(self):
        if self._surface_areas is None:
            self._surface_areas = \
                self._compute_surface_areas(self.regular_cells)
            if self.fcc is not None:
                ffc_ids, ffc_vals = self.fcc.surface_areas()
                numpy.append(self._surface_areas[0], ffc_ids)
                numpy.append(self._surface_areas[1], ffc_vals)
        return self._surface_areas

    def get_control_volume_centroids(self):
        # This function is necessary, e.g., for Lloyd's
        # smoothing <https://en.wikipedia.org/wiki/Lloyd%27s_algorithm>.
        #
        # The centroid of any volume V is given by
        #
        #   c = \int_V x / \int_V 1.
        #
        # The denominator is the control volume. The numerator can be computed
        # by making use of the fact that the control volume around any vertex
        # v_0 is composed of right triangles, two for each adjacent cell.
        if self._cv_centroids is None:
            _, v = self._compute_integral_x(self.regular_cells)
            # Again, make use of the fact that edge k is opposite of node k in
            # every cell. Adding the arrays first makes the work for
            # numpy.add.at lighter.
            ids = self.cells['nodes'][self.regular_cells].T
            vals = numpy.array([
                v[1, 1] + v[0, 2],
                v[1, 2] + v[0, 0],
                v[1, 0] + v[0, 1],
                ])
            centroid_data = [(ids, vals)]
            if self.fcc is not None:
                centroid_data.append(self.fcc.integral_x())
            # add it all up
            self._cv_centroids = numpy.zeros((len(self.node_coords), 3))
            for d in centroid_data:
                numpy.add.at(self._cv_centroids, d[0], d[1])
            # Divide by the control volume
            self._cv_centroids /= self.get_control_volumes()[:, None]
        return self._cv_centroids

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges['nodes'])),
                'cells': range(len(self.cells['nodes'])),
                }

        # Get vertices on the boundary edges
        boundary_edges = numpy.where(self.is_boundary_edge)[0]
        boundary_vertices = numpy.unique(
                self.edges['nodes'][boundary_edges].flatten()
                )

        self.subdomains['boundary'] = {
                'vertices': boundary_vertices,
                'edges': boundary_edges
                }
        return

    def create_edges(self):
        '''Setup edge-node and edge-cell relations.
        '''
        self.cells['nodes'].sort(axis=1)
        # Order the edges such that node 0 doesn't occur in edge 0 etc., i.e.,
        # node k is opposite of edge k.
        a = numpy.concatenate([
            self.cells['nodes'][:, [1, 2]],
            self.cells['nodes'][:, [0, 2]],
            self.cells['nodes'][:, [0, 1]]
            ])

        # Find the unique edges
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv, cts = numpy.unique(
                b,
                return_index=True,
                return_inverse=True,
                return_counts=True
                )
        edge_nodes = a[idx]

        self.is_boundary_edge = (cts == 1)

        self.edges = {
            'nodes': edge_nodes,
            }

        # cell->edges relationship
        num_cells = len(self.cells['nodes'])
        cells_edges = inv.reshape([3, num_cells]).T
        cells_nodes = self.cells['nodes']
        self.cells = {
            'nodes': cells_nodes,
            'edges': cells_edges
            }

        # store inv for possible later use in create_edge_cells
        self._inv = inv

        return

    def compute_edge_cells(self):
        '''This creates edge->cell relations. As an upstream relation, this is
        relatively expensive to compute and hardly ever necessary.
        '''
        num_cells = len(self.cells['nodes'])
        edge_cells = [[] for k in range(len(self.edges['nodes']))]
        for k, edge_id in enumerate(self._inv):
            edge_cells[edge_id].append(k % num_cells)
        return edge_cells

    def _get_control_volume_contribs(self):
        if self._control_volume_contribs is None:
            # Compute the control volumes. Note that
            #   0.5 * (0.5 * edge_length) * covolume
            # = 0.25 * edge_length**2 * ce_ratio_edge_ratio
            self._control_volume_contribs = \
                0.25 * self.ei_dot_ei * self.ce_ratios_per_half_edge
        return self._control_volume_contribs

    def get_cell_circumcenters(self):
        if self.cell_circumcenters is None:
            node_cells = self.cells['nodes'].T
            self.cell_circumcenters = compute_triangle_circumcenters(
                    self.node_coords[node_cells],
                    self.ei_dot_ei,
                    self.ei_dot_ej
                    )
        return self.cell_circumcenters

    def _compute_integral_x(self, cell_ids):
        '''Computes the integral of x,

          \int_V x,

        over all atomic "triangles", i.e., areas cornered by a node, an edge
        midpoint, and a circumcenter.
        '''
        # The integral of any linear function over a triangle is the average of
        # the values of the function in each of the three corners, times the
        # area of the triangle.
        right_triangle_vols = self._get_control_volume_contribs()[:, cell_ids]

        node_edges = self.node_edge_cells[..., cell_ids]

        corner = self.node_coords[node_edges]
        edge_midpoints = 0.5 * (corner[0] + corner[1])
        cc = self.get_cell_circumcenters()[cell_ids]

        average = (corner + edge_midpoints[None] + cc[None, None]) / 3.0

        contribs = right_triangle_vols[None, :, :, None] * average

        return node_edges, contribs

    def _compute_surface_areas(self, cell_ids):
        '''For each edge, one half of the the edge goes to each of the end
        points. Used for Neumann boundary conditions if on the boundary of the
        mesh and transition conditions if in the interior.
        '''
        # Each of the three edges may contribute to the surface areas of all
        # three vertices. Here, only the two adjacent nodes receive a
        # contribution, but other approaches (e.g., the flat cell corrector),
        # may contribute to all three nodes.
        cn = self.cells['nodes'][cell_ids]
        ids = numpy.stack([cn, cn, cn], axis=1)

        half_el = 0.5 * self.get_edge_lengths()[..., cell_ids]
        zero = numpy.zeros([half_el.shape[1]])
        vals = numpy.stack([
            numpy.column_stack([zero, half_el[0], half_el[0]]),
            numpy.column_stack([half_el[1], zero, half_el[1]]),
            numpy.column_stack([half_el[2], half_el[2], zero]),
            ], axis=1)

        return ids, vals

#     def compute_gradient(self, u):
#         '''Computes an approximation to the gradient :math:`\\nabla u` of a
#         given scalar valued function :math:`u`, defined in the node points.
#         This is taken from :cite:`NME2187`,
#
#            Discrete gradient method in solid mechanics,
#            Lu, Jia and Qian, Jing and Han, Weimin,
#            International Journal for Numerical Methods in Engineering,
#            http://dx.doi.org/10.1002/nme.2187.
#         '''
#         if self.cell_circumcenters is None:
#             X = self.node_coords[self.cells['nodes']]
#             self.cell_circumcenters = self.compute_triangle_circumcenters(X)
#
#         if 'cells' not in self.edges:
#             self.edges['cells'] = self.compute_edge_cells()
#
#         # This only works for flat meshes.
#         assert (abs(self.node_coords[:, 2]) < 1.0e-10).all()
#         node_coords2d = self.node_coords[:, :2]
#         cell_circumcenters2d = self.cell_circumcenters[:, :2]
#
#         num_nodes = len(node_coords2d)
#         assert len(u) == num_nodes
#
#         gradient = numpy.zeros((num_nodes, 2), dtype=u.dtype)
#
#         # Create an empty 2x2 matrix for the boundary nodes to hold the
#         # edge correction ((17) in [1]).
#         boundary_matrices = {}
#         for node in self.get_vertices('boundary'):
#             boundary_matrices[node] = numpy.zeros((2, 2))
#
#         for edge_id, edge in enumerate(self.edges['cells']):
#             # Compute edge length.
#             node0 = self.edges['nodes'][edge_id][0]
#             node1 = self.edges['nodes'][edge_id][1]
#
#             # Compute coedge length.
#             if len(self.edges['cells'][edge_id]) == 1:
#                 # Boundary edge.
#                 edge_midpoint = 0.5 * (
#                         node_coords2d[node0] +
#                         node_coords2d[node1]
#                         )
#                 cell0 = self.edges['cells'][edge_id][0]
#                 coedge_midpoint = 0.5 * (
#                         cell_circumcenters2d[cell0] +
#                         edge_midpoint
#                         )
#             elif len(self.edges['cells'][edge_id]) == 2:
#                 cell0 = self.edges['cells'][edge_id][0]
#                 cell1 = self.edges['cells'][edge_id][1]
#                 # Interior edge.
#                 coedge_midpoint = 0.5 * (
#                         cell_circumcenters2d[cell0] +
#                         cell_circumcenters2d[cell1]
#                         )
#             else:
#                 raise RuntimeError(
#                         'Edge needs to have either one or two neighbors.'
#                         )
#
#             # Compute the coefficient r for both contributions
#             coeffs = self.ce_ratios[edge_id] / \
#                 self.control_volumes[self.edges['nodes'][edge_id]]
#
#             # Compute R*_{IJ} ((11) in [1]).
#             r0 = (coedge_midpoint - node_coords2d[node0]) * coeffs[0]
#             r1 = (coedge_midpoint - node_coords2d[node1]) * coeffs[1]
#
#             diff = u[node1] - u[node0]
#
#             gradient[node0] += r0 * diff
#             gradient[node1] -= r1 * diff
#
#             # Store the boundary correction matrices.
#             edge_coords = node_coords2d[node1] - node_coords2d[node0]
#             if node0 in boundary_matrices:
#                 boundary_matrices[node0] += numpy.outer(r0, edge_coords)
#             if node1 in boundary_matrices:
#                 boundary_matrices[node1] += numpy.outer(r1, -edge_coords)
#
#         # Apply corrections to the gradients on the boundary.
#         for k, value in boundary_matrices.items():
#             gradient[k] = numpy.linalg.solve(value, gradient[k])
#
#         return gradient

    def compute_curl(self, vector_field):
        '''Computes the curl of a vector field over the mesh. While the vector
        field is point-based, the curl will be cell-based. The approximation is
        based on

        .. math::
            n\cdot curl(F) = \lim_{A\\to 0} |A|^{-1} <\int_{dGamma}, F> dr;

        see <https://en.wikipedia.org/wiki/Curl_(mathematics)>. Actually, to
        approximate the integral, one would only need the projection of the
        vector field onto the edges at the midpoint of the edges.
        '''
        # Compute the projection of A on the edge at each edge midpoint.
        # Take the average of `vector_field` at the endpoints to get the
        # approximate value at the edge midpoint.
        A = 0.5 * numpy.sum(vector_field[self.node_edge_cells], axis=0)
        # sum of <edge, A> for all three edges
        sum_edge_dot_A = numpy.einsum('ijk, ijk->j', self.half_edge_coords, A)

        # Get normalized vector orthogonal to triangle
        z = numpy.cross(
            self.half_edge_coords[0],
            self.half_edge_coords[1]
            )

        # Now compute
        #
        #    curl = z / ||z|| * sum_edge_dot_A / |A|.
        #
        # Since ||z|| = 2*|A|, one can save a sqrt and do
        #
        #    curl = z * sum_edge_dot_A * 0.5 / |A|^2.
        #
        curl = z * (0.5 * sum_edge_dot_A / self.cell_volumes**2)[..., None]
        return curl

    def num_delaunay_violations(self):
        # Delaunay violations are present exactly on the interior edges where
        # the ce_ratio is negative. Count those.
        ce_ratios = self.get_ce_ratios_per_edge()
        return numpy.sum(ce_ratios[~self.is_boundary_edge] < 0.0)

    def show(
            self,
            show_ce_ratios=True,
            show_centroids=True,
            mesh_color='k',
            boundary_edge_color=None,
            comesh_color=[0.8, 0.8, 0.8],
            show_axes=True
            ):
        '''Show the mesh using matplotlib.

        :param show_ce_ratios: If true, show all ce_ratios of the mesh, too.
        :type show_ce_ratios: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')
        if not show_axes:
            ax.set_axis_off()

        xmin = numpy.amin(self.node_coords[:, 0])
        xmax = numpy.amax(self.node_coords[:, 0])
        ymin = numpy.amin(self.node_coords[:, 1])
        ymax = numpy.amax(self.node_coords[:, 1])

        width = xmax - xmin
        xmin -= 0.1 * width
        xmax += 0.1 * width

        height = ymax - ymin
        ymin -= 0.1 * height
        ymax += 0.1 * height

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        new_red = '#d62728'  # mpl 2.0 default red

        # Get edges, cut off z-component.
        e = self.node_coords[self.edges['nodes']][:, :, :2]
        # Plot regular edges, mark those with negative ce-ratio red.
        ce_ratios = self.get_ce_ratios_per_edge()
        pos = ce_ratios >= 0
        line_segments0 = LineCollection(e[pos], color=mesh_color)
        ax.add_collection(line_segments0)
        #
        neg = numpy.logical_not(pos)
        line_segments1 = LineCollection(e[neg], color=new_red)
        ax.add_collection(line_segments1)

        if show_ce_ratios:
            # Connect all cell circumcenters with the edge midpoints
            cc = self.get_cell_circumcenters()

            edge_midpoints = 0.5 * (
                self.node_coords[self.edges['nodes'][:, 0]] +
                self.node_coords[self.edges['nodes'][:, 1]]
                )

            # Plot connection of the circumcenter to the midpoint of all three
            # axes.
            a = numpy.stack([
                    cc[:, :2],
                    edge_midpoints[self.cells['edges'][:, 0], :2]
                    ], axis=1)
            b = numpy.stack([
                    cc[:, :2],
                    edge_midpoints[self.cells['edges'][:, 1], :2]
                    ], axis=1)
            c = numpy.stack([
                    cc[:, :2],
                    edge_midpoints[self.cells['edges'][:, 2], :2]
                    ], axis=1)

            line_segments = LineCollection(
                numpy.concatenate([a, b, c]),
                color=comesh_color
                )
            ax.add_collection(line_segments)

        if boundary_edge_color:
            boundary_edges = self.subdomains['boundary']['edges']
            e = self.node_coords[self.edges['nodes'][boundary_edges]][:, :, :2]
            line_segments1 = LineCollection(e, color=boundary_edge_color)
            ax.add_collection(line_segments1)

        if show_centroids:
            centroids = self.get_control_volume_centroids()
            ax.plot(
                centroids[:, 0],
                centroids[:, 1],
                linestyle='',
                marker='.',
                color=new_red,
                )

        return fig

    def show_vertex(self, node_id, show_ce_ratio=True):
        '''Plot the vicinity of a node and its ce_ratio.

        :param node_id: Node ID of the node to be shown.
        :type node_id: int

        :param show_ce_ratio: If true, shows the ce_ratio of the node, too.
        :type show_ce_ratio: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        plt.axis('equal')

        # Find the edges that contain the vertex
        edge_ids = numpy.where((self.edges['nodes'] == node_id).any(axis=1))[0]
        # ... and plot them
        for node_ids in self.edges['nodes'][edge_ids]:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], 'k')

        # Highlight ce_ratios.
        if show_ce_ratio:
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = \
                    self.compute_triangle_circumcenters(
                            X,
                            self.ei_dot_ei,
                            self.ei_dot_ej
                            )

            # Find the cells that contain the vertex
            cell_ids = numpy.where(
                (self.cells['nodes'] == node_id).any(axis=1)
                )[0]

            for cell_id in cell_ids:
                for edge_id in self.cells['edges'][cell_id]:
                    if node_id not in self.edges['nodes'][edge_id]:
                        continue
                    node_ids = self.edges['nodes'][edge_id]
                    edge_midpoint = 0.5 * (
                            self.node_coords[node_ids[0]] +
                            self.node_coords[node_ids[1]]
                            )
                    p = _column_stack(
                        self.cell_circumcenters[cell_id],
                        edge_midpoint
                        )
                    q = numpy.column_stack([
                        self.cell_circumcenters[cell_id],
                        edge_midpoint,
                        self.node_coords[node_id]
                        ])
                    ax.fill(q[0], q[1], color='0.5')
                    ax.plot(p[0], p[1], color='0.7')
        return
