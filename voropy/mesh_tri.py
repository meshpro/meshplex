# -*- coding: utf-8 -*-
#
import numpy
from .base import (
    _base_mesh, _row_dot, compute_tri_areas_and_ce_ratios,
    compute_triangle_circumcenters
    )
from .helpers import grp_start_len


__all__ = ['MeshTri']


def _column_stack(a, b):
    # https://stackoverflow.com/a/39638773/353337
    return numpy.stack([a, b], axis=1)


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
    # pylint: disable=len-as-condition
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
    tol = 1.0e-10
    assert all(abs(ce_ratios[1] - ce_ratios[2]) < tol * ce_ratios[1])
    return ce_ratios[[0, 1]]


# pylint: disable=too-many-instance-attributes
class FlatCellCorrector(object):
    '''Cells can be flat such that the circumcenter is outside the cell, e.g.,

                               p0
                               _^_
                           ___/   \___
                       ___/           \___
                   ___/                   \___
               ___/  \                     /  \___
           ___/       \                   /       \___
          /____________\_________________/____________\
         p1             \       |       /              p2
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
          /____________\_________________/____________\
         p1                                            p2
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


# pylint: disable=too-many-instance-attributes
class MeshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: MeshTri
    '''
    def __init__(self, nodes, cells, flat_cell_correction=None):
        '''Initialization.
        '''
        # Sort cells and nodes, first every row, then the rows themselves. This
        # helps in many downstream applications, e.g., when constructing linear
        # systems with the cells/edges. (When converting to CSR format, the
        # I/J entries must be sorted.)
        # Don't use cells.sort(axis=1) to avoid
        # ```
        # ValueError: sort array is read-only
        # ```
        cells = numpy.sort(cells, axis=1)
        cells = cells[cells[:, 0].argsort()]

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
        is_used[cells] = True
        assert all(is_used)

        self.cells = {
            'nodes': cells
            }

        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._surface_areas = None
        self.edges = None
        self.cell_circumcenters = None
        self._signed_tri_areas = None
        self.subdomains = {}
        self.is_boundary_node = None
        self.is_boundary_edge = None
        self.is_boundary_face = None

        # compute data
        # Create the idx_hierarchy (nodes->edges->cells), i.e., the value of
        # `self.idx_hierarchy[0, 2, 27]` is the index of the node of cell 27,
        # edge 2, node 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`,
        # where `n` is the number of cells. Make sure that the k-th edge is
        # opposite of the k-th point in the triangle.
        self.local_idx = numpy.array([
            [1, 2],
            [2, 0],
            [0, 1],
            ]).T
        # Map idx back to the nodes. This is useful if quantities which are in
        # idx shape need to be added up into nodes (e.g., equation system rhs).
        nds = self.cells['nodes'].T
        self.idx_hierarchy = nds[self.local_idx]

        # The inverted local index.
        # This array specifies for each of the three nodes which edge endpoints
        # correspond to it. For the above local_idx, this should give
        #
        #    [[(1, 1), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (0, 1)]]
        #
        self.local_idx_inv = [
            [tuple(i) for i in zip(*numpy.where(self.local_idx == k))]
            for k in range(3)
            ]

        # Create the corresponding edge coordinates.
        self.half_edge_coords = \
            self.node_coords[self.idx_hierarchy[1]] - \
            self.node_coords[self.idx_hierarchy[0]]

        self.ei_dot_ej = numpy.einsum(
            'ijk, ijk->ij',
            self.half_edge_coords[[1, 2, 0]],
            self.half_edge_coords[[2, 0, 1]]
            )

        e = self.half_edge_coords
        self.ei_dot_ei = numpy.einsum('ijk, ijk->ij', e, e)

        self.cell_volumes, self.ce_ratios_per_half_edge = \
            compute_tri_areas_and_ce_ratios(self.ei_dot_ej)

        self.fcc_type = flat_cell_correction
        if flat_cell_correction is None:
            self.fcc = None
            self.regular_cells = numpy.s_[:]
        else:
            if flat_cell_correction == 'full':
                # All cells with a negative c/e ratio are redone.
                edge_needs_fcc = self.ce_ratios_per_half_edge < 0.0
            else:
                assert flat_cell_correction == 'boundary'
                # This best imitates the classical notion of control volumes.
                # Only cells with a negative c/e ratio on the boundary are
                # redone. Of course, this requires identifying boundary edges
                # first.
                if self.edges is None:
                    self.create_edges()
                edge_needs_fcc = numpy.logical_and(
                    self.ce_ratios_per_half_edge < 0.0,
                    self.is_boundary_edge
                    )

            fcc_local_edge, self.fcc_cells = numpy.where(edge_needs_fcc)
            self.regular_cells = numpy.where(
                ~numpy.any(edge_needs_fcc, axis=0)
                )[0]

            self.fcc = FlatCellCorrector(
                self.cells['nodes'][self.fcc_cells],
                fcc_local_edge,
                self.node_coords
                )
            self.ce_ratios_per_half_edge[:, self.fcc_cells] = \
                self.fcc.get_ce_ratios().T

        return

    def update_node_coordinates(self, X):
        assert self.fcc is None
        assert X.shape == self.node_coords.shape

        self.node_coords = X

        if self.half_edge_coords is not None:
            self.half_edge_coords = (
                self.node_coords[self.idx_hierarchy[1]] -
                self.node_coords[self.idx_hierarchy[0]]
                )

        if self.ei_dot_ej is not None:
            self.ei_dot_ej = numpy.einsum(
                'ijk, ijk->ij',
                self.half_edge_coords[[1, 2, 0]],
                self.half_edge_coords[[2, 0, 1]]
                )

        if self.ei_dot_ei is not None:
            e = self.half_edge_coords
            self.ei_dot_ei = numpy.einsum('ijk, ijk->ij', e, e)

        if self.cell_volumes is not None or self.ce_ratios_per_half_edge is not None:
            self.cell_volumes, self.ce_ratios_per_half_edge = \
                compute_tri_areas_and_ce_ratios(self.ei_dot_ej)

        self._interior_edge_lengths = None
        self.cell_circumcenters = None
        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._surface_areas = None
        self._signed_tri_areas = None
        return

    def get_boundary_vertices(self):
        self.mark_boundary()
        return numpy.where(self.is_boundary_node)[0]

    def get_ce_ratios(self, cell_ids=None):
        if cell_ids is not None:
            return self.ce_ratios_per_half_edge[cell_ids]
        return self.ce_ratios_per_half_edge

    def get_nondelaunay_edges(self):
        if 'edges' not in self.cells:
            self.create_edges()

        neg = self.get_ce_ratios() < 0.0

        print(neg.shape)
        candidate_edges = self.cells['edges'].T[neg]
        print(candidate_edges)
        print(len(candidate_edges))
        print(len(numpy.unique(candidate_edges)))
        exit(1)

        sum_ce_ratios = numpy.zeros(len(candidate_edges))
        numpy.add.at(
            sum_ce_ratios,
            self.cells['edges'].T,
            self.ce_ratios_per_half_edge
            )
        exit(1)
        return

    def get_ce_ratios_per_interior_edge(self):
        if self._interior_ce_ratios is None:
            if 'edges' not in self.cells:
                self.create_edges()

            # self._ce_ratios = numpy.zeros(len(self.edges['nodes']))
            # numpy.add.at(
            #     self._ce_ratios,
            #     self.cells['edges'].T,
            #     self.ce_ratios_per_half_edge
            #     )
            # self._interior_ce_ratios = \
            #     self._ce_ratios[~self.is_boundary_edge_individual]

            # sum up from self.ce_ratios_per_half_edge
            if self._edges_cells is None:
                self._compute_edges_cells()

            self._interior_ce_ratios = \
                numpy.zeros(self._edges_local[2].shape[0])
            for i in [0, 1]:
                # Interior edges = edges with _2_ adjacent cells
                idx = [
                    self._edges_local[2][:, i],
                    self._edges_cells[2][:, i],
                    ]
                self._interior_ce_ratios += self.ce_ratios_per_half_edge[idx]

        return self._interior_ce_ratios

    def get_control_volumes(self):
        if self._control_volumes is None:
            v = self.get_cell_partitions()[..., self.regular_cells]

            # Summing up the arrays first makes the work for numpy.add.at
            # lighter.
            ids = self.cells['nodes'][self.regular_cells].T
            vals = numpy.array([
                sum([v[i] for i in numpy.where(self.local_idx.T == k)[0]])
                for k in range(3)
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

    def get_signed_tri_areas(self):
        '''Signed area of a triangle in 2D.
        '''
        # http://mathworld.wolfram.com/TriangleArea.html
        assert self.node_coords.shape[1] == 2, \
            'Signed areas only make sense for triangles in 2D.'

        if self._signed_tri_areas is None:
            # One could make p contiguous by adding a copy(), but that's not
            # really worth it here.
            p = self.node_coords[self.cells['nodes']].T
            # <https://stackoverflow.com/q/50411583/353337>
            self._signed_tri_areas = (
                + p[0][2] * (p[1][0] - p[1][1])
                + p[0][0] * (p[1][1] - p[1][2])
                + p[0][1] * (p[1][2] - p[1][0])
                ) / 2
        return self._signed_tri_areas

    def mark_boundary(self):
        if self.edges is None:
            self.create_edges()

        self.is_boundary_node = numpy.zeros(len(self.node_coords), dtype=bool)
        self.is_boundary_node[
            self.idx_hierarchy[..., self.is_boundary_edge]
            ] = True

        assert self.is_boundary_edge is not None
        self.is_boundary_face = self.is_boundary_edge
        return

    def create_edges(self):
        '''Set up edge-node and edge-cell relations.
        '''
        # Reshape into individual edges.
        # Sort the columns to make it possible for `unique()` to identify
        # individual edges.
        s = self.idx_hierarchy.shape
        a = numpy.sort(self.idx_hierarchy.reshape(s[0], s[1]*s[2]).T)
        b = numpy.ascontiguousarray(a).view(
            numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
            )
        _, idx, inv, cts = numpy.unique(
            b,
            return_index=True,
            return_inverse=True,
            return_counts=True
            )

        # No edge has more than 2 cells. This assertion fails, for example, if
        # cells are listed twice.
        assert numpy.all(cts < 3)

        self.is_boundary_edge = (cts[inv] == 1).reshape(s[1:])

        self.is_boundary_edge_individual = (cts == 1)

        self.edges = {
            'nodes': a[idx],
            }

        # cell->edges relationship
        self.cells['edges'] = inv.reshape(3, -1).T

        self._edges_cells = None
        self._edges_local = None
        self._edge_gid_to_edge_list = None
        return

    def _compute_edges_cells(self):
        '''This creates interior edge->cells relations. As an upstream
        relation, this is relatively expensive to compute. Luckily, it's not
        necessary for many applications.
        '''
        num_cells = len(self.cells['nodes'])
        num_edges = len(self.edges['nodes'])

        # <https://stackoverflow.com/a/50395231/353337>
        edges_flat = self.cells['edges'].flatten()
        idx_sort = numpy.argsort(edges_flat)
        idx_start, count = grp_start_len(edges_flat[idx_sort])
        res1 = idx_sort[idx_start[count==1]][:, numpy.newaxis]
        idx = idx_start[count==2]
        res2 = numpy.column_stack([idx_sort[idx], idx_sort[idx + 1]])
        self._edges_cells = [
            [],  # no edges with zero adjacent cells
            res1 // 3,
            res2 // 3,
            ]
        self._edges_local = [
            [],  # no edges with zero adjacent cells
            res1 % 3,
            res2 % 3,
            ]

        # Store an index {boundary,interior}_edge -> edge_gid
        self._edge_to_edge_gid = [
            [],
            numpy.where(self.is_boundary_edge_individual)[0],
            numpy.where(~self.is_boundary_edge_individual)[0]
            ]

        # For each edge, store the number of adjacent cells plus the index into
        # the respective edge array.
        self._edge_gid_to_edge_list = numpy.empty((num_edges, 2), dtype=int)
        self._edge_gid_to_edge_list[:, 0] = count
        c1 = (count==1)
        l1 = numpy.sum(c1)
        self._edge_gid_to_edge_list[c1, 1] = numpy.arange(l1)
        c2 = (count==2)
        l2 = numpy.sum(c2)
        self._edge_gid_to_edge_list[c2, 1] = numpy.arange(l2)
        assert l1 + l2 == len(count)

        return

    def get_face_partitions(self):
        # face = edge for triangles.
        # The partition is simply along the center of the edge.
        edge_lengths = self.get_edge_lengths()
        return numpy.array([0.5 * edge_lengths, 0.5 * edge_lengths])

    def get_cell_partitions(self):
        if self._cell_partitions is None:
            # Compute the control volumes. Note that
            #   0.5 * (0.5 * edge_length) * covolume
            # = 0.25 * edge_length**2 * ce_ratio_edge_ratio
            self._cell_partitions = \
                0.25 * self.ei_dot_ei * self.ce_ratios_per_half_edge
        return self._cell_partitions

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
        right_triangle_vols = self.get_cell_partitions()[:, cell_ids]

        node_edges = self.idx_hierarchy[..., cell_ids]

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
#            https://doi.org/10.1002/nme.2187.
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
#         for edge_gid, edge in enumerate(self.edges['cells']):
#             # Compute edge length.
#             node0 = self.edges['nodes'][edge_gid][0]
#             node1 = self.edges['nodes'][edge_gid][1]
#
#             # Compute coedge length.
#             if len(self.edges['cells'][edge_gid]) == 1:
#                 # Boundary edge.
#                 edge_midpoint = 0.5 * (
#                         node_coords2d[node0] +
#                         node_coords2d[node1]
#                         )
#                 cell0 = self.edges['cells'][edge_gid][0]
#                 coedge_midpoint = 0.5 * (
#                         cell_circumcenters2d[cell0] +
#                         edge_midpoint
#                         )
#             elif len(self.edges['cells'][edge_gid]) == 2:
#                 cell0 = self.edges['cells'][edge_gid][0]
#                 cell1 = self.edges['cells'][edge_gid][1]
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
#             coeffs = self.ce_ratios[edge_gid] / \
#                 self.control_volumes[self.edges['nodes'][edge_gid]]
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
        A = 0.5 * numpy.sum(vector_field[self.idx_hierarchy], axis=0)
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
        return numpy.sum(ce_ratios[~self.is_boundary_edge_individual] < 0.0)

    def show(self, *args, **kwargs):
        from matplotlib import pyplot as plt
        self.plot(*args, **kwargs)
        plt.show()
        return

    def save_png(self, filename, *args, **kwargs):
        from matplotlib import pyplot as plt
        self.plot(*args, **kwargs)
        plt.savefig(filename, transparent=False)
        return

    def plot(self,
             show_coedges=True,
             show_centroids=True,
             mesh_color='k',
             boundary_edge_color=None,
             comesh_color=(0.8, 0.8, 0.8),
             show_axes=True):
        '''Show the mesh using matplotlib.
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

        if self.edges is None:
            self.create_edges()

        # Get edges, cut off z-component.
        e = self.node_coords[self.edges['nodes']][:, :, :2]
        # Plot regular edges, mark those with negative ce-ratio red.
        ce_ratios = self.get_ce_ratios_per_edge()
        pos = ce_ratios >= 0
        line_segments0 = LineCollection(e[pos], color=mesh_color)
        ax.add_collection(line_segments0)
        #
        neg = ~pos
        line_segments1 = LineCollection(e[neg], color=new_red)
        ax.add_collection(line_segments1)

        if show_coedges:
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
            e = self.node_coords[
                self.edges['nodes'][self.is_boundary_edge_individual]
                ][:, :, :2]
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
        edge_gids = numpy.where((self.edges['nodes'] == node_id).any(axis=1))[0]
        # ... and plot them
        for node_ids in self.edges['nodes'][edge_gids]:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], 'k')

        # Highlight ce_ratios.
        if show_ce_ratio:
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = self.compute_triangle_circumcenters(
                    X,
                    self.ei_dot_ei,
                    self.ei_dot_ej
                    )

            # Find the cells that contain the vertex
            cell_ids = numpy.where(
                (self.cells['nodes'] == node_id).any(axis=1)
                )[0]

            for cell_id in cell_ids:
                for edge_gid in self.cells['edges'][cell_id]:
                    if node_id not in self.edges['nodes'][edge_gid]:
                        continue
                    node_ids = self.edges['nodes'][edge_gid]
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

    def flip_until_delaunay(self):
        # There are no interior edges with negative covolume-edge ratio
        # when using full flat cell correction.
        assert self.fcc_type != 'full'

        # If all coedge/edge ratios are positive, all cells are Delaunay.
        ce_ratios = self.get_ce_ratios()
        if numpy.all(ce_ratios > 0):
            return False

        # If all _interior_ coedge/edge ratios are positive, all cells are
        # Delaunay.
        if self.is_boundary_edge is None:
            self.create_edges()
        self.mark_boundary()
        if numpy.all(ce_ratios[~self.is_boundary_edge] > 0):
            return False

        needs_flipping = self.get_ce_ratios_per_interior_edge() < 0.0

        num_flip_steps = 0
        while numpy.any(needs_flipping):
            num_flip_steps += 1
            self._flip_edges(needs_flipping)
            needs_flipping = self.get_ce_ratios_per_interior_edge() < 0.0

        return num_flip_steps > 1

    def _flip_edges(self, is_flip_interior_edge):
        '''Flips the given edges.
        '''
        assert self.fcc_type != 'full'

        interior_edges_cells = self._edges_cells[2]
        interior_edges_local = self._edges_local[2]

        # Can only handle the case where each cell has at most one edge to flip.
        # Use `unique` here as there are usually only very edges in the list.
        _, counts = numpy.unique(
            interior_edges_cells[is_flip_interior_edge],
            return_counts=True
            )
        assert numpy.all(counts < 2), 'Can flip at most one edge per cell.'

        update_cell_ids = []
        update_interior_edge_ids = []
        for interior_edge_id in numpy.where(is_flip_interior_edge)[0]:
            adj_cells = interior_edges_cells[interior_edge_id]
            lid = interior_edges_local[interior_edge_id]

            #        3                   3
            #        A                   A
            #       /|\                 / \
            #      / | \               /   \
            #     /  |  \             /  1  \
            #   0/ 0 |   \1   ==>   0/_______\1
            #    \   | 1 /           \       /
            #     \  |  /             \  0  /
            #      \ | /               \   /
            #       \|/                 \ /
            #        V                   V
            #        2                   2
            #
            # Make sure i0 is the cell with the lower node index.
            node_id0 = self.cells['nodes'][adj_cells[0], lid[0]]
            node_id1 = self.cells['nodes'][adj_cells[1], lid[1]]
            if node_id0 < node_id1:
                i0, i1 = 0, 1
            else:
                i0, i1 = 1, 0

            verts = numpy.array([
                self.cells['nodes'][adj_cells[i0], lid[i0]],
                self.cells['nodes'][adj_cells[i1], lid[i1]],
                self.cells['nodes'][adj_cells[i0], (lid[i0] + 1) % 3],
                self.cells['nodes'][adj_cells[i0], (lid[i0] + 2) % 3],
                ])

            # update edges
            edge_gid = self._edge_to_edge_gid[2][interior_edge_id]
            # Vertices are ordered.
            self.edges['nodes'][edge_gid] = verts[[0, 1]]
            # No need to touch self.is_boundary_edge,
            # self.is_boundary_edge_individual; we're only flipping interior
            # edges.

            # Set new cells
            self.cells['nodes'][adj_cells[0]] = verts[[0, 1, 2]]
            self.cells['nodes'][adj_cells[1]] = verts[[0, 1, 3]]

            # Set up new cells->edges relationships.
            # First store the old edges in a dictionary with the global edge
            # indices, then look for the new edge indices in them. Perhaps
            # overkill, there may be a more elegant solution.
            old_edges_idx = numpy.concatenate([
                self.cells['edges'][adj_cells[0]],
                self.cells['edges'][adj_cells[1]],
                ])
            d = {
                tuple(self.edges['nodes'][idx]): idx
                for idx in old_edges_idx
                }
            # Now update cells['edges']
            self.cells['edges'][adj_cells[0]] = numpy.array([
                d[tuple(numpy.sort(verts[[1, 2]]))],
                d[tuple(numpy.sort(verts[[2, 0]]))],
                edge_gid
                ])
            self.cells['edges'][adj_cells[1]] = numpy.array([
                d[tuple(numpy.sort(verts[[1, 3]]))],
                d[tuple(numpy.sort(verts[[3, 0]]))],
                edge_gid
                ])

            # Update the edge->cells relationship. It doesn't change for the
            # edge that was flipped, but for some of the other edges.
            for c0, c1 in [adj_cells, adj_cells[::-1]]:
                for lid, gid in enumerate(self.cells['edges'][c0]):
                    k, idx = self._edge_gid_to_edge_list[gid]

                    c0_idx = numpy.where(self._edges_cells[k][idx] == c0)[0]
                    if len(c0_idx) > 0:
                        i = c0_idx[0]
                        self._edges_local[k][idx][i] = lid
                    else:
                        # If c0 is not in _edges_cells, then the other cell
                        # must be. Swap.
                        i = numpy.where(self._edges_cells[k][idx] == c1)[0][0]
                        self._edges_cells[k][idx][i] = c0
                        self._edges_local[k][idx][i] = lid

            # Schedule the cell ids for updates.
            update_cell_ids.append(adj_cells)
            # Same for edge ids
            k, edge_gids = self._edge_gid_to_edge_list[
                self.cells['edges'][adj_cells].flatten()
                ].T
            update_interior_edge_ids.append(edge_gids[k==2])

        update_cell_ids = numpy.unique(numpy.concatenate(update_cell_ids))
        update_interior_edge_ids = numpy.unique(numpy.concatenate(
            update_interior_edge_ids
            ))

        self._update_cell_values(update_cell_ids, update_interior_edge_ids)
        return

    def _update_cell_values(self, cell_ids, interior_edge_ids):
        '''Updates all sorts of cell information for the given cell IDs.
        '''
        # update idx_hierarchy
        nds = self.cells['nodes'][cell_ids].T
        self.idx_hierarchy[..., cell_ids] = nds[self.local_idx]

        # update self.half_edge_coords
        # TODO report numpy bug for vectorization
        for cell_id in cell_ids:
            self.half_edge_coords[:, cell_id, :] = (
                self.node_coords[self.idx_hierarchy[1, ..., cell_id]] -
                self.node_coords[self.idx_hierarchy[0, ..., cell_id]]
                )

        # update self.ei_dot_ej
        self.ei_dot_ej[:, cell_ids] = numpy.einsum(
            'ijk, ijk->ij',
            self.half_edge_coords[[1, 2, 0]][:, cell_ids],
            self.half_edge_coords[[2, 0, 1]][:, cell_ids]
            )

        # update self.ei_dot_ei
        e = self.half_edge_coords[:, cell_ids]
        self.ei_dot_ei[:, cell_ids] = numpy.einsum('ijk, ijk->ij', e, e)

        # update cell_volumes, ce_ratios_per_half_edge
        cv, ce = compute_tri_areas_and_ce_ratios(self.ei_dot_ej[:, cell_ids])
        self.cell_volumes[cell_ids] = cv
        self.ce_ratios_per_half_edge[:, cell_ids] = ce

        if self._interior_ce_ratios is not None:
            self._interior_ce_ratios[interior_edge_ids] = 0.0
            for i in [0, 1]:
                self._interior_ce_ratios[interior_edge_ids] += \
                    self.ce_ratios_per_half_edge[
                        self._edges_local[2][interior_edge_ids, i],
                        self._edges_cells[2][interior_edge_ids, i]
                        ]

        if self._signed_tri_areas is not None:
            # One could make p contiguous by adding a copy(), but that's not
            # really worth it here.
            p = self.node_coords[self.cells['nodes'][cell_ids]].T
            # <https://stackoverflow.com/q/50411583/353337>
            self._signed_tri_areas[cell_ids] = (
                + p[0][2] * (p[1][0] - p[1][1])
                + p[0][0] * (p[1][1] - p[1][2])
                + p[0][1] * (p[1][2] - p[1][0])
                ) / 2

        # TODO update self._edge_lengths
        assert self._edge_lengths is None

        # TODO update self.cell_circumcenters
        assert self.cell_circumcenters is None

        # TODO update self._control_volumes
        assert self._control_volumes is None

        # TODO update self._cell_partitions
        assert self._cell_partitions is None

        # TODO update self._cv_centroids
        assert self._cv_centroids is None

        # TODO update self._surface_areas
        assert self._surface_areas is None

        # TODO update self.subdomains
        assert self.subdomains == {}

        return
