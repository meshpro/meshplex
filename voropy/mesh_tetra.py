# -*- coding: utf-8 -*-
#
import numpy
from voropy.base import \
        _base_mesh, \
        _row_dot, \
        compute_triangle_circumcenters

__all__ = ['MeshTetra']


class MeshTetra(_base_mesh):
    '''Class for handling tetrahedral meshes.

    .. inheritance-diagram:: MeshTetra
    '''
    def __init__(self, node_coords, cells, mode='geometric'):
        '''Initialization.
        '''
        # Sort cells and nodes, first every row, then the rows themselves. This
        # helps in many downstream applications, e.g., when constructing linear
        # systems with the cells/edges. (When converting to CSR format, the
        # I/J entries must be sorted.)
        cells.sort(axis=1)
        cells = cells[cells[:, 0].argsort()]

        super(MeshTetra, self).__init__(node_coords, cells)

        # Assert that all vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = numpy.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # nodes = nodes[uvertices]
        # ```
        # helps.
        is_used = numpy.zeros(len(node_coords), dtype=bool)
        is_used[cells.flat] = True
        assert all(is_used)

        self.cells = {
            'nodes': cells
            }

        self._mode = mode
        self._ce_ratios = None
        self._control_volumes = None
        self._circumcenters = None
        self.subdomains = {}

        # Arrange the node_face_cells such that node k is opposite of face k in
        # each cell.
        nds = self.cells['nodes'].T
        idx = numpy.array([
            [1, 2, 3],
            [2, 3, 0],
            [3, 0, 1],
            [0, 1, 2],
            ]).T
        self.node_face_cells = nds[idx]

        # Arrange the idx_hierarchy (node->edge->face->cells) such that
        #
        #   * node k is opposite of edge k in each face,
        #   * duplicate edges are in the same spot of the each of the faces,
        #   * all nodes are in order ([1, 2], [2, 3], [3, 1]),
        #   * face edges form a circle and the curl points in the same
        #     direction (either all outside or all inside of the cell)
        #
        self.local_idx = numpy.array([
            [[2, 3], [3, 1], [1, 2]],
            [[3, 2], [2, 0], [0, 3]],
            [[0, 1], [1, 3], [3, 0]],
            [[1, 0], [0, 2], [2, 1]],
            ]).T
        self.idx_hierarchy = nds[self.local_idx]

        # The inverted local index.
        # This array specifies for each of the three nodes which edge endpoints
        # correspond to it.
        self.local_idx_inv = [
            [tuple(i) for i in zip(*numpy.where(self.local_idx == node_idx))]
            for node_idx in range(4)
            ]

        # create ei_dot_ei, ei_dot_ej
        self.edge_coords = \
            self.node_coords[self.idx_hierarchy[1]] - \
            self.node_coords[self.idx_hierarchy[0]]
        self.ei_dot_ei = numpy.einsum(
                'ijkl, ijkl->ijk',
                self.edge_coords,
                self.edge_coords
                )
        self.ei_dot_ej = numpy.einsum(
            'ijkl, ijkl->ijk',
            self.edge_coords[[1, 2, 0]],
            self.edge_coords[[2, 0, 1]]
            # This is equivalent:
            # numpy.roll(self.edge_coords, 1, axis=0),
            # numpy.roll(self.edge_coords, 2, axis=0),
            )

        self._ce_ratios = self._compute_ce_ratios_geometric()

        return

    def get_ce_ratios(self):
        if self._ce_ratios is None:
            assert self._mode in ['geometric', 'algebraic']
            if self._mode == 'geometric':
                self._ce_ratios = self._compute_ce_ratios_geometric()
            else:  # 'algebraic'
                num_edges = len(self.edges['nodes'])
                self._ce_ratios = numpy.zeros(num_edges, dtype=float)
                raise RuntimeError('Disabled')
                idx, vals = self._compute_ce_ratios_algebraic()
                numpy.add.at(self._ce_ratios, idx, vals)
                self.circumcenter_face_distances = None
        return self._ce_ratios

    def mark_boundary(self):
        if 'faces' not in self.cells:
            self.create_cell_face_relationships()

        # Get vertices on the boundary faces
        boundary_faces = numpy.where(self.is_boundary_face)[0]
        boundary_vertices = numpy.unique(
                self.faces['nodes'][boundary_faces].flatten()
                )
        # boundary_edges = numpy.unique(
        #         self.faces['edges'][boundary_faces].flatten()
        #         )

        self.subdomains['boundary'] = {
                'vertices': boundary_vertices,
                # 'edges': boundary_edges,
                'faces': boundary_faces
                }
        return

    def create_cell_face_relationships(self):
        # All possible faces.
        # Face k is opposite of node k in each cell.
        # Make sure that the indices in each row are in ascending order. This
        # makes it easier to find unique rows.
        sorted_nds = numpy.sort(self.cells['nodes'], axis=1).T
        a = numpy.hstack([
            sorted_nds[[1, 2, 3]],
            sorted_nds[[0, 2, 3]],
            sorted_nds[[0, 1, 3]],
            sorted_nds[[0, 1, 2]]
            ]).T

        # Find the unique faces
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv, cts = numpy.unique(
                b,
                return_index=True,
                return_inverse=True,
                return_counts=True
                )
        face_nodes = a[idx]

        self.is_boundary_face = (cts == 1)

        self.faces = {
            'nodes': face_nodes
            }

        # cell->faces relationship
        num_cells = len(self.cells['nodes'])
        cells_faces = inv.reshape([4, num_cells]).T
        self.cells['faces'] = cells_faces

        # Store the opposing nodes too
        self.cells['opposing vertex'] = self.cells['nodes']

        # save for create_edge_cells
        self._inv_faces = inv

        return

    def create_face_edge_relationships(self):
        a = numpy.vstack([
            self.faces['nodes'][:, [1, 2]],
            self.faces['nodes'][:, [2, 0]],
            self.faces['nodes'][:, [0, 1]]
            ])

        # Find the unique edges
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv = numpy.unique(
                b,
                return_index=True,
                return_inverse=True
                )
        edge_nodes = a[idx]

        self.edges = {
            'nodes': edge_nodes
            }

        # face->edge relationship
        num_faces = len(self.faces['nodes'])
        face_edges = inv.reshape([3, num_faces]).T
        self.faces['edges'] = face_edges

        return

    def _compute_cell_circumcenters(self):
        '''Computes the center of the circumsphere of each cell.
        '''
        cell_coords = self.node_coords[self.cells['nodes']]

        # This used to be
        # ```
        # a = cell_coords[:, 1, :] - cell_coords[:, 0, :]
        # b = cell_coords[:, 2, :] - cell_coords[:, 0, :]
        # c = cell_coords[:, 3, :] - cell_coords[:, 0, :]
        # a_cross_b = numpy.cross(a, b)
        # b_cross_c = numpy.cross(b, c)
        # c_cross_a = numpy.cross(c, a)
        # ```
        # The array X below unified a, b, c.
        X = cell_coords[:, [1, 2, 3], :] - cell_coords[:, [0], :]
        X_dot_X = numpy.einsum('ijk, ijk->ij', X, X)
        X_shift = cell_coords[:, [2, 3, 1], :] - cell_coords[:, [0], :]
        X_cross_Y = numpy.cross(X, X_shift)

        a = X[:, 0, :]
        a_dot_a = X_dot_X[:, 0]
        b_dot_b = X_dot_X[:, 1]
        c_dot_c = X_dot_X[:, 2]
        a_cross_b = X_cross_Y[:, 0, :]
        b_cross_c = X_cross_Y[:, 1, :]
        c_cross_a = X_cross_Y[:, 2, :]

        # Compute scalar triple product without using cross.
        # The product is highly symmetric, so it's a little funny if there
        # should be no single einsum to compute it; see
        # <http://stackoverflow.com/q/42158228/353337>.
        omega = _row_dot(a, b_cross_c)

        self._circumcenters = cell_coords[:, 0, :] + (
                b_cross_c * a_dot_a[:, None] +
                c_cross_a * b_dot_b[:, None] +
                a_cross_b * c_dot_c[:, None]
                ) / (2.0 * omega[:, None])

        return

# Question:
# We're looking for an explicit expression for the algebraic c/e ratios. Might
# it be that, analogous to the triangle dot product, the "triple product" has
# something to do with it?
# "triple product": Project one edge onto the plane spanned by the two others.
#
#     def _compute_ce_ratios_algebraic(self):
#         # Precompute edges.
#         edges = \
#             self.node_coords[self.edges['nodes'][:, 1]] - \
#             self.node_coords[self.edges['nodes'][:, 0]]
#
#         # create cells -> edges
#         num_cells = len(self.cells['nodes'])
#         cells_edges = numpy.empty((num_cells, 6), dtype=int)
#         for cell_id, face_ids in enumerate(self.cells['faces']):
#             edges_set = set(self.faces['edges'][face_ids].flatten())
#             cells_edges[cell_id] = list(edges_set)
#
#         self.cells['edges'] = cells_edges
#
#         # Build the equation system:
#         # The equation
#         #
#         # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
#         #
#         # has to hold for all vectors u in the plane spanned by the edges,
#         # particularly by the edges themselves.
#         cells_edges = edges[self.cells['edges']]
#         # <http://stackoverflow.com/a/38110345/353337>
#         A = numpy.einsum('ijk,ilk->ijl', cells_edges, cells_edges)
#         A = A**2
#
#         # Compute the RHS  cell_volume * <edge, edge>.
#         # The dot product <edge, edge> is also on the diagonals of A (before
#         # squaring), but simply computing it again is cheaper than extracting
#         # it from A.
#         edge_dot_edge = _row_dot(edges, edges)
#         rhs = edge_dot_edge[self.cells['edges']] \
#             * self.cell_volumes[..., None]
#
#         # Solve all k-by-k systems at once ("broadcast"). (`k` is the number
#         # of edges per simplex here.)
#         # If the matrix A is (close to) singular if and only if the cell is
#         # (close to being) degenerate. Hence, it has volume 0, and so all the
#         # edge coefficients are 0, too. Hence, do nothing.
#         sol = numpy.linalg.solve(A, rhs)
#
#         return self.cells['edges'], sol

    def _compute_ce_ratios_geometric(self):

        # opposing nodes, faces
        v_op = self.cells['nodes'].T
        v = self.node_face_cells

        e = numpy.array([
            self.node_coords[v[0]] - self.node_coords[v_op],
            self.node_coords[v[1]] - self.node_coords[v_op],
            self.node_coords[v[2]] - self.node_coords[v_op],
            self.node_coords[v[1]] - self.node_coords[v[0]],
            self.node_coords[v[2]] - self.node_coords[v[1]],
            self.node_coords[v[0]] - self.node_coords[v[2]],
            ])
        ei_dot_ej = numpy.einsum('ijkl, hjkl->ihjk', e, e)

        # This is the reference expression.
        # a = (
        #     2 * _my_dot(x0_cross_x1, x2)**2 -
        #     _my_dot(
        #         x0_cross_x1 + x1_cross_x2 + x2_cross_x0,
        #         x0_cross_x1 * x2_dot_x2[..., None] +
        #         x1_cross_x2 * x0_dot_x0[..., None] +
        #         x2_cross_x0 * x1_dot_x1[..., None]
        #     )) / (12.0 * face_areas)

        # Note that
        #
        #    6*tet_volume = abs(<x0 x x1, x2>)
        #                 = abs(<x1 x x2, x0>)
        #                 = abs(<x2 x x0, x1>).
        #
        # Also,
        #
        #    <a x b, c x d> = <a, c> <b, d> - <a, d> <b, c>.
        #
        # All those dot products can probably be cleaned up good.
        # TODO simplify

        # This expression is from brute_simplify
        # print(self.ei_dot_ej.shape)
        # ei_dot_ej_shift1 = ei_dot_ej[[1, 2, 0]]
        # ei_dot_ej_shift2 = ei_dot_ej[[2, 0, 1]]
        # ei_dot_ej_shift1 * ei_dot_ej_shift2
        # exit(1)
        zeta = (
            + ei_dot_ej[0, 2] * ei_dot_ej[3, 5] * ei_dot_ej[5, 4]
            + ei_dot_ej[0, 1] * ei_dot_ej[3, 5] * ei_dot_ej[3, 4]
            + ei_dot_ej[1, 2] * ei_dot_ej[3, 4] * ei_dot_ej[4, 5]
            + self.ei_dot_ej[0] * self.ei_dot_ej[1] * self.ei_dot_ej[2]
            )

        # From base.py, but spelled out here since we can avoid one sqrt when
        # computing the c/e ratios for the faces.
        alpha = (
            + self.ei_dot_ej[2] * self.ei_dot_ej[0]
            + self.ei_dot_ej[0] * self.ei_dot_ej[1]
            + self.ei_dot_ej[1] * self.ei_dot_ej[2]
            )
        # face_ce_ratios = -self.ei_dot_ej * 0.25 / face_areas[None]
        face_ce_ratios_div_face_areas = -self.ei_dot_ej / alpha

        # sum(self.circumcenter_face_distances * face_areas / 3) = cell_volumes
        # =>
        # cell_volumes = numpy.sqrt(sum(zeta / 72))
        self.cell_volumes = numpy.sqrt(numpy.sum(zeta, axis=0) / 72.0)

        #
        # self.circumcenter_face_distances =
        #    zeta / (24.0 * face_areas) / self.cell_volumes[None]
        # ce_ratios = \
        #     0.5 * face_ce_ratios * self.circumcenter_face_distances[None],
        #
        # so
        ce_ratios = \
            zeta/48.0 * face_ce_ratios_div_face_areas / self.cell_volumes[None]

        # Distances of the cell circumcenter to the faces.
        face_areas = 0.5 * numpy.sqrt(alpha)
        self.circumcenter_face_distances = \
            zeta / (24.0 * face_areas) / self.cell_volumes[None]

        return ce_ratios

    def get_cell_circumcenters(self):
        if self._circumcenters is None:
            self._compute_cell_circumcenters()
        return self._circumcenters

    def get_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        if self._control_volumes is None:
            #   1/3. * (0.5 * edge_length) * covolume
            # = 1/6 * edge_length**2 * ce_ratio_edge_ratio
            v = self.ei_dot_ei * self.get_ce_ratios() / 6.0
            # Explicitly sum up contributions per cell first. Makes
            # numpy.add.at faster.
            # For every node k (range(4)), check for which edges k appears in
            # local_idx, and sum() up the v's from there.
            idx = self.local_idx
            vals = numpy.array([
                sum([v[i, j] for i, j in zip(*numpy.where(idx == k)[1:])])
                for k in range(4)
                ]).T
            #
            self._control_volumes = numpy.zeros(len(self.node_coords))
            numpy.add.at(self._control_volumes, self.cells['nodes'], vals)
        return self._control_volumes

    def num_delaunay_violations(self):
        # Delaunay violations are present exactly on the interior faces where
        # the sum of the signed distances between face circumcenter and
        # tetrahedron circumcenter is negative.
        if self.circumcenter_face_distances is None:
            self._compute_ce_ratios_geometric()

        if 'faces' not in self.cells:
            self.create_cell_face_relationships()

        sums = numpy.zeros(len(self.faces['nodes']))
        numpy.add.at(
                sums,
                self.cells['faces'].T,
                self.circumcenter_face_distances
                )

        return numpy.sum(sums < 0.0)

    def show(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('equal')

        if self._circumcenters is None:
            self._compute_cell_circumcenters()

        X = self.node_coords
        for cell_id in range(len(self.cells['nodes'])):
            cc = self._circumcenters[cell_id]
            #
            x = X[self.node_face_cells[..., [cell_id]]]
            face_ccs = compute_triangle_circumcenters(
                    x, self.ei_dot_ei, self.ei_dot_ej
                    )
            # draw the face circumcenters
            ax.plot(face_ccs[..., 0], face_ccs[..., 1], face_ccs[..., 2], 'go')
            # draw the connections
            #   tet circumcenter---face circumcenter
            for face_cc in face_ccs:
                ax.plot(
                    [cc[..., 0], face_cc[..., 0]],
                    [cc[..., 1], face_cc[..., 1]],
                    [cc[..., 2], face_cc[..., 2]],
                    'b-'
                    )
        return

    def show_edge(self, edge_id):
        '''Displays edge with ce_ratio.

        :param edge_id: Edge ID for which to show the ce_ratio.
        :type edge_id: int
        '''
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt

        if 'faces' not in self.cells:
            self.create_cell_face_relationships()
        if 'edges' not in self.faces:
            self.create_face_edge_relationships()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('equal')

        # find all faces with this edge
        adj_face_ids = numpy.where(
            (self.faces['edges'] == edge_id).any(axis=1)
            )[0]
        # find all cells with the faces
        # <http://stackoverflow.com/a/38481969/353337>
        adj_cell_ids = numpy.where(numpy.in1d(
            self.cells['faces'], adj_face_ids
            ).reshape(self.cells['faces'].shape).any(axis=1)
            )[0]

        # plot all those adjacent cells; first collect all edges
        adj_edge_ids = numpy.unique([
            adj_edge_id
            for adj_cell_id in adj_cell_ids
            for face_id in self.cells['faces'][adj_cell_id]
            for adj_edge_id in self.faces['edges'][face_id]
            ])
        col = 'k'
        for adj_edge_id in adj_edge_ids:
            x = self.node_coords[self.edges['nodes'][adj_edge_id]]
            ax.plot(x[:, 0], x[:, 1], x[:, 2], col)

        # make clear which is edge_id
        x = self.node_coords[self.edges['nodes'][edge_id]]
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color=col, linewidth=3.0)

        # connect the face circumcenters with the corresponding cell
        # circumcenters
        X = self.node_coords
        for cell_id in adj_cell_ids:
            cc = self._circumcenters[cell_id]
            #
            x = X[self.node_face_cells[..., [cell_id]]]
            face_ccs = compute_triangle_circumcenters(
                    x, self.ei_dot_ei, self.ei_dot_ej
                    )
            # draw the face circumcenters
            ax.plot(face_ccs[..., 0], face_ccs[..., 1], face_ccs[..., 2], 'go')
            # draw the connections
            #   tet circumcenter---face circumcenter
            for face_cc in face_ccs:
                ax.plot(
                    [cc[..., 0], face_cc[..., 0]],
                    [cc[..., 1], face_cc[..., 1]],
                    [cc[..., 2], face_cc[..., 2]],
                    'b-'
                    )

        # draw the cell circumcenters
        cc = self._circumcenters[adj_cell_ids]
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], 'ro')
        return
