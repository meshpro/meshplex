# -*- coding: utf-8 -*-
#
import numpy
from voropy.base import \
        _base_mesh, \
        _row_dot, \
        compute_tri_areas_and_ce_ratios, \
        compute_triangle_circumcenters

__all__ = ['MeshTetra']


def _my_dot(a, b):
    return numpy.einsum('ijk, ijk->ij', a, b)


class MeshTetra(_base_mesh):
    '''Class for handling tetrahedral meshes.

    .. inheritance-diagram:: MeshTetra
    '''
    def __init__(self, node_coords, cells, mode='geometric'):
        '''Initialization.
        '''
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

        super(MeshTetra, self).__init__(node_coords, cells)

        self.cells = {
            'nodes': cells
            }

        self.create_cell_circumcenters_and_volumes()

        # adjacent entities
        self.cells['nodes'].sort(axis=1)
        self.create_cell_face_relationships()
        self.create_face_edge_relationships()

        self._mode = mode
        self._ce_ratios = None
        self._control_volumes = None

        self.mark_default_subdomains()

        # Arrange the cell_face_nodes such that node k is opposite of face k in
        # each cell.
        nds = self.cells['nodes']
        self.cell_face_nodes = numpy.stack([
            nds[:, [1, 2, 3]],
            nds[:, [2, 3, 0]],
            nds[:, [3, 0, 1]],
            nds[:, [0, 1, 2]],
            ], axis=1)
        # Arrange the cell_face_edge_nodes such that node k is opposite of edge
        # k in each face.
        self.cell_face_edge_nodes = numpy.stack([
            numpy.stack([
                nds[:, [2, 3]], nds[:, [3, 1]], nds[:, [1, 2]]
                ], axis=1),
            numpy.stack([
                nds[:, [3, 0]], nds[:, [0, 2]], nds[:, [2, 3]]
                ], axis=1),
            numpy.stack([
                nds[:, [0, 1]], nds[:, [1, 3]], nds[:, [3, 0]]
                ], axis=1),
            numpy.stack([
                nds[:, [1, 2]], nds[:, [2, 0]], nds[:, [0, 1]]
                ], axis=1),
            ], axis=1)

        return

    def get_ce_ratios(self):
        if self._ce_ratios is None:
            assert self._mode in ['geometric', 'algebraic']
            if self._mode == 'geometric':
                return self.compute_ce_ratios_geometric()
            else:  # 'algebraic'
                num_edges = len(self.edges['nodes'])
                self._ce_ratios = numpy.zeros(num_edges, dtype=float)
                raise RuntimeError('Disabled')
                idx, vals = self.compute_ce_ratios_algebraic()
                numpy.add.at(self._ce_ratios, idx, vals)
                self.circumcenter_face_distances = None
        return self._ce_ratios

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                # 'edges': range(len(self.edges['nodes'])),
                'faces': range(len(self.faces['nodes']))
                }

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
        self.cells['nodes'].sort(axis=1)

        # All possible faces.
        # Face k is opposite of node k in each cell.
        a = numpy.vstack([
            self.cells['nodes'][:, [1, 2, 3]],
            self.cells['nodes'][:, [0, 2, 3]],
            self.cells['nodes'][:, [0, 1, 3]],
            self.cells['nodes'][:, [0, 1, 2]]
            ])

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
        # TODO [1,2], [2,0], [0,1]
        a = numpy.vstack([
            self.faces['nodes'][:, [0, 1]],
            self.faces['nodes'][:, [0, 2]],
            self.faces['nodes'][:, [1, 2]]
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

    def create_cell_circumcenters_and_volumes(self):
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

        # Compute scalar triple product <a, b, c> = <b, c, a> = <c, a, b>.
        # The product is highly symmetric, so it's a little funny if there
        # should be no single einsum to compute it; see
        # <http://stackoverflow.com/q/42158228/353337>.
        omega = _row_dot(a, b_cross_c)

        self.cell_circumcenters = cell_coords[:, 0, :] + (
                b_cross_c * a_dot_a[:, None] +
                c_cross_a * b_dot_b[:, None] +
                a_cross_b * c_dot_c[:, None]
                ) / (2.0 * omega[:, None])

        # https://en.wikipedia.org/wiki/Tetrahedron#Volume
        self.cell_volumes = abs(omega) / 6.0
        return

#     def compute_ce_ratios_algebraic(self):
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

    def compute_ce_ratios_geometric(self):

        # prepare face edges
        e = self.node_coords[self.cell_face_edge_nodes[..., 1]] - \
            self.node_coords[self.cell_face_edge_nodes[..., 0]]

        e0 = e[:, :, 0, :]
        e1 = e[:, :, 1, :]
        e2 = e[:, :, 2, :]
        face_areas, face_ce_ratios = \
            compute_tri_areas_and_ce_ratios(e0, e1, e2)

        v0 = self.cell_face_nodes[:, :, 0]
        v1 = self.cell_face_nodes[:, :, 1]
        v2 = self.cell_face_nodes[:, :, 2]
        # opposing node
        v_op = self.cells['nodes']

        x0 = self.node_coords[v0] - self.node_coords[v_op]
        x1 = self.node_coords[v1] - self.node_coords[v_op]
        x2 = self.node_coords[v2] - self.node_coords[v_op]

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
        # TODO can those perhaps be expressed as dot products of x_ - x_, i.e.,
        #      edges of the considered face
        x0_dot_x0 = _my_dot(x0, x0)
        x1_dot_x1 = _my_dot(x1, x1)
        x2_dot_x2 = _my_dot(x2, x2)
        x0_dot_x1 = _my_dot(x0, x1)
        x1_dot_x2 = _my_dot(x1, x2)
        x2_dot_x0 = _my_dot(x2, x0)
        # # alpha = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x0_cross_x1>
        # alpha = \
        #     x0_dot_x0 * x1_dot_x1 - x0_dot_x1**2 + \
        #     x0_dot_x1 * x1_dot_x2 - x1_dot_x1 * x2_dot_x0 + \
        #     x2_dot_x0 * x0_dot_x1 - x1_dot_x2 * x0_dot_x0
        # # beta = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x1_cross_x2>
        # beta = \
        #     x0_dot_x1 * x1_dot_x2 - x2_dot_x0 * x1_dot_x1 + \
        #     x1_dot_x1 * x2_dot_x2 - x1_dot_x2**2 + \
        #     x1_dot_x2 * x2_dot_x0 - x2_dot_x2 * x0_dot_x1
        # # gamma = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x2_cross_x0>
        # gamma = \
        #     x2_dot_x0 * x0_dot_x1 - x0_dot_x0 * x1_dot_x2 + \
        #     x1_dot_x2 * x2_dot_x0 - x0_dot_x1 * x2_dot_x2 + \
        #     x0_dot_x0 * x2_dot_x2 - x2_dot_x0**2

        delta = \
            x0_dot_x0 * x1_dot_x1 * x2_dot_x2 - x2_dot_x2 * x0_dot_x1**2 + \
            x0_dot_x1 * x1_dot_x2 * x2_dot_x2 - x2_dot_x2 * x1_dot_x1 * x2_dot_x0 + \
            x2_dot_x0 * x0_dot_x1 * x2_dot_x2 - x2_dot_x2 * x1_dot_x2 * x0_dot_x0 + \
            x0_dot_x1 * x1_dot_x2 * x0_dot_x0 - x0_dot_x0 * x2_dot_x0 * x1_dot_x1 + \
            x1_dot_x1 * x2_dot_x2 * x0_dot_x0 - x0_dot_x0 * x1_dot_x2**2 + \
            x1_dot_x2 * x2_dot_x0 * x0_dot_x0 - x0_dot_x0 * x2_dot_x2 * x0_dot_x1 + \
            x2_dot_x0 * x0_dot_x1 * x1_dot_x1 - x1_dot_x1 * x0_dot_x0 * x1_dot_x2 + \
            x1_dot_x2 * x2_dot_x0 * x1_dot_x1 - x1_dot_x1 * x0_dot_x1 * x2_dot_x2 + \
            x0_dot_x0 * x2_dot_x2 * x1_dot_x1 - x1_dot_x1 * x2_dot_x0**2

        # delta2 = \
        #     _my_dot(x1 - x0, x2 - x1) * \
        #     _my_dot(x2 - x1, x0 - x2) * \
        #     _my_dot(x0 - x2, x1 - x0)
        #
        # print(delta - delta2)
        # exit(1)

        a = (
            72.0 * self.cell_volumes[:, None]**2
            - delta
            # - alpha * x2_dot_x2
            # - beta * x0_dot_x0
            # - gamma * x1_dot_x1
            ) / (12.0 * face_areas)

        # Distances of the cell circumcenter to the faces.
        # (shape: num_cells x 4)
        self.circumcenter_face_distances = \
            0.5 * a / self.cell_volumes[:, None]

        # Multiply
        s = 0.5 * face_ce_ratios * self.circumcenter_face_distances[..., None]

        return s

#     def compute_ce_ratios_geometric_back(self):
#
#         # prepare face edges
#         e = self.node_coords[self.edges['nodes'][self.faces['edges'], 1]] - \
#             self.node_coords[self.edges['nodes'][self.faces['edges'], 0]]
#         e0 = e[:, 0, :]
#         e1 = e[:, 1, :]
#         e2 = e[:, 2, :]
#         areas, face_ce_ratios = compute_tri_areas_and_ce_ratios(e0, e1, e2)
#         face_areas = areas[self.cells['faces']]
#         fce_ratios = face_ce_ratios[self.cells['faces']]
#
#         v0 = self.faces['nodes'][self.cells['faces']][:, :, 0]
#         v1 = self.faces['nodes'][self.cells['faces']][:, :, 1]
#         v2 = self.faces['nodes'][self.cells['faces']][:, :, 2]
#         v_op = self.cells['opposing vertex']
#
#         x0 = self.node_coords[v0] - self.node_coords[v_op]
#         x1 = self.node_coords[v1] - self.node_coords[v_op]
#         x2 = self.node_coords[v2] - self.node_coords[v_op]
#
#         # This is the reference expression.
#         # a = (
#         #     2 * _my_dot(x0_cross_x1, x2)**2 -
#         #     _my_dot(
#         #         x0_cross_x1 + x1_cross_x2 + x2_cross_x0,
#         #         x0_cross_x1 * x2_dot_x2[..., None] +
#         #         x1_cross_x2 * x0_dot_x0[..., None] +
#         #         x2_cross_x0 * x1_dot_x1[..., None]
#         #     )) / (12.0 * face_areas)
#
#         # Note that
#         #
#         #    6*tet_volume = abs(<x0 x x1, x2>)
#         #                 = abs(<x1 x x2, x0>)
#         #                 = abs(<x2 x x0, x1>).
#         #
#         # Also,
#         #
#         #    <a x b, c x d> = <a, c> <b, d> - <a, d> <b, c>.
#         #
#         # All those dot products can probably be cleaned up good.
#         # TODO simplify
#         # TODO can those perhaps be expressed as dot products of x_ - x_, i.e.,
#         #      edges of the considered face
#         x0_dot_x0 = _my_dot(x0, x0)
#         x1_dot_x1 = _my_dot(x1, x1)
#         x2_dot_x2 = _my_dot(x2, x2)
#         x0_dot_x1 = _my_dot(x0, x1)
#         x1_dot_x2 = _my_dot(x1, x2)
#         x2_dot_x0 = _my_dot(x2, x0)
#         # # alpha = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x0_cross_x1>
#         # alpha = \
#         #     x0_dot_x0 * x1_dot_x1 - x0_dot_x1**2 + \
#         #     x0_dot_x1 * x1_dot_x2 - x1_dot_x1 * x2_dot_x0 + \
#         #     x2_dot_x0 * x0_dot_x1 - x1_dot_x2 * x0_dot_x0
#         # # beta = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x1_cross_x2>
#         # beta = \
#         #     x0_dot_x1 * x1_dot_x2 - x2_dot_x0 * x1_dot_x1 + \
#         #     x1_dot_x1 * x2_dot_x2 - x1_dot_x2**2 + \
#         #     x1_dot_x2 * x2_dot_x0 - x2_dot_x2 * x0_dot_x1
#         # # gamma = <x0_cross_x1 + x1_cross_x2 + x2_cross_x0, x2_cross_x0>
#         # gamma = \
#         #     x2_dot_x0 * x0_dot_x1 - x0_dot_x0 * x1_dot_x2 + \
#         #     x1_dot_x2 * x2_dot_x0 - x0_dot_x1 * x2_dot_x2 + \
#         #     x0_dot_x0 * x2_dot_x2 - x2_dot_x0**2
#
#         delta = \
#             x0_dot_x0 * x1_dot_x1 * x2_dot_x2 - x2_dot_x2 * x0_dot_x1**2 + \
#             x0_dot_x1 * x1_dot_x2 * x2_dot_x2 - x2_dot_x2 * x1_dot_x1 * x2_dot_x0 + \
#             x2_dot_x0 * x0_dot_x1 * x2_dot_x2 - x2_dot_x2 * x1_dot_x2 * x0_dot_x0 + \
#             x0_dot_x1 * x1_dot_x2 * x0_dot_x0 - x0_dot_x0 * x2_dot_x0 * x1_dot_x1 + \
#             x1_dot_x1 * x2_dot_x2 * x0_dot_x0 - x0_dot_x0 * x1_dot_x2**2 + \
#             x1_dot_x2 * x2_dot_x0 * x0_dot_x0 - x0_dot_x0 * x2_dot_x2 * x0_dot_x1 + \
#             x2_dot_x0 * x0_dot_x1 * x1_dot_x1 - x1_dot_x1 * x0_dot_x0 * x1_dot_x2 + \
#             x1_dot_x2 * x2_dot_x0 * x1_dot_x1 - x1_dot_x1 * x0_dot_x1 * x2_dot_x2 + \
#             x0_dot_x0 * x2_dot_x2 * x1_dot_x1 - x1_dot_x1 * x2_dot_x0**2
#
#         # delta2 = \
#         #     _my_dot(x1 - x0, x2 - x1) * \
#         #     _my_dot(x2 - x1, x0 - x2) * \
#         #     _my_dot(x0 - x2, x1 - x0)
#         #
#         # print(delta - delta2)
#         #exit(1)
#
#         a = (
#             72.0 * self.cell_volumes[:, None]**2
#             - delta
#             # - alpha * x2_dot_x2
#             # - beta * x0_dot_x0
#             # - gamma * x1_dot_x1
#             ) / (12.0 * face_areas)
#
#         # Distances of the cell circumcenter to the faces.
#         # (shape: num_cells x 4)
#         self.circumcenter_face_distances = \
#             0.5 * a / self.cell_volumes[:, None]
#
#         # Multiply
#         s = 0.5 * fce_ratios * self.circumcenter_face_distances[..., None]
#
#         idx = self.faces['edges'][self.cells['faces']]
#         return idx, s

    def get_cell_circumcenters(self):
        return self.cell_circumcenters

    def get_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        if self._control_volumes is None:

            #   1/3. * (0.5 * edge_length) * covolume
            # = 1/6 * edge_length**2 * ce_ratio_edge_ratio
            ce = self.compute_ce_ratios_geometric()
            idx = self.cell_face_edge_nodes
            e = self.node_coords[idx[..., 1]] - \
                self.node_coords[idx[..., 0]]
            vals = _row_dot(e, e) * ce / 6.0
            vals = numpy.stack([vals, vals], axis=-1)
            # TODO explicitly sum up contributions per cell first
            #      (like mesh_tri)

            self._control_volumes = \
                numpy.zeros(len(self.node_coords), dtype=float)
            numpy.add.at(self._control_volumes, idx, vals)
        return self._control_volumes

    def num_delaunay_violations(self):
        # Delaunay violations are present exactly on the interior faces where
        # the sum of the signed distances between face circumcenter and
        # tetrahedron circumcenter is negative.
        if self.circumcenter_face_distances is None:
            self.compute_ce_ratios_geometric()

        sums = numpy.zeros(len(self.faces['nodes']))
        numpy.add.at(
                sums,
                self.cells['faces'],
                self.circumcenter_face_distances
                )

        return numpy.sum(sums < 0.0)

    def show(self):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.axis('equal')

        for edge_nodes in self.edges['nodes']:
            x = self.node_coords[edge_nodes]
            ax.plot(x[:, 0], x[:, 1], x[:, 2], 'k')
        return

    def show_edge(self, edge_id):
        '''Displays edge with ce_ratio.

        :param edge_id: Edge ID for which to show the ce_ratio.
        :type edge_id: int
        '''
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt

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
        for cell_id in adj_cell_ids:
            cc = self.cell_circumcenters[cell_id]
            for face_id in self.cells['faces'][cell_id]:
                if edge_id in self.faces['edges'][face_id]:
                    # draw the connection
                    #   tet circumcenter---face circumcenter
                    X = self.node_coords[self.faces['nodes'][[face_id]]]
                    fcc = compute_triangle_circumcenters(X)
                    ax.plot(
                        [cc[0], fcc[0, 0]],
                        [cc[1], fcc[0, 1]],
                        [cc[2], fcc[0, 2]],
                        'b-'
                        )
                    # draw the face circumcenter
                    ax.plot(fcc[:, 0], fcc[:, 1], fcc[:, 2], 'go')

        # draw the cell circumcenters
        cc = self.cell_circumcenters[adj_cell_ids]
        ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], 'ro')
        return
