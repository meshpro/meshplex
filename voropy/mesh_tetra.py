# -*- coding: utf-8 -*-
#
import numpy
from voropy.base import \
        _base_mesh, \
        compute_triangle_circumcenters

__all__ = ['MeshTetra']


# pylint: disable=too-many-instance-attributes
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
        # Don't use cells.sort(axis=1) to avoid
        # ```
        # ValueError: sort array is read-only
        # ```
        cells = numpy.sort(cells, axis=1)
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
        is_used[cells] = True
        assert all(is_used)

        self.cells = {
            'nodes': cells
            }

        self._mode = mode
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
        #   * all nodes are in domino order ([1, 2], [2, 3], [3, 1]),
        #   * the same edges are easy to find:
        #      - edge 0: face+1, edge 2
        #      - edge 1: face+2, edge 1
        #      - edge 2: face+3, edge 0
        #
        self.local_idx = numpy.array([
            [[2, 3], [3, 1], [1, 2]],
            [[3, 0], [0, 2], [2, 3]],
            [[0, 1], [1, 3], [3, 0]],
            [[1, 2], [2, 0], [0, 1]],
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

        self.ce_ratios = self._compute_ce_ratios_geometric()

        self.is_boundary_node = None
        self._inv_faces = None
        self.edges = None
        self.is_boundary_face_individual = None
        self.is_boundary_face = None
        self.faces = None
        return

    def get_ce_ratios(self):
        return self.ce_ratios

    def mark_boundary(self):
        if 'faces' not in self.cells:
            self.create_cell_face_relationships()

        self.is_boundary_node = numpy.zeros(len(self.node_coords), dtype=bool)
        self.is_boundary_node[
            self.faces['nodes'][self.is_boundary_face_individual]
            ] = True
        return

    def create_cell_face_relationships(self):
        # Reshape into individual faces, and take the first node per edge. (The
        # face is fully characterized by it.) Sort the columns to make it
        # possible for `unique()` to identify individual faces.
        s = self.idx_hierarchy.shape
        a = self.idx_hierarchy.reshape([s[0], s[1], s[2]*s[3]]).T
        a = numpy.sort(a[:, :, 0])

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

        # No face has more than 2 cells. This assertion fails, for example, if
        # cells are listed twice.
        assert all(cts < 3)

        self.is_boundary_face = (cts[inv] == 1).reshape(s[2:])
        self.is_boundary_face_individual = (cts == 1)

        self.faces = {
            'nodes': a[idx]
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
        # Just like for triangular cells, tetrahedron circumcenters are most
        # easily computed with the quadrilateral coordinates available.
        # Luckily, we have the circumcenter-face distances (cfd):
        #
        #   CC = (
        #       + cfd[0] * face_area[0] / sum(cfd*face_area) * X[0]
        #       + cfd[1] * face_area[1] / sum(cfd*face_area) * X[1]
        #       + cfd[2] * face_area[2] / sum(cfd*face_area) * X[2]
        #       + cfd[3] * face_area[3] / sum(cfd*face_area) * X[3]
        #       )
        #
        # (Compare with
        # <https://en.wikipedia.org/wiki/Trilinear_coordinates#Between_Cartesian_and_trilinear_coordinates>.)
        # Because of
        #
        #    cfd = zeta / (24.0 * face_areas) / self.cell_volumes[None]
        #
        # we have
        #
        #   CC = sum_k (zeta[k] / sum(zeta) * X[k]).
        #
        alpha = self._zeta / numpy.sum(self._zeta, axis=0)

        self._circumcenters = numpy.sum(
            alpha[None].T * self.node_coords[self.cells['nodes']],
            axis=1
            )
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
        # For triangles, the covolume/edgelength ratios are
        #
        #   [1]   ce_ratios = -<ei, ej> / cell_volume / 4;
        #
        # for tetrahedra, is somewhat more tricky. This is the reference
        # expression:
        #
        # ce_ratios = (
        #     2 * _my_dot(x0_cross_x1, x2)**2 -
        #     _my_dot(
        #         x0_cross_x1 + x1_cross_x2 + x2_cross_x0,
        #         x0_cross_x1 * x2_dot_x2[..., None] +
        #         x1_cross_x2 * x0_dot_x0[..., None] +
        #         x2_cross_x0 * x1_dot_x1[..., None]
        #     )) / (12.0 * face_areas)
        #
        # Tedious simplification steps (with the help of
        # <https://github.com/nschloe/brute_simplify>) lead to
        #
        #   zeta = (
        #       + ei_dot_ej[0, 2] * ei_dot_ej[3, 5] * ei_dot_ej[5, 4]
        #       + ei_dot_ej[0, 1] * ei_dot_ej[3, 5] * ei_dot_ej[3, 4]
        #       + ei_dot_ej[1, 2] * ei_dot_ej[3, 4] * ei_dot_ej[4, 5]
        #       + self.ei_dot_ej[0] * self.ei_dot_ej[1] * self.ei_dot_ej[2]
        #       ).
        #
        # for the face [1, 2, 3] (with edges [3, 4, 5]), where nodes and edges
        # are ordered like
        #
        #                        3
        #                        ^
        #                       /|\
        #                      / | \
        #                     /  \  \
        #                    /    \  \
        #                   /      |  \
        #                  /       |   \
        #                 /        \    \
        #                /         4\    \
        #               /            |    \
        #              /2            |     \5
        #             /              \      \
        #            /                \      \
        #           /            _____|       \
        #          /        ____/     2\_      \
        #         /    ____/1            \_     \
        #        /____/                    \_    \
        #       /________                   3\_   \
        #      0         \__________           \___\
        #                        0  \______________\\
        #                                            1
        #
        # This is not a too obvious extension of -<ei, ej> in [1]. However,
        # consider the fact that this contains all pairwise dot-products of
        # edges not part of the respective face (<e0, e1>, <e1, e2>, <e2, e0>),
        # each of them weighted with dot-products of edges in the respective
        # face.
        #
        # Note that, to retrieve the covolume-edge ratio, one divides by
        #
        #       alpha = (
        #           + ei_dot_ej[3, 5] * ei_dot_ej[5, 4]
        #           + ei_dot_ej[3, 5] * ei_dot_ej[3, 4]
        #           + ei_dot_ej[3, 4] * ei_dot_ej[4, 5]
        #           )
        #
        # (which is the square of the face area). It's funny that there should
        # be no further simplification in zeta/alpha, but nothing has been
        # found here yet.
        #
        ee = self.ei_dot_ej
        self._zeta = (
            - ee[2, [1, 2, 3, 0]] * ee[1] * ee[2]
            - ee[1, [2, 3, 0, 1]] * ee[2] * ee[0]
            - ee[0, [3, 0, 1, 2]] * ee[0] * ee[1]
            + ee[0] * ee[1] * ee[2]
            )

        # From base.py, but spelled out here since we can avoid one sqrt when
        # computing the c/e ratios for the faces.
        alpha = (
            + self.ei_dot_ej[2] * self.ei_dot_ej[0]
            + self.ei_dot_ej[0] * self.ei_dot_ej[1]
            + self.ei_dot_ej[1] * self.ei_dot_ej[2]
            )
        # pylint: disable=invalid-unary-operand-type
        # face_ce_ratios = -self.ei_dot_ej * 0.25 / face_areas[None]
        face_ce_ratios_div_face_areas = -self.ei_dot_ej / alpha

        # TODO Check out the Cayley-Menger determinant
        # <http://mathworld.wolfram.com/Cayley-MengerDeterminant.html
        #
        # sum(self.circumcenter_face_distances * face_areas / 3) = cell_volumes
        # =>
        # cell_volumes = numpy.sqrt(sum(zeta / 72))
        self.cell_volumes = numpy.sqrt(numpy.sum(self._zeta, axis=0) / 72.0)

        #
        # self.circumcenter_face_distances =
        #    zeta / (24.0 * face_areas) / self.cell_volumes[None]
        # ce_ratios = \
        #     0.5 * face_ce_ratios * self.circumcenter_face_distances[None],
        #
        # so
        ce_ratios = \
            self._zeta/48.0 \
            * face_ce_ratios_div_face_areas / self.cell_volumes[None]

        # Distances of the cell circumcenter to the faces.
        face_areas = 0.5 * numpy.sqrt(alpha)
        self.circumcenter_face_distances = \
            self._zeta / (24.0 * face_areas) / self.cell_volumes[None]

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
            v = self.ei_dot_ei * self.ce_ratios / 6.0
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
        # pylint: disable=unused-variable,relative-import
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
        # pylint: disable=unused-variable,relative-import
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
