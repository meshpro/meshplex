import os
import warnings

import numpy

from .base import _BaseMesh
from .helpers import (
    compute_ce_ratios,
    compute_tri_areas,
    compute_triangle_circumcenters,
    grp_start_len,
    unique_rows,
)

__all__ = ["MeshTri"]


class MeshTri(_BaseMesh):
    """Class for handling triangular meshes."""

    def __init__(self, points, cells, sort_cells=False):
        """Initialization."""
        if sort_cells:
            # Sort cells, first every row, then the rows themselves. This helps in many
            # downstream applications, e.g., when constructing linear systems with the
            # cells/edges. (When converting to CSR format, the I/J entries must be
            # sorted.) Don't use cells.sort(axis=1) to avoid
            # ```
            # ValueError: sort array is read-only
            # ```
            cells = numpy.sort(cells, axis=1)
            cells = cells[cells[:, 0].argsort()]

        points = numpy.asarray(points)
        cells = numpy.asarray(cells)
        assert len(points.shape) == 2, f"Illegal point coordinates shape {points.shape}"
        assert (
            len(cells.shape) == 2 and cells.shape[1] == 3
        ), f"Illegal cells shape {cells.shape}"

        self._points = numpy.asarray(points)
        # prevent accidental override of parts of the array
        self._points.setflags(write=False)
        super().__init__(points, cells)

        # Check which vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = numpy.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # points = points[uvertices]
        # ```
        # helps.
        self.point_is_used = numpy.zeros(len(points), dtype=bool)
        self.point_is_used[cells] = True

        # reset all data that changes when point coordinates change
        self._reset_point_data()

        self.cells = {"points": cells}

        self._cv_cell_mask = None
        self.edges = None
        self.subdomains = {}
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_edge = None
        self._is_boundary_edge_gid = None
        self._is_boundary_facet = None
        self._is_boundary_cell = None
        self._edges_cells = None
        self._edge_gid_to_edge_list = None
        self._edge_to_edge_gid = None

        # compute data
        # Create the idx_hierarchy (points->edges->cells), i.e., the value of
        # `self.idx_hierarchy[0, 2, 27]` is the index of the point of cell 27, edge 2,
        # point 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`, where `n` is the
        # number of cells. Make sure that the k-th edge is opposite of the k-th point in
        # the triangle.
        self.local_idx = numpy.array([[1, 2], [2, 0], [0, 1]]).T
        # Map idx back to the points. This is useful if quantities which are in idx
        # shape need to be added up into points (e.g., equation system rhs).
        nds = self.cells["points"].T
        self.idx_hierarchy = nds[self.local_idx]

        # The inverted local index.
        # This array specifies for each of the three points which edge endpoints
        # correspond to it. For the above local_idx, this should give
        #
        #    [[(1, 1), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (0, 1)]]
        #
        self.local_idx_inv = [
            [tuple(i) for i in zip(*numpy.where(self.local_idx == k))] for k in range(3)
        ]

    def __repr__(self):
        num_points = len(self.points)
        num_cells = len(self.cells["points"])
        string = f"<meshplex triangle mesh, {num_points} cells, {num_cells} points>"
        return string

    # prevent overriding points without adapting the other mesh data
    @property
    def points(self):
        return self._points

    def _reset_point_data(self):
        """Reset all data that changes when point coordinates changes."""
        self._half_edge_coords = None
        self._ei_dot_ei = None
        self._ei_dot_ej = None
        self._cell_volumes = None
        self._ce_ratios = None
        self._interior_edge_lengths = None
        self._cell_circumcenters = None
        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._surface_areas = None
        self._signed_cell_areas = None
        self._cell_centroids = None

    @points.setter
    def points(self, new_points):
        new_points = numpy.asarray(new_points)
        assert new_points.shape == self._points.shape
        self._points = new_points
        # reset all computed values
        self._reset_point_data()

    def set_points(self, new_points, idx=slice(None)):
        self.points.setflags(write=True)
        self.points[idx] = new_points
        self.points.setflags(write=False)
        self._reset_point_data()

    @property
    def euler_characteristic(self):
        # number of vertices - number of edges + number of faces
        if "edges" not in self.cells:
            self.create_edges()
        return (
            self.points.shape[0]
            - self.edges["points"].shape[0]
            + self.cells["points"].shape[0]
        )

    @property
    def genus(self):
        # https://math.stackexchange.com/a/85164/36678
        return 1 - self.euler_characteristic / 2

    @property
    def half_edge_coords(self):
        if self._half_edge_coords is None:
            self._half_edge_coords = (
                self.points[self.idx_hierarchy[1]] - self.points[self.idx_hierarchy[0]]
            )
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
            # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
            # TODO reorganize the data?
            self._ei_dot_ei = numpy.einsum(
                "ijk, ijk->ij", self.half_edge_coords, self.half_edge_coords
            )
        return self._ei_dot_ei

    @property
    def ei_dot_ej(self):
        if self._ei_dot_ej is None:
            # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
            # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
            # TODO reorganize the data?
            self._ei_dot_ej = numpy.einsum(
                "ijk, ijk->ij",
                self.half_edge_coords[[1, 2, 0]],
                self.half_edge_coords[[2, 0, 1]],
            )
        return self._ei_dot_ej

    @property
    def cell_volumes(self):
        if self._cell_volumes is None:
            self._cell_volumes = compute_tri_areas(self.ei_dot_ej)
        return self._cell_volumes

    @property
    def ce_ratios(self):
        if self._ce_ratios is None:
            self._ce_ratios = compute_ce_ratios(self.ei_dot_ej, self.cell_volumes)
        return self._ce_ratios

    def remove_cells(self, remove_array):
        """Remove cells and take care of all the dependent data structures. The input
        argument `remove_array` can be a boolean array or a list of indices.
        """
        remove_array = numpy.asarray(remove_array)
        if len(remove_array) == 0:
            return 0

        if remove_array.dtype == int:
            keep = numpy.ones(len(self.cells["points"]), dtype=bool)
            keep[remove_array] = False
        else:
            assert remove_array.dtype == bool
            keep = ~remove_array

        assert len(keep) == len(self.cells["points"]), "Wrong length of index array."

        if numpy.all(keep):
            return 0

        self.cells["points"] = self.cells["points"][keep]
        self.idx_hierarchy = self.idx_hierarchy[..., keep]

        if self._cell_volumes is not None:
            self._cell_volumes = self._cell_volumes[keep]

        if self._ce_ratios is not None:
            self._ce_ratios = self._ce_ratios[:, keep]

        if self._half_edge_coords is not None:
            self._half_edge_coords = self._half_edge_coords[:, keep]

        if self._ei_dot_ej is not None:
            self._ei_dot_ej = self._ei_dot_ej[:, keep]

        if self._ei_dot_ei is not None:
            self._ei_dot_ei = self._ei_dot_ei[:, keep]

        if self._cell_centroids is not None:
            self._cell_centroids = self._cell_centroids[keep]

        if self._cell_circumcenters is not None:
            self._cell_circumcenters = self._cell_circumcenters[keep]

        if self._cell_partitions is not None:
            self._cell_partitions = self._cell_partitions[keep]

        self._interior_edge_lengths = None
        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._surface_areas = None
        self._signed_cell_areas = None
        self._is_boundary_point = None

        # handle edges; this is a bit messy
        if "edges" in self.cells:
            num_edges_old = len(self.edges["points"])
            adjacent_edges, counts = numpy.unique(
                numpy.concatenate(self.cells["edges"][~keep]), return_counts=True
            )
            # remove edge entirely either if 2 adjacent cells are removed or if it
            # is a boundary edge and 1 adjacent cells is removed
            is_edge_removed = (counts == 2) | (
                (counts == 1) & self._is_boundary_edge_gid[adjacent_edges]
            )

            # set the new boundary edges
            self._is_boundary_edge_gid[adjacent_edges[~is_edge_removed]] = True
            # Now actually remove the edges. This includes a reindexing.
            remaining_edges = numpy.ones(len(self._is_boundary_edge_gid), dtype=bool)
            remaining_edges[adjacent_edges[is_edge_removed]] = False
            # make sure there is only edges["points"], not edges["cells"] etc.
            assert len(self.edges) == 1
            self.edges["points"] = self.edges["points"][remaining_edges]
            self._is_boundary_edge_gid = self._is_boundary_edge_gid[remaining_edges]

            self.cells["edges"] = self.cells["edges"][keep]
            new_index = numpy.arange(num_edges_old) - numpy.cumsum(~remaining_edges)
            self.cells["edges"] = new_index[self.cells["edges"]]

            # These could also be updated, but let's implement it when needed
            self._is_boundary_edge = None
            self._is_boundary_cell = None
            self._edges_cells = None
            self._edge_gid_to_edge_list = None
            self._edge_to_edge_gid = None

        return numpy.sum(~keep)

    @property
    def ce_ratios_per_interior_edge(self):
        if self._interior_ce_ratios is None:
            if "edges" not in self.cells:
                self.create_edges()

            n = self.edges["points"].shape[0]
            ce_ratios = numpy.bincount(
                self.cells["edges"].reshape(-1),
                self.ce_ratios.T.reshape(-1),
                minlength=n,
            )

            self._interior_ce_ratios = ce_ratios[~self._is_boundary_edge_gid]

            # # sum up from self.ce_ratios
            # if self._edges_cells is None:
            #     self._compute_edges_cells()

            # self._interior_ce_ratios = \
            #     numpy.zeros(self._edges_local[2].shape[0])
            # for i in [0, 1]:
            #     # Interior edges = edges with _2_ adjacent cells
            #     idx = [
            #         self._edges_local[2][:, i],
            #         self._edges_cells[2][:, i],
            #         ]
            #     self._interior_ce_ratios += self.ce_ratios[idx]

        return self._interior_ce_ratios

    def get_control_volumes(self, cell_mask=None):
        """The control volumes around each vertex. Optionally disregard the
        contributions from particular cells. This is useful, for example, for
        temporarily disregarding flat cells on the boundary when performing Lloyd mesh
        optimization.
        """
        if cell_mask is None:
            cell_mask = numpy.zeros(self.cell_partitions.shape[1], dtype=bool)

        if self._control_volumes is None or numpy.any(cell_mask != self._cv_cell_mask):
            # Summing up the arrays first makes the work on bincount a bit lighter.
            v = self.cell_partitions[:, ~cell_mask]
            vals = numpy.array([v[1] + v[2], v[2] + v[0], v[0] + v[1]])
            # sum all the vals into self._control_volumes at ids
            self.cells["points"][~cell_mask].T.reshape(-1)
            self._control_volumes = numpy.bincount(
                self.cells["points"][~cell_mask].T.reshape(-1),
                weights=vals.reshape(-1),
                minlength=len(self.points),
            )
            self._cv_cell_mask = cell_mask
        return self._control_volumes

    @property
    def control_volumes(self):
        """The control volumes around each vertex."""
        return self.get_control_volumes()

    # @property
    # def surface_areas(self):
    #     """"""
    #     if self._surface_areas is None:
    #         self._surface_areas = self._compute_surface_areas()
    #     return self._surface_areas

    def get_control_volume_centroids(self, cell_mask=None):
        """
        The centroid of any volume V is given by

        .. math::
          c = \\int_V x / \\int_V 1.

        The denominator is the control volume. The numerator can be computed by making
        use of the fact that the control volume around any vertex is composed of right
        triangles, two for each adjacent cell.

        Optionally disregard the contributions from particular cells. This is useful,
        for example, for temporarily disregarding flat cells on the boundary when
        performing Lloyd mesh optimization.
        """
        if cell_mask is None:
            cell_mask = numpy.zeros(self.cell_partitions.shape[1], dtype=bool)

        if self._cv_centroids is None or numpy.any(cell_mask != self._cvc_cell_mask):
            _, v = self._compute_integral_x()
            v = v[:, :, ~cell_mask, :]

            # Again, make use of the fact that edge k is opposite of point k in every
            # cell. Adding the arrays first makes the work for bincount lighter.
            ids = self.cells["points"][~cell_mask].T
            vals = numpy.array(
                [v[1, 1] + v[0, 2], v[1, 2] + v[0, 0], v[1, 0] + v[0, 1]]
            )
            # add it all up
            n = len(self.points)
            self._cv_centroids = numpy.array(
                [
                    numpy.bincount(
                        ids.reshape(-1), vals[..., k].reshape(-1), minlength=n
                    )
                    for k in range(vals.shape[-1])
                ]
            ).T

            # Divide by the control volume
            cv = self.get_control_volumes(cell_mask=cell_mask)[:, None]
            # self._cv_centroids /= numpy.where(cv > 0.0, cv, 1.0)
            self._cv_centroids /= cv
            self._cvc_cell_mask = cell_mask
            assert numpy.all(cell_mask == self._cv_cell_mask)

        return self._cv_centroids

    @property
    def control_volume_centroids(self):
        return self.get_control_volume_centroids()

    @property
    def signed_cell_areas(self):
        """Signed area of a triangle in 2D."""
        if self._signed_cell_areas is None:
            self._signed_cell_areas = self.compute_signed_cell_areas()
        return self._signed_cell_areas

    def compute_signed_cell_areas(self, idx=slice(None)):
        # http://mathworld.wolfram.com/TriangleArea.html
        assert (
            self.points.shape[1] == 2
        ), "Signed areas only make sense for triangles in 2D."

        # One could make p contiguous by adding a copy(), but that's not
        # really worth it here.
        p = self.points[self.cells["points"][idx]].T
        # <https://stackoverflow.com/q/50411583/353337>
        return (
            +p[0][2] * (p[1][0] - p[1][1])
            + p[0][0] * (p[1][1] - p[1][2])
            + p[0][1] * (p[1][2] - p[1][0])
        ) / 2

    def mark_boundary(self):
        if self.edges is None:
            self.create_edges()

        self._is_boundary_point = numpy.zeros(len(self.points), dtype=bool)
        self._is_boundary_point[self.idx_hierarchy[..., self.is_boundary_edge]] = True

        self._is_interior_point = self.point_is_used & ~self.is_boundary_point

        self._is_boundary_facet = self.is_boundary_edge

    @property
    def is_boundary_cell(self):
        if self._is_boundary_cell is None:
            self._is_boundary_cell = numpy.any(self.is_boundary_edge, axis=0)
        return self._is_boundary_cell

    @property
    def is_boundary_edge(self):
        if self._is_boundary_edge is None:
            self.create_edges()
        return self._is_boundary_edge

    @property
    def is_boundary_edge_gid(self):
        if self._is_boundary_edge_gid is None:
            self.create_edges()
        return self._is_boundary_edge_gid

    @property
    def is_boundary_point(self):
        if self._is_boundary_point is None:
            self.mark_boundary()
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self.mark_boundary()
        return self._is_interior_point

    @property
    def is_boundary_facet(self):
        if self._is_boundary_facet is None:
            self.mark_boundary()
        return self._is_boundary_facet

    def create_edges(self):
        """Set up edge->point and edge->cell relations."""
        # Reshape into individual edges.
        # Sort the columns to make it possible for `unique()` to identify
        # individual edges.
        s = self.idx_hierarchy.shape
        a = numpy.sort(self.idx_hierarchy.reshape(s[0], -1).T)
        a_unique, inv, cts = unique_rows(a)

        assert numpy.all(
            cts < 3
        ), "No edge has more than 2 cells. Are cells listed twice?"

        self._is_boundary_edge = (cts[inv] == 1).reshape(s[1:])
        self._is_boundary_edge_gid = cts == 1

        self.edges = {"points": a_unique}

        # cell->edges relationship
        self.cells["edges"] = inv.reshape(3, -1).T

        self._edges_cells = None
        self._edge_gid_to_edge_list = None

        # Store an index {boundary,interior}_edge -> edge_gid
        self._edge_to_edge_gid = [
            [],
            numpy.where(self._is_boundary_edge_gid)[0],
            numpy.where(~self._is_boundary_edge_gid)[0],
        ]

    @property
    def edges_cells(self):
        if self._edges_cells is None:
            self._compute_edges_cells()
        return self._edges_cells

    def _compute_edges_cells(self):
        """This creates interior edge->cells relations. While it's not
        necessary for many applications, it sometimes does come in handy.
        """
        if self.edges is None:
            self.create_edges()

        num_edges = len(self.edges["points"])

        # count = numpy.bincount(self.cells["edges"].reshape(-1), minlength=num_edges)

        # <https://stackoverflow.com/a/50395231/353337>
        edges_flat = self.cells["edges"].flat
        idx_sort = numpy.argsort(edges_flat)
        idx_start, count = grp_start_len(edges_flat[idx_sort])
        res1 = idx_sort[idx_start[count == 1]][:, numpy.newaxis]
        idx = idx_start[count == 2]
        res2 = numpy.column_stack([idx_sort[idx], idx_sort[idx + 1]])
        self._edges_cells = [
            [],  # there are no edges with zero adjacent cells
            res1 // 3,
            res2 // 3,
        ]
        # self._edges_local = [
        #     [],  # no edges with zero adjacent cells
        #     res1 % 3,
        #     res2 % 3,
        #     ]

        # For each edge, store the number of adjacent cells plus the index into
        # the respective edge array.
        self._edge_gid_to_edge_list = numpy.empty((num_edges, 2), dtype=int)
        self._edge_gid_to_edge_list[:, 0] = count
        c1 = count == 1
        l1 = numpy.sum(c1)
        self._edge_gid_to_edge_list[c1, 1] = numpy.arange(l1)
        c2 = count == 2
        l2 = numpy.sum(c2)
        self._edge_gid_to_edge_list[c2, 1] = numpy.arange(l2)
        assert l1 + l2 == len(count)
        return

    @property
    def edge_gid_to_edge_list(self):
        """"""
        if self._edge_gid_to_edge_list is None:
            self._compute_edges_cells()
        return self._edge_gid_to_edge_list

    @property
    def face_partitions(self):
        # face = edge for triangles.
        # The partition is simply along the center of the edge.
        edge_lengths = self.edge_lengths
        return numpy.array([0.5 * edge_lengths, 0.5 * edge_lengths])

    @property
    def cell_partitions(self):
        if self._cell_partitions is None:
            # Compute the control volumes. Note that
            #
            #   0.5 * (0.5 * edge_length) * covolume
            # = 0.25 * edge_length**2 * ce_ratio_edge_ratio
            #
            self._cell_partitions = 0.25 * self.ei_dot_ei * self.ce_ratios
        return self._cell_partitions

    @property
    def cell_circumcenters(self):
        if self._cell_circumcenters is None:
            point_cells = self.cells["points"].T
            self._cell_circumcenters = compute_triangle_circumcenters(
                self.points[point_cells], self.ei_dot_ei, self.ei_dot_ej
            )
        return self._cell_circumcenters

    def compute_centroids(self, idx=slice(None)):
        return numpy.sum(self.points[self.cells["points"][idx]], axis=1) / 3.0

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all triangles."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_centroids()
        return self._cell_centroids

    @property
    def cell_barycenters(self):
        """See cell_centroids."""
        return self.cell_centroids

    @property
    def cell_incenters(self):
        """Get the midpoints of the incircles."""
        # https://en.wikipedia.org/wiki/Incenter#Barycentric_coordinates
        abc = numpy.sqrt(self.ei_dot_ei)
        abc /= numpy.sum(abc, axis=0)
        return numpy.einsum("ij,jik->jk", abc, self.points[self.cells["points"]])

    @property
    def cell_inradius(self):
        """Get the inradii of all cells"""
        # See <http://mathworld.wolfram.com/Incircle.html>.
        abc = numpy.sqrt(self.ei_dot_ei)
        return 2 * self.cell_volumes / numpy.sum(abc, axis=0)

    @property
    def cell_circumradius(self):
        """Get the circumradii of all cells"""
        # See <http://mathworld.wolfram.com/Circumradius.html>.
        a, b, c = numpy.sqrt(self.ei_dot_ei)
        return (a * b * c) / numpy.sqrt(
            (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
        )

    @property
    def cell_quality(self):
        warnings.warn(
            "Use `q_radius_ratio`. This method will be removed in a future release."
        )
        return self.q_radius_ratio

    @property
    def q_radius_ratio(self):
        """2 * inradius / circumradius (min 0, max 1)"""
        # q = 2 * r_in / r_out
        #   = (-a+b+c) * (+a-b+c) * (+a+b-c) / (a*b*c),
        #
        # where r_in is the incircle radius and r_out the circumcircle radius
        # and a, b, c are the edge lengths.
        a, b, c = numpy.sqrt(self.ei_dot_ei)
        return (-a + b + c) * (a - b + c) * (a + b - c) / (a * b * c)

    @property
    def angles(self):
        """All angles in the triangle."""
        # The cosines of the angles are the negative dot products of the normalized
        # edges adjacent to the angle.
        norms = numpy.sqrt(self.ei_dot_ei)
        normalized_ei_dot_ej = numpy.array(
            [
                self.ei_dot_ej[0] / norms[1] / norms[2],
                self.ei_dot_ej[1] / norms[2] / norms[0],
                self.ei_dot_ej[2] / norms[0] / norms[1],
            ]
        )
        return numpy.arccos(-normalized_ei_dot_ej)

    def _compute_integral_x(self):
        # Computes the integral of x,
        #
        #   \\int_V x,
        #
        # over all atomic "triangles", i.e., areas cornered by a point, an edge midpoint,
        # and a circumcenter.

        # The integral of any linear function over a triangle is the average of the
        # values of the function in each of the three corners, times the area of the
        # triangle.
        right_triangle_vols = self.cell_partitions

        point_edges = self.idx_hierarchy

        corner = self.points[point_edges]
        edge_midpoints = 0.5 * (corner[0] + corner[1])
        cc = self.cell_circumcenters

        average = (corner + edge_midpoints[None] + cc[None, None]) / 3.0

        contribs = right_triangle_vols[None, :, :, None] * average

        return point_edges, contribs

    # def _compute_surface_areas(self, cell_ids):
    #     # For each edge, one half of the the edge goes to each of the end points. Used
    #     # for Neumann boundary conditions if on the boundary of the mesh and transition
    #     # conditions if in the interior.
    #     #
    #     # Each of the three edges may contribute to the surface areas of all three
    #     # vertices. Here, only the two adjacent points receive a contribution, but other
    #     # approaches, may contribute to all three points.
    #     cn = self.cells["points"][cell_ids]
    #     ids = numpy.stack([cn, cn, cn], axis=1)

    #     half_el = 0.5 * self.edge_lengths[..., cell_ids]
    #     zero = numpy.zeros([half_el.shape[1]])
    #     vals = numpy.stack(
    #         [
    #             numpy.column_stack([zero, half_el[0], half_el[0]]),
    #             numpy.column_stack([half_el[1], zero, half_el[1]]),
    #             numpy.column_stack([half_el[2], half_el[2], zero]),
    #         ],
    #         axis=1,
    #     )

    #     return ids, vals

    #     def compute_gradient(self, u):
    #         '''Computes an approximation to the gradient :math:`\\nabla u` of a
    #         given scalar valued function :math:`u`, defined in the point points.
    #         This is taken from :cite:`NME2187`,
    #
    #            Discrete gradient method in solid mechanics,
    #            Lu, Jia and Qian, Jing and Han, Weimin,
    #            International Journal for Numerical Methods in Engineering,
    #            https://doi.org/10.1002/nme.2187.
    #         '''
    #         if self.cell_circumcenters is None:
    #             X = self.points[self.cells['points']]
    #             self.cell_circumcenters = self.compute_triangle_circumcenters(X)
    #
    #         if 'cells' not in self.edges:
    #             self.edges['cells'] = self.compute_edge_cells()
    #
    #         # This only works for flat meshes.
    #         assert (abs(self.points[:, 2]) < 1.0e-10).all()
    #         points2d = self.points[:, :2]
    #         cell_circumcenters2d = self.cell_circumcenters[:, :2]
    #
    #         num_points = len(points2d)
    #         assert len(u) == num_points
    #
    #         gradient = numpy.zeros((num_points, 2), dtype=u.dtype)
    #
    #         # Create an empty 2x2 matrix for the boundary points to hold the
    #         # edge correction ((17) in [1]).
    #         boundary_matrices = {}
    #         for point in self.get_vertices('boundary'):
    #             boundary_matrices[point] = numpy.zeros((2, 2))
    #
    #         for edge_gid, edge in enumerate(self.edges['cells']):
    #             # Compute edge length.
    #             point0 = self.edges['points'][edge_gid][0]
    #             point1 = self.edges['points'][edge_gid][1]
    #
    #             # Compute coedge length.
    #             if len(self.edges['cells'][edge_gid]) == 1:
    #                 # Boundary edge.
    #                 edge_midpoint = 0.5 * (
    #                         points2d[point0] +
    #                         points2d[point1]
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
    #                 self.control_volumes[self.edges['points'][edge_gid]]
    #
    #             # Compute R*_{IJ} ((11) in [1]).
    #             r0 = (coedge_midpoint - points2d[point0]) * coeffs[0]
    #             r1 = (coedge_midpoint - points2d[point1]) * coeffs[1]
    #
    #             diff = u[point1] - u[point0]
    #
    #             gradient[point0] += r0 * diff
    #             gradient[point1] -= r1 * diff
    #
    #             # Store the boundary correction matrices.
    #             edge_coords = points2d[point1] - points2d[point0]
    #             if point0 in boundary_matrices:
    #                 boundary_matrices[point0] += numpy.outer(r0, edge_coords)
    #             if point1 in boundary_matrices:
    #                 boundary_matrices[point1] += numpy.outer(r1, -edge_coords)
    #
    #         # Apply corrections to the gradients on the boundary.
    #         for k, value in boundary_matrices.items():
    #             gradient[k] = numpy.linalg.solve(value, gradient[k])
    #
    #         return gradient

    def compute_curl(self, vector_field):
        """Computes the curl of a vector field over the mesh. While the vector field is
        point-based, the curl will be cell-based. The approximation is based on

        .. math::
            n\\cdot curl(F) = \\lim_{A\\to 0} |A|^{-1} <\\int_{dGamma}, F> dr;

        see https://en.wikipedia.org/wiki/Curl_(mathematics). Actually, to approximate
        the integral, one would only need the projection of the vector field onto the
        edges at the midpoint of the edges.
        """
        # Compute the projection of A on the edge at each edge midpoint. Take the
        # average of `vector_field` at the endpoints to get the approximate value at the
        # edge midpoint.
        A = 0.5 * numpy.sum(vector_field[self.idx_hierarchy], axis=0)
        # sum of <edge, A> for all three edges
        sum_edge_dot_A = numpy.einsum("ijk, ijk->j", self.half_edge_coords, A)

        # Get normalized vector orthogonal to triangle
        z = numpy.cross(self.half_edge_coords[0], self.half_edge_coords[1])

        # Now compute
        #
        #    curl = z / ||z|| * sum_edge_dot_A / |A|.
        #
        # Since ||z|| = 2*|A|, one can save a sqrt and do
        #
        #    curl = z * sum_edge_dot_A * 0.5 / |A|^2.
        #
        curl = z * (0.5 * sum_edge_dot_A / self.cell_volumes ** 2)[..., None]
        return curl

    def num_delaunay_violations(self):
        """Number of edges where the Delaunay condition is violated."""
        # Delaunay violations are present exactly on the interior edges where the
        # ce_ratio is negative. Count those.
        return numpy.sum(self.ce_ratios_per_interior_edge < 0.0)

    def show(self, *args, fullscreen=False, **kwargs):
        """Show the mesh (see plot())."""
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        if fullscreen:
            mng = plt.get_current_fig_manager()
            # mng.frame.Maximize(True)
            mng.window.showMaximized()
        plt.show()
        plt.close()

    def save(self, filename, *args, **kwargs):
        """Save the mesh to a file."""
        _, file_extension = os.path.splitext(filename)
        if file_extension in [".png", ".svg"]:
            import matplotlib.pyplot as plt

            self.plot(*args, **kwargs)
            plt.savefig(filename, transparent=True, bbox_inches="tight")
            plt.close()
        else:
            self.write(filename)
        return

    def plot(
        self,
        show_coedges=True,
        control_volume_centroid_color=None,
        mesh_color="k",
        nondelaunay_edge_color=None,
        boundary_edge_color=None,
        comesh_color=(0.8, 0.8, 0.8),
        show_axes=True,
        cell_quality_coloring=None,
        show_point_numbers=False,
        show_cell_numbers=False,
        cell_mask=None,
        show_edge_numbers=False,
        mark_cells=None,
    ):
        """Show the mesh using matplotlib."""
        # Importing matplotlib takes a while, so don't do that at the header.
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.patches import Polygon

        fig = plt.figure()
        ax = fig.gca()
        plt.axis("equal")
        if not show_axes:
            ax.set_axis_off()

        xmin = numpy.amin(self.points[:, 0])
        xmax = numpy.amax(self.points[:, 0])
        ymin = numpy.amin(self.points[:, 1])
        ymax = numpy.amax(self.points[:, 1])

        width = xmax - xmin
        xmin -= 0.1 * width
        xmax += 0.1 * width

        height = ymax - ymin
        ymin -= 0.1 * height
        ymax += 0.1 * height

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # for k, x in enumerate(self.points):
        #     if self.is_boundary_point[k]:
        #         plt.plot(x[0], x[1], "g.")
        #     else:
        #         plt.plot(x[0], x[1], "r.")

        if show_point_numbers:
            for i, x in enumerate(self.points):
                plt.text(
                    x[0],
                    x[1],
                    str(i),
                    bbox=dict(facecolor="w", alpha=0.7),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if show_cell_numbers:
            for i, x in enumerate(self.cell_centroids):
                plt.text(
                    x[0],
                    x[1],
                    str(i),
                    bbox=dict(facecolor="r", alpha=0.5),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        # coloring
        if cell_quality_coloring:
            cmap, cmin, cmax, show_colorbar = cell_quality_coloring
            plt.tripcolor(
                self.points[:, 0],
                self.points[:, 1],
                self.cells["points"],
                self.q_radius_ratio,
                shading="flat",
                cmap=cmap,
                vmin=cmin,
                vmax=cmax,
            )
            if show_colorbar:
                plt.colorbar()

        if mark_cells is not None:
            patches = []
            for idx in mark_cells:
                patches.append(Polygon(self.points[self.cells["points"][idx]]))
            p = PatchCollection(patches, facecolor="C1")
            ax.add_collection(p)

        if self.edges is None:
            self.create_edges()

        # Get edges, cut off z-component.
        e = self.points[self.edges["points"]][:, :, :2]

        if nondelaunay_edge_color is None:
            line_segments0 = LineCollection(e, color=mesh_color)
            ax.add_collection(line_segments0)
        else:
            # Plot regular edges, mark those with negative ce-ratio red.
            ce_ratios = self.ce_ratios_per_interior_edge
            pos = ce_ratios >= 0

            is_pos = numpy.zeros(len(self.edges["points"]), dtype=bool)
            is_pos[self._edge_to_edge_gid[2][pos]] = True

            # Mark Delaunay-conforming boundary edges
            is_pos_boundary = self.ce_ratios[self.is_boundary_edge] >= 0
            is_pos[self._edge_to_edge_gid[1][is_pos_boundary]] = True

            line_segments0 = LineCollection(e[is_pos], color=mesh_color)
            ax.add_collection(line_segments0)
            #
            line_segments1 = LineCollection(e[~is_pos], color=nondelaunay_edge_color)
            ax.add_collection(line_segments1)

        if show_coedges:
            # Connect all cell circumcenters with the edge midpoints
            cc = self.cell_circumcenters

            edge_midpoints = 0.5 * (
                self.points[self.edges["points"][:, 0]]
                + self.points[self.edges["points"][:, 1]]
            )

            # Plot connection of the circumcenter to the midpoint of all three
            # axes.
            a = numpy.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 0], :2]], axis=1
            )
            b = numpy.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 1], :2]], axis=1
            )
            c = numpy.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 2], :2]], axis=1
            )

            line_segments = LineCollection(
                numpy.concatenate([a, b, c]), color=comesh_color
            )
            ax.add_collection(line_segments)

        if boundary_edge_color:
            e = self.points[self.edges["points"][self.is_boundary_edge_gid]][:, :, :2]
            line_segments1 = LineCollection(e, color=boundary_edge_color)
            ax.add_collection(line_segments1)

        if control_volume_centroid_color is not None:
            centroids = self.get_control_volume_centroids(cell_mask=cell_mask)
            ax.plot(
                centroids[:, 0],
                centroids[:, 1],
                linestyle="",
                marker=".",
                color=control_volume_centroid_color,
            )
            for k, centroid in enumerate(centroids):
                plt.text(
                    centroid[0],
                    centroid[1],
                    str(k),
                    bbox=dict(facecolor=control_volume_centroid_color, alpha=0.7),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        return fig

    def show_vertex(self, *args, **kwargs):
        """Show the mesh around a vertex (see plot_vertex())."""
        import matplotlib.pyplot as plt

        self.plot_vertex(*args, **kwargs)
        plt.show()
        plt.close()
        return

    def plot_vertex(self, point_id, show_ce_ratio=True):
        """Plot the vicinity of a point and its covolume/edgelength ratio.

        :param point_id: Node ID of the point to be shown.
        :type point_id: int

        :param show_ce_ratio: If true, shows the ce_ratio of the point, too.
        :type show_ce_ratio: bool, optional
        """
        # Importing matplotlib takes a while, so don't do that at the header.
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        plt.axis("equal")

        if self.edges is None:
            self.create_edges()

        # Find the edges that contain the vertex
        edge_gids = numpy.where((self.edges["points"] == point_id).any(axis=1))[0]
        # ... and plot them
        for point_ids in self.edges["points"][edge_gids]:
            x = self.points[point_ids]
            ax.plot(x[:, 0], x[:, 1], "k")

        # Highlight ce_ratios.
        if show_ce_ratio:
            if self.cell_circumcenters is None:
                X = self.points[self.cells["points"]]
                self.cell_circumcenters = self.compute_triangle_circumcenters(
                    X, self.ei_dot_ei, self.ei_dot_ej
                )

            # Find the cells that contain the vertex
            cell_ids = numpy.where((self.cells["points"] == point_id).any(axis=1))[0]

            for cell_id in cell_ids:
                for edge_gid in self.cells["edges"][cell_id]:
                    if point_id not in self.edges["points"][edge_gid]:
                        continue
                    point_ids = self.edges["points"][edge_gid]
                    edge_midpoint = 0.5 * (
                        self.points[point_ids[0]] + self.points[point_ids[1]]
                    )
                    p = numpy.stack(
                        [self.cell_circumcenters[cell_id], edge_midpoint], axis=1
                    )
                    q = numpy.column_stack(
                        [
                            self.cell_circumcenters[cell_id],
                            edge_midpoint,
                            self.points[point_id],
                        ]
                    )
                    ax.fill(q[0], q[1], color="0.5")
                    ax.plot(p[0], p[1], color="0.7")
        return

    def flip_until_delaunay(self):
        """Flip edges until the mesh is fully Delaunay."""
        num_flips = 0
        # If all coedge/edge ratios are positive, all cells are Delaunay.
        if numpy.all(self.ce_ratios > 0):
            return num_flips

        # If all _interior_ coedge/edge ratios are positive, all cells are Delaunay.
        if self.is_boundary_edge is None:
            self.mark_boundary()
        if numpy.all(self.ce_ratios[~self.is_boundary_edge] > 0):
            return num_flips

        if self._edges_cells is None:
            self._compute_edges_cells()

        ce_ratios_per_interior_edge = self.ce_ratios_per_interior_edge
        while numpy.any(ce_ratios_per_interior_edge < 0.0):
            is_flip_interior_edge = ce_ratios_per_interior_edge < 0.0

            interior_edges_cells = self._edges_cells[2]
            adj_cells = interior_edges_cells[is_flip_interior_edge].T

            # Check if there are cells for which more than one edge needs to be flipped.
            # For those, only flip one edge, namely that with the smaller (more
            # negative) ce_ratio.
            cell_gids, num_flips_per_cell = numpy.unique(adj_cells, return_counts=True)
            critical_cell_gids = cell_gids[num_flips_per_cell > 1]
            while numpy.any(num_flips_per_cell > 1):
                for cell_gid in critical_cell_gids:
                    edge_gids = self.cells["edges"][cell_gid]
                    num_adj_cells, edge_id = self._edge_gid_to_edge_list[edge_gids].T
                    edge_ids = edge_id[num_adj_cells == 2]
                    k = numpy.argmin(ce_ratios_per_interior_edge[edge_ids])
                    is_flip_interior_edge[edge_ids] = False
                    is_flip_interior_edge[edge_ids[k]] = True

                adj_cells = interior_edges_cells[is_flip_interior_edge].T
                cell_gids, num_flips_per_cell = numpy.unique(
                    adj_cells, return_counts=True
                )
                critical_cell_gids = cell_gids[num_flips_per_cell > 1]

            self.flip_interior_edges(is_flip_interior_edge)
            num_flips += numpy.sum(is_flip_interior_edge)
            ce_ratios_per_interior_edge = self.ce_ratios_per_interior_edge

        return num_flips

    def flip_interior_edges(self, is_flip_interior_edge):
        interior_edges_cells = self.edges_cells[2]
        # adj_cells = interior_edges_cells[is_flip_interior_edge].T

        edge_gids = self._edge_to_edge_gid[2][is_flip_interior_edge]
        adj_cells = interior_edges_cells[is_flip_interior_edge].T

        # Get the local ids of the edge in the two adjacent cells.
        # Get all edges of the adjacent cells
        ec = self.cells["edges"][adj_cells]
        # Find where the edge sits.
        hits = ec == edge_gids[None, :, None]
        # Make sure that there is exactly one match per cell
        assert numpy.all(numpy.sum(hits, axis=2) == 1)
        # translate to lids
        idx = numpy.empty(hits.shape, dtype=int)
        idx[..., 0] = 0
        idx[..., 1] = 1
        idx[..., 2] = 2
        lids = idx[hits].reshape((2, -1))

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
        verts = numpy.array(
            [
                self.cells["points"][adj_cells[0], lids[0]],
                self.cells["points"][adj_cells[1], lids[1]],
                self.cells["points"][adj_cells[0], (lids[0] + 1) % 3],
                self.cells["points"][adj_cells[0], (lids[0] + 2) % 3],
            ]
        )

        # Do the neighboring cells have equal orientation (both point sets
        # clockwise/counterclockwise)?
        equal_orientation = (
            self.cells["points"][adj_cells[0], (lids[0] + 1) % 3]
            == self.cells["points"][adj_cells[1], (lids[1] + 2) % 3]
        )

        # Set new cells.
        # Make sure that positive/negative area orientation is preserved.
        self.cells["points"][adj_cells[0]] = verts[[0, 2, 1]].T
        self.cells["points"][adj_cells[1]] = verts[[0, 1, 3]].T

        # Reset flipped edges
        self.edges["points"][edge_gids] = numpy.sort(verts[[0, 1]].T, axis=1)
        # No need to touch self.is_boundary_edge or self.is_boundary_edge_gid;
        # we're only flipping interior edges.

        # Set up new cells->edges relationships.
        previous_edges = self.cells["edges"][adj_cells].copy()

        i0 = numpy.ones(equal_orientation.shape[0], dtype=int)
        i0[~equal_orientation] = 2
        i1 = numpy.ones(equal_orientation.shape[0], dtype=int)
        i1[equal_orientation] = 2

        self.cells["edges"][adj_cells[0]] = numpy.column_stack(
            [
                numpy.choose((lids[1] + i0) % 3, previous_edges[1].T),
                edge_gids,
                numpy.choose((lids[0] + 2) % 3, previous_edges[0].T),
            ]
        )
        self.cells["edges"][adj_cells[1]] = numpy.column_stack(
            [
                numpy.choose((lids[1] + i1) % 3, previous_edges[1].T),
                numpy.choose((lids[0] + 1) % 3, previous_edges[0].T),
                edge_gids,
            ]
        )

        # update is_boundary_edge
        for k in range(3):
            self.is_boundary_edge[k, adj_cells] = self.is_boundary_edge_gid[
                self.cells["edges"][adj_cells, k]
            ]

        # Update the edge->cells relationship. It doesn't change for the edge that was
        # flipped, but for two of the other edges.
        confs = [
            (0, 1, numpy.choose((lids[0] + 1) % 3, previous_edges[0].T)),
            (1, 0, numpy.choose((lids[1] + i0) % 3, previous_edges[1].T)),
        ]
        for conf in confs:
            c, d, edge_gids = conf
            num_adj_cells, edge_id = self._edge_gid_to_edge_list[edge_gids].T

            k1 = num_adj_cells == 1
            k2 = num_adj_cells == 2
            assert numpy.all(numpy.logical_xor(k1, k2))

            # outer boundary edges
            edge_id1 = edge_id[k1]
            assert numpy.all(self.edges_cells[1][edge_id1][:, 0] == adj_cells[c, k1])
            self.edges_cells[1][edge_id1, 0] = adj_cells[d, k1]

            # interior edges
            edge_id2 = edge_id[k2]
            is_column0 = self.edges_cells[2][edge_id2][:, 0] == adj_cells[c, k2]
            is_column1 = self.edges_cells[2][edge_id2][:, 1] == adj_cells[c, k2]
            assert numpy.all(numpy.logical_xor(is_column0, is_column1))
            #
            self.edges_cells[2][edge_id2[is_column0], 0] = adj_cells[d, k2][is_column0]
            self.edges_cells[2][edge_id2[is_column1], 1] = adj_cells[d, k2][is_column1]

        # Schedule the cell ids for data updates
        update_cell_ids = numpy.unique(adj_cells.T.flat)
        # Same for edge ids
        k, edge_gids = self._edge_gid_to_edge_list[
            self.cells["edges"][update_cell_ids].flat
        ].T
        update_interior_edge_ids = numpy.unique(edge_gids[k == 2])

        self._update_cell_values(update_cell_ids, update_interior_edge_ids)

    def remove_boundary_cells(self, criterion):
        """Helper method for removing cells along the boundary.
        The input criterion is a boolean array of length `sum(mesh.is_boundary_cell)`.

        This helps, for example, in the following scenario.
        When points are moving around, flip_until_delaunay() makes sure the mesh remains
        a Delaunay mesh. This does not work on boundaries where very flat cells can
        still occur or cells may even 'invert'. (The interior point moves outside.) In
        this case, the boundary cell can be removed, and the newly outward node is made
        a boundary node."""
        num_removed = 0
        while True:
            crit = criterion(self.is_boundary_cell)
            if numpy.all(~crit):
                break
            idx = self.is_boundary_cell.copy()
            idx[idx] = crit
            n = self.remove_cells(idx)
            num_removed += n
            if n == 0:
                break
        return num_removed

    def _update_cell_values(self, cell_ids, interior_edge_ids):
        """Updates all sorts of cell information for the given cell IDs."""
        # update idx_hierarchy
        nds = self.cells["points"][cell_ids].T
        self.idx_hierarchy[..., cell_ids] = nds[self.local_idx]

        # update self.half_edge_coords
        self.half_edge_coords[:, cell_ids, :] = numpy.moveaxis(
            self.points[self.idx_hierarchy[1, ..., cell_ids]]
            - self.points[self.idx_hierarchy[0, ..., cell_ids]],
            0,
            1,
        )

        # update self.ei_dot_ej
        self.ei_dot_ej[:, cell_ids] = numpy.einsum(
            "ijk, ijk->ij",
            self.half_edge_coords[[1, 2, 0]][:, cell_ids],
            self.half_edge_coords[[2, 0, 1]][:, cell_ids],
        )

        # update self.ei_dot_ei
        e = self.half_edge_coords[:, cell_ids]
        self.ei_dot_ei[:, cell_ids] = numpy.einsum("ijk, ijk->ij", e, e)

        # update cell_volumes, ce_ratios_per_half_edge
        cv = compute_tri_areas(self.ei_dot_ej[:, cell_ids])
        ce = compute_ce_ratios(self.ei_dot_ej[:, cell_ids], cv)
        self.cell_volumes[cell_ids] = cv
        self._ce_ratios[:, cell_ids] = ce

        if self._interior_ce_ratios is not None:
            self._interior_ce_ratios[interior_edge_ids] = 0.0
            edge_gids = self._edge_to_edge_gid[2][interior_edge_ids]
            adj_cells = self.edges_cells[2][interior_edge_ids]

            is_edge = numpy.array(
                [
                    self.cells["edges"][adj_cells[:, 0]][:, k] == edge_gids
                    for k in range(3)
                ]
            )
            assert numpy.all(numpy.sum(is_edge, axis=0) == 1)
            for k in range(3):
                self._interior_ce_ratios[
                    interior_edge_ids[is_edge[k]]
                ] += self.ce_ratios[k, adj_cells[is_edge[k], 0]]

            is_edge = numpy.array(
                [
                    self.cells["edges"][adj_cells[:, 1]][:, k] == edge_gids
                    for k in range(3)
                ]
            )
            assert numpy.all(numpy.sum(is_edge, axis=0) == 1)
            for k in range(3):
                self._interior_ce_ratios[
                    interior_edge_ids[is_edge[k]]
                ] += self.ce_ratios[k, adj_cells[is_edge[k], 1]]

        # TODO update those values
        self._cell_centroids = None
        self._edge_lengths = None
        self._cell_circumcenters = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._surface_areas = None
        self._signed_cell_areas = None
        self.subdomains = {}
