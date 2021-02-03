import os
import warnings

import numpy as np

from .base import _SimplexMesh
from .helpers import (
    compute_ce_ratios,
    compute_tri_areas,
    compute_triangle_circumcenters,
    grp_start_len,
    unique_rows,
)

__all__ = ["MeshTri"]


class MeshTri(_SimplexMesh):
    """Class for handling triangular meshes."""

    def __init__(self, points, cells, sort_cells=False):
        """Initialization."""
        super().__init__(points, cells, sort_cells=sort_cells)

        # reset all data that changes when point coordinates change
        self._reset_point_data()

        self._cv_cell_mask = None
        self.edges = None
        self.subdomains = {}
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_edge_local = None
        self._is_boundary_edge = None
        self._is_boundary_cell = None
        self._edges_cells = None
        self._edges_cells_idx = None
        self._boundary_edges = None
        self._interior_edges = None
        self._is_point_used = None

    def __repr__(self):
        num_points = len(self.points)
        num_cells = len(self.cells["points"])
        string = f"<meshplex triangle mesh, {num_points} points, {num_cells} cells>"
        return string

    def _reset_point_data(self):
        """Reset all data that changes when point coordinates changes."""
        self._half_edge_coords = None
        self._edge_lengths = None
        self._ei_dot_ei = None
        self._ei_dot_ej = None
        self._cell_volumes = None
        self._ce_ratios = None
        self._cell_circumcenters = None
        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._signed_cell_areas = None
        self._cell_centroids = None

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
    def cell_volumes(self):
        if self._cell_volumes is None:
            self._cell_volumes = compute_tri_areas(self.ei_dot_ej)
        return self._cell_volumes

    @property
    def ce_ratios(self):
        if self._ce_ratios is None:
            self._ce_ratios = compute_ce_ratios(self.ei_dot_ej, self.cell_volumes)
        return self._ce_ratios

    def remove_dangling_points(self):
        """Remove all points which aren't part of an array
        """
        is_part_of_cell = np.zeros(self.points.shape[0], dtype=bool)
        is_part_of_cell[self.cells["points"].flat] = True

        new_point_idx = np.cumsum(is_part_of_cell) - 1

        self.cells["points"] = new_point_idx[self.cells["points"]]
        self._points = self._points[is_part_of_cell]

    def remove_cells(self, remove_array):
        """Remove cells and take care of all the dependent data structures. The input
        argument `remove_array` can be a boolean array or a list of indices.
        """
        # Although this method doesn't compute anything new, the reorganization of the
        # data structure is fairly expensive. This is mostly due to the fact that mask
        # copies like `a[mask]` take long if `a` is large, even if `mask` is True almost
        # everywhere.
        # Keep an eye on <https://stackoverflow.com/q/65035280/353337> for possible
        # workarounds.
        remove_array = np.asarray(remove_array)
        if len(remove_array) == 0:
            return 0

        if remove_array.dtype == int:
            keep = np.ones(len(self.cells["points"]), dtype=bool)
            keep[remove_array] = False
        else:
            assert remove_array.dtype == bool
            keep = ~remove_array

        assert len(keep) == len(self.cells["points"]), "Wrong length of index array."

        if np.all(keep):
            return 0

        # handle edges; this is a bit messy
        if "edges" in self.cells:
            # updating the boundary data is a lot easier with edges_cells
            if self._edges_cells is None:
                self._compute_edges_cells()

            # Set edge to is_boundary_edge_local=True if it is adjacent to a removed
            # cell.
            edge_ids = self.cells["edges"][~keep].flatten()
            # only consider interior edges
            edge_ids = edge_ids[self.is_interior_edge[edge_ids]]
            idx = self.edges_cells_idx[edge_ids]
            cell_id = self.edges_cells["interior"][1:3, idx].T
            local_edge_id = self.edges_cells["interior"][3:5, idx].T
            self._is_boundary_edge_local[local_edge_id, cell_id] = True
            # now remove the entries corresponding to the removed cells
            self._is_boundary_edge_local = self._is_boundary_edge_local[:, keep]

            if self._is_boundary_cell is not None:
                self._is_boundary_cell[cell_id] = True
                self._is_boundary_cell = self._is_boundary_cell[keep]

            # update edges_cells
            keep_b_ec = keep[self.edges_cells["boundary"][1]]
            keep_i_ec0, keep_i_ec1 = keep[self.edges_cells["interior"][1:3]]
            # move ec from interior to boundary if exactly one of the two adjacent cells
            # was removed

            keep_i_0 = keep_i_ec0 & ~keep_i_ec1
            keep_i_1 = keep_i_ec1 & ~keep_i_ec0
            self._edges_cells["boundary"] = np.array(
                [
                    # edge id
                    np.concatenate(
                        [
                            self._edges_cells["boundary"][0, keep_b_ec],
                            self._edges_cells["interior"][0, keep_i_0],
                            self._edges_cells["interior"][0, keep_i_1],
                        ]
                    ),
                    # cell id
                    np.concatenate(
                        [
                            self._edges_cells["boundary"][1, keep_b_ec],
                            self._edges_cells["interior"][1, keep_i_0],
                            self._edges_cells["interior"][2, keep_i_1],
                        ]
                    ),
                    # local edge id
                    np.concatenate(
                        [
                            self._edges_cells["boundary"][2, keep_b_ec],
                            self._edges_cells["interior"][3, keep_i_0],
                            self._edges_cells["interior"][4, keep_i_1],
                        ]
                    ),
                ]
            )

            keep_i = keep_i_ec0 & keep_i_ec1

            # this memory copy isn't too fast
            self._edges_cells["interior"] = self._edges_cells["interior"][:, keep_i]

            num_edges_old = len(self.edges["points"])
            adjacent_edges, counts = np.unique(
                self.cells["edges"][~keep].flat, return_counts=True
            )
            # remove edge entirely either if 2 adjacent cells are removed or if it is a
            # boundary edge and 1 adjacent cells are removed
            is_edge_removed = (counts == 2) | (
                (counts == 1) & self._is_boundary_edge[adjacent_edges]
            )

            # set the new boundary edges
            self._is_boundary_edge[adjacent_edges[~is_edge_removed]] = True
            # Now actually remove the edges. This includes a reindexing.
            assert self._is_boundary_edge is not None
            keep_edges = np.ones(len(self._is_boundary_edge), dtype=bool)
            keep_edges[adjacent_edges[is_edge_removed]] = False

            # make sure there is only edges["points"], not edges["cells"] etc.
            assert self.edges is not None
            assert len(self.edges) == 1
            self.edges["points"] = self.edges["points"][keep_edges]
            self._is_boundary_edge = self._is_boundary_edge[keep_edges]

            # update edge and cell indices
            self.cells["edges"] = self.cells["edges"][keep]
            new_index_edges = np.arange(num_edges_old) - np.cumsum(~keep_edges)
            self.cells["edges"] = new_index_edges[self.cells["edges"]]
            num_cells_old = len(self.cells["points"])
            new_index_cells = np.arange(num_cells_old) - np.cumsum(~keep)

            # this takes fairly long
            ec = self._edges_cells
            ec["boundary"][0] = new_index_edges[ec["boundary"][0]]
            ec["boundary"][1] = new_index_cells[ec["boundary"][1]]
            ec["interior"][0] = new_index_edges[ec["interior"][0]]
            ec["interior"][1:3] = new_index_cells[ec["interior"][1:3]]

            # simply set those to None; their reset is cheap
            self._edges_cells_idx = None
            self._boundary_edges = None
            self._interior_edges = None

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
            self._cell_partitions = self._cell_partitions[:, keep]

        if self._signed_cell_areas is not None:
            self._signed_cell_areas = self._signed_cell_areas[keep]

        # TODO These could also be updated, but let's implement it when needed
        self._interior_ce_ratios = None
        self._control_volumes = None
        self._cv_cell_mask = None
        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._is_point_used = None
        self._is_interior_point = None
        self._is_boundary_point = None

        return np.sum(~keep)

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
            if np.all(~crit):
                break
            idx = self.is_boundary_cell.copy()
            idx[idx] = crit
            n = self.remove_cells(idx)
            num_removed += n
            if n == 0:
                break
        return num_removed

    @property
    def ce_ratios_per_interior_edge(self):
        if self._interior_ce_ratios is None:
            if "edges" not in self.cells:
                self.create_edges()

            n = self.edges["points"].shape[0]
            ce_ratios = np.bincount(
                self.cells["edges"].reshape(-1),
                self.ce_ratios.T.reshape(-1),
                minlength=n,
            )

            self._interior_ce_ratios = ce_ratios[~self._is_boundary_edge]

            # # sum up from self.ce_ratios
            # if self._edges_cells is None:
            #     self._compute_edges_cells()

            # self._interior_ce_ratios = \
            #     np.zeros(self._edges_local[2].shape[0])
            # for i in [0, 1]:
            #     # Interior edges = edges with _2_ adjacent cells
            #     idx = [
            #         self._edges_local[2][:, i],
            #         self._edges_cells["interior"][:, i],
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
            cell_mask = np.zeros(self.cell_partitions.shape[1], dtype=bool)

        if self._control_volumes is None or np.any(cell_mask != self._cv_cell_mask):
            # Summing up the arrays first makes the work on bincount a bit lighter.
            v = self.cell_partitions[:, ~cell_mask]
            vals = np.array([v[1] + v[2], v[2] + v[0], v[0] + v[1]])
            # sum all the vals into self._control_volumes at ids
            self.cells["points"][~cell_mask].T.reshape(-1)
            self._control_volumes = np.bincount(
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
            cell_mask = np.zeros(self.cell_partitions.shape[1], dtype=bool)

        if self._cv_centroids is None or np.any(cell_mask != self._cvc_cell_mask):
            _, v = self._compute_integral_x()
            v = v[:, :, ~cell_mask, :]

            # Again, make use of the fact that edge k is opposite of point k in every
            # cell. Adding the arrays first makes the work for bincount lighter.
            ids = self.cells["points"][~cell_mask].T
            vals = np.array([v[1, 1] + v[0, 2], v[1, 2] + v[0, 0], v[1, 0] + v[0, 1]])
            # add it all up
            n = len(self.points)
            self._cv_centroids = np.array(
                [
                    np.bincount(ids.reshape(-1), vals[..., k].reshape(-1), minlength=n)
                    for k in range(vals.shape[-1])
                ]
            ).T

            # Divide by the control volume
            cv = self.get_control_volumes(cell_mask=cell_mask)[:, None]
            # self._cv_centroids /= np.where(cv > 0.0, cv, 1.0)
            self._cv_centroids /= cv
            self._cvc_cell_mask = cell_mask
            assert np.all(cell_mask == self._cv_cell_mask)

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
        assert (
            self.points.shape[1] == 2
        ), "Signed areas only make sense for triangles in 2D."
        # On <https://stackoverflow.com/q/50411583/353337>, we have a number of
        # alternatives computing the oriented area, but it's fastest with the
        # half-edges.
        x = self.half_edge_coords
        return (x[0, idx, 1] * x[2, idx, 0] - x[0, idx, 0] * x[2, idx, 1]) / 2

    def mark_boundary(self):
        warnings.warn(
            "mark_boundary() does nothing. "
            "Boundary entities are computed on the fly."
        )

    @property
    def is_boundary_cell(self):
        if self._is_boundary_cell is None:
            assert self.is_boundary_edge_local is not None
            self._is_boundary_cell = np.any(self.is_boundary_edge_local, axis=0)
        return self._is_boundary_cell

    @property
    def is_boundary_edge_local(self):
        if self._is_boundary_edge_local is None:
            self.create_edges()
        return self._is_boundary_edge_local

    is_boundary_facet_local = is_boundary_edge_local

    @property
    def is_boundary_edge(self):
        if self._is_boundary_edge is None:
            self.create_edges()
        return self._is_boundary_edge

    @property
    def is_interior_edge(self):
        return ~self._is_boundary_edge

    @property
    def boundary_edges(self):
        if self._boundary_edges is None:
            self._boundary_edges = np.where(self.is_boundary_edge)[0]
        return self._boundary_edges

    @property
    def interior_edges(self):
        if self._interior_edges is None:
            self._interior_edges = np.where(~self.is_boundary_edge)[0]
        return self._interior_edges

    @property
    def is_boundary_point(self):
        if self._is_boundary_point is None:
            self._is_boundary_point = np.zeros(len(self.points), dtype=bool)
            self._is_boundary_point[
                self.idx_hierarchy[..., self.is_boundary_edge_local]
            ] = True
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self._is_interior_point = self.is_point_used & ~self.is_boundary_point
        return self._is_interior_point

    def create_edges(self):
        """Set up edge->point and edge->cell relations."""
        # Reshape into individual edges.
        # Sort the columns to make it possible for `unique()` to identify
        # individual edges.
        s = self.idx_hierarchy.shape
        a = np.sort(self.idx_hierarchy.reshape(s[0], -1).T)
        a_unique, inv, cts = unique_rows(a)

        assert np.all(cts < 3), "No edge has more than 2 cells. Are cells listed twice?"

        self._is_boundary_edge_local = (cts[inv] == 1).reshape(s[1:])
        self._is_boundary_edge = cts == 1

        self.edges = {"points": a_unique}

        # cell->edges relationship
        self.cells["edges"] = inv.reshape(3, -1).T

        self._edges_cells = None
        self._edges_cells_idx = None

    @property
    def edges_cells(self):
        if self._edges_cells is None:
            self._compute_edges_cells()
        return self._edges_cells

    def _compute_edges_cells(self):
        """This creates edge->cells relations. While it's not necessary for many
        applications, it sometimes does come in handy, for example for mesh
        manipulation.
        """
        if self.edges is None:
            self.create_edges()

        # num_edges = len(self.edges["points"])
        # count = np.bincount(self.cells["edges"].flat, minlength=num_edges)

        # <https://stackoverflow.com/a/50395231/353337>
        edges_flat = self.cells["edges"].flat
        idx_sort = np.argsort(edges_flat)
        sorted_edges = edges_flat[idx_sort]
        idx_start, count = grp_start_len(sorted_edges)

        # count is redundant with is_boundary/interior_edge
        assert np.all((count == 1) == self.is_boundary_edge)
        assert np.all((count == 2) == self.is_interior_edge)

        idx_start_count_1 = idx_start[self.is_boundary_edge]
        idx_start_count_2 = idx_start[self.is_interior_edge]
        res1 = idx_sort[idx_start_count_1]
        res2 = idx_sort[np.array([idx_start_count_2, idx_start_count_2 + 1])]

        edge_id_boundary = sorted_edges[idx_start_count_1]
        edge_id_interior = sorted_edges[idx_start_count_2]

        # It'd be nicer if we could organize the data differently, e.g., as a structured
        # array or as a dict. Those possibilities are slower, unfortunately, for some
        # operations in remove_cells() (and perhaps elsewhere).
        # <https://github.com/numpy/numpy/issues/17850>
        self._edges_cells = {
            # rows:
            #  0: edge id
            #  1: cell id
            #  2: local edge id (0, 1, or 2)
            "boundary": np.array([edge_id_boundary, res1 // 3, res1 % 3]),
            # rows:
            #  0: edge id
            #  1: cell id 0
            #  2: cell id 1
            #  3: local edge id 0 (0, 1, or 2)
            #  4: local edge id 1 (0, 1, or 2)
            "interior": np.array([edge_id_interior, *(res2 // 3), *(res2 % 3)]),
        }

        self._edges_cells_idx = None

    @property
    def edges_cells_idx(self):
        if self._edges_cells_idx is None:
            if self._edges_cells is None:
                self._compute_edges_cells()
            assert self.is_boundary_edge is not None
            # For each edge, store the index into the respective edge array.
            num_edges = len(self.edges["points"])
            self._edges_cells_idx = np.empty(num_edges, dtype=int)
            num_b = np.sum(self.is_boundary_edge)
            num_i = np.sum(self.is_interior_edge)
            self._edges_cells_idx[self.edges_cells["boundary"][0]] = np.arange(num_b)
            self._edges_cells_idx[self.edges_cells["interior"][0]] = np.arange(num_i)
        return self._edges_cells_idx

    @property
    def cell_partitions(self):
        if self._cell_partitions is None:
            # Compute the control volume contributions. Note that
            #
            #   0.5 * (0.5 * edge_length) * covolume
            # = 0.25 * edge_length ** 2 * ce_ratio_edge_ratio
            #
            self._cell_partitions = self.ei_dot_ei * self.ce_ratios / 4
        return self._cell_partitions

    @property
    def cell_circumcenters(self):
        if self._cell_circumcenters is None:
            point_cells = self.cells["points"].T
            self._cell_circumcenters = compute_triangle_circumcenters(
                self.points[point_cells], self.cell_partitions
            )
        return self._cell_circumcenters

    @property
    def cell_incenters(self):
        """Get the midpoints of the incircles."""
        # https://en.wikipedia.org/wiki/Incenter#Barycentric_coordinates
        abc = self.edge_lengths / np.sum(self.edge_lengths, axis=0)
        return np.einsum("ij,jik->jk", abc, self.points[self.cells["points"]])

    @property
    def cell_inradius(self):
        """Get the inradii of all cells"""
        # See <http://mathworld.wolfram.com/Incircle.html>.
        return 2 * self.cell_volumes / np.sum(self.edge_lengths, axis=0)

    @property
    def cell_circumradius(self):
        """Get the circumradii of all cells"""
        # See <http://mathworld.wolfram.com/Circumradius.html> and
        # <https://en.wikipedia.org/wiki/Cayley%E2%80%93Menger_determinant#Finding_the_circumradius_of_a_simplex>.
        return np.prod(self.edge_lengths, axis=0) / 4 / self.cell_volumes

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
        a, b, c = self.edge_lengths
        return (-a + b + c) * (a - b + c) * (a + b - c) / (a * b * c)

    @property
    def angles(self):
        """All angles in the triangle."""
        # The cosines of the angles are the negative dot products of the normalized
        # edges adjacent to the angle.
        norms = self.edge_lengths
        normalized_ei_dot_ej = np.array(
            [
                self.ei_dot_ej[0] / norms[1] / norms[2],
                self.ei_dot_ej[1] / norms[2] / norms[0],
                self.ei_dot_ej[2] / norms[0] / norms[1],
            ]
        )
        return np.arccos(-normalized_ei_dot_ej)

    def _compute_integral_x(self):
        # Computes the integral of x,
        #
        #   \\int_V x,
        #
        # over all atomic "triangles", i.e., areas cornered by a point, an edge
        # midpoint, and a circumcenter.

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
    #     ids = np.stack([cn, cn, cn], axis=1)

    #     half_el = 0.5 * self.edge_lengths[..., cell_ids]
    #     zero = np.zeros([half_el.shape[1]])
    #     vals = np.stack(
    #         [
    #             np.column_stack([zero, half_el[0], half_el[0]]),
    #             np.column_stack([half_el[1], zero, half_el[1]]),
    #             np.column_stack([half_el[2], half_el[2], zero]),
    #         ],
    #         axis=1,
    #     )

    #     return ids, vals

    #     def compute_gradient(self, u):
    #         '''Computes an approximation to the gradient :math:`\\nabla u` of a
    #         given scalar valued function :math:`u`, defined in the points.
    #         This is taken from
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
    #         gradient = np.zeros((num_points, 2), dtype=u.dtype)
    #
    #         # Create an empty 2x2 matrix for the boundary points to hold the
    #         # edge correction ((17) in [1]).
    #         boundary_matrices = {}
    #         for point in self.get_vertices('boundary'):
    #             boundary_matrices[point] = np.zeros((2, 2))
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
    #                 boundary_matrices[point0] += np.outer(r0, edge_coords)
    #             if point1 in boundary_matrices:
    #                 boundary_matrices[point1] += np.outer(r1, -edge_coords)
    #
    #         # Apply corrections to the gradients on the boundary.
    #         for k, value in boundary_matrices.items():
    #             gradient[k] = np.linalg.solve(value, gradient[k])
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
        A = 0.5 * np.sum(vector_field[self.idx_hierarchy], axis=0)
        # sum of <edge, A> for all three edges
        sum_edge_dot_A = np.einsum("ijk, ijk->j", self.half_edge_coords, A)

        # Get normalized vector orthogonal to triangle
        z = np.cross(self.half_edge_coords[0], self.half_edge_coords[1])

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
        return np.sum(self.ce_ratios_per_interior_edge < 0.0)

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
        show_edge_numbers=False,
        show_cell_numbers=False,
        cell_mask=None,
        mark_points=None,
        mark_edges=None,
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

        xmin = np.amin(self.points[:, 0])
        xmax = np.amax(self.points[:, 0])
        ymin = np.amin(self.points[:, 1])
        ymax = np.amax(self.points[:, 1])

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
                    bbox={"facecolor": "w", "alpha": 0.7},
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if show_edge_numbers:
            if self.edges is None:
                self.create_edges()
            for i, point_ids in enumerate(self.edges["points"]):
                midpoint = np.sum(self.points[point_ids], axis=0) / 2
                plt.text(
                    midpoint[0],
                    midpoint[1],
                    str(i),
                    bbox={"facecolor": "b", "alpha": 0.7},
                    color="w",
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        if show_cell_numbers:
            for i, x in enumerate(self.cell_centroids):
                plt.text(
                    x[0],
                    x[1],
                    str(i),
                    bbox={"facecolor": "r", "alpha": 0.5},
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

        if mark_points is not None:
            idx = mark_points
            plt.plot(self.points[idx, 0], self.points[idx, 1], "x", color="r")

        if mark_cells is not None:
            if np.asarray(mark_cells).dtype == bool:
                mark_cells = np.where(mark_cells)[0]

            patches = [
                Polygon(self.points[self.cells["points"][idx]]) for idx in mark_cells
            ]
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

            is_pos = np.zeros(len(self.edges["points"]), dtype=bool)
            is_pos[self.interior_edges[pos]] = True

            # Mark Delaunay-conforming boundary edges
            is_pos_boundary = self.ce_ratios[self.is_boundary_edge_local] >= 0
            is_pos[self.boundary_edges[is_pos_boundary]] = True

            line_segments0 = LineCollection(e[is_pos], color=mesh_color)
            ax.add_collection(line_segments0)
            #
            line_segments1 = LineCollection(e[~is_pos], color=nondelaunay_edge_color)
            ax.add_collection(line_segments1)

        if mark_edges is not None:
            e = self.points[self.edges["points"][mark_edges]][..., :2]
            ax.add_collection(LineCollection(e, color="r"))

        if show_coedges:
            # Connect all cell circumcenters with the edge midpoints
            cc = self.cell_circumcenters

            edge_midpoints = 0.5 * (
                self.points[self.edges["points"][:, 0]]
                + self.points[self.edges["points"][:, 1]]
            )

            # Plot connection of the circumcenter to the midpoint of all three
            # axes.
            a = np.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 0], :2]], axis=1
            )
            b = np.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 1], :2]], axis=1
            )
            c = np.stack(
                [cc[:, :2], edge_midpoints[self.cells["edges"][:, 2], :2]], axis=1
            )

            line_segments = LineCollection(
                np.concatenate([a, b, c]), color=comesh_color
            )
            ax.add_collection(line_segments)

        if boundary_edge_color:
            e = self.points[self.edges["points"][self.is_boundary_edge]][:, :, :2]
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
        edge_gids = np.where((self.edges["points"] == point_id).any(axis=1))[0]
        # ... and plot them
        for point_ids in self.edges["points"][edge_gids]:
            x = self.points[point_ids]
            ax.plot(x[:, 0], x[:, 1], "k")

        # Highlight ce_ratios.
        if show_ce_ratio:
            # Find the cells that contain the vertex
            cell_ids = np.where((self.cells["points"] == point_id).any(axis=1))[0]

            for cell_id in cell_ids:
                for edge_gid in self.cells["edges"][cell_id]:
                    if point_id not in self.edges["points"][edge_gid]:
                        continue
                    point_ids = self.edges["points"][edge_gid]
                    edge_midpoint = 0.5 * (
                        self.points[point_ids[0]] + self.points[point_ids[1]]
                    )
                    p = np.stack(
                        [self.cell_circumcenters[cell_id], edge_midpoint], axis=1
                    )
                    q = np.column_stack(
                        [
                            self.cell_circumcenters[cell_id],
                            edge_midpoint,
                            self.points[point_id],
                        ]
                    )
                    ax.fill(q[0], q[1], color="0.5")
                    ax.plot(p[0], p[1], color="0.7")
        return

    def flip_until_delaunay(self, tol=0.0, max_steps=100):
        """Flip edges until the mesh is fully Delaunay (up to `tol`)."""
        num_flips = 0
        assert tol >= 0.0
        # If all coedge/edge ratios are positive, all cells are Delaunay.
        if np.all(self.ce_ratios > -0.5 * tol):
            return num_flips

        # Now compute the boundary edges. A little more costly, but we'd have to do that
        # anyway. If all _interior_ coedge/edge ratios are positive, all cells are
        # Delaunay.
        if np.all(self.ce_ratios[~self.is_boundary_edge_local] > -0.5 * tol):
            return num_flips

        step = 0

        while np.any(self.ce_ratios_per_interior_edge < -tol):
            step += 1
            if step > max_steps:
                m = np.min(self.ce_ratios_per_interior_edge)
                warnings.warn(
                    f"Maximum number of edge flips reached. Smallest ce-ratio: {m:.3e}."
                )
                break
            is_flip_interior_edge = self.ce_ratios_per_interior_edge < -tol

            interior_edges_cells = self.edges_cells["interior"][1:3].T
            adj_cells = interior_edges_cells[is_flip_interior_edge].T

            # Check if there are cells for which more than one edge needs to be flipped.
            # For those, only flip one edge, namely that with the smaller (more
            # negative) ce_ratio.
            cell_gids, num_flips_per_cell = np.unique(adj_cells, return_counts=True)
            critical_cell_gids = cell_gids[num_flips_per_cell > 1]
            while np.any(num_flips_per_cell > 1):
                for cell_gid in critical_cell_gids:
                    edge_gids = self.cells["edges"][cell_gid]
                    is_interior_edge = self.is_interior_edge[edge_gids]
                    idx = self.edges_cells_idx[edge_gids[is_interior_edge]]
                    k = np.argmin(self.ce_ratios_per_interior_edge[idx])
                    is_flip_interior_edge[idx] = False
                    is_flip_interior_edge[idx[k]] = True

                adj_cells = interior_edges_cells[is_flip_interior_edge].T
                cell_gids, num_flips_per_cell = np.unique(adj_cells, return_counts=True)
                critical_cell_gids = cell_gids[num_flips_per_cell > 1]

            self.flip_interior_edges(is_flip_interior_edge)
            num_flips += np.sum(is_flip_interior_edge)

        return num_flips

    def flip_interior_edges(self, is_flip_interior_edge):
        edges_cells_flip = self.edges_cells["interior"][:, is_flip_interior_edge]
        edge_gids = edges_cells_flip[0]
        adj_cells = edges_cells_flip[1:3]
        lids = edges_cells_flip[3:5]

        #        3                   3
        #        A                   A
        #       /|\                 / \
        #     3/ | \2             3/   \2
        #     /  |  \             /  1  \
        #   0/ 0 |   \1   ==>   0/_______\1
        #    \   | 1 /           \       /
        #     \  |  /             \  0  /
        #     0\ | /1             0\   /1
        #       \|/                 \ /
        #        V                   V
        #        2                   2
        #
        v = np.array(
            [
                self.cells["points"][adj_cells[0], lids[0]],
                self.cells["points"][adj_cells[1], lids[1]],
                self.cells["points"][adj_cells[0], (lids[0] + 1) % 3],
                self.cells["points"][adj_cells[0], (lids[0] + 2) % 3],
            ]
        )

        # This must be computed before the points are reset
        equal_orientation = (
            self.cells["points"][adj_cells[0], (lids[0] + 1) % 3]
            == self.cells["points"][adj_cells[1], (lids[1] + 2) % 3]
        )

        # Set up new cells->points relationships.
        # Make sure that positive/negative area orientation is preserved. This is
        # especially important for signed area computations: In a mesh of all positive
        # areas, you don't want a negative area appear after an edge flip.
        self.cells["points"][adj_cells[0]] = np.column_stack([v[0], v[2], v[1]])
        self.cells["points"][adj_cells[1]] = np.column_stack([v[0], v[1], v[3]])

        # Set up new edges->points relationships.
        self.edges["points"][edge_gids] = np.sort(np.column_stack([v[0], v[1]]), axis=1)

        # Set up new cells->edges relationships.
        previous_edges = self.cells["edges"][adj_cells].copy()  # TODO need copy?
        # Do the neighboring cells have equal orientation (both point sets
        # clockwise/counterclockwise)?
        #
        # edges as in the above ascii art
        i0 = np.ones(equal_orientation.shape[0], dtype=int)
        i0[~equal_orientation] = 2
        i1 = np.ones(equal_orientation.shape[0], dtype=int)
        i1[equal_orientation] = 2
        e = [
            np.choose((lids[0] + 2) % 3, previous_edges[0].T),
            np.choose((lids[1] + i0) % 3, previous_edges[1].T),
            np.choose((lids[1] + i1) % 3, previous_edges[1].T),
            np.choose((lids[0] + 1) % 3, previous_edges[0].T),
        ]
        # The order here is tightly coupled to self.cells["points"] above
        self.cells["edges"][adj_cells[0]] = np.column_stack([e[1], edge_gids, e[0]])
        self.cells["edges"][adj_cells[1]] = np.column_stack([e[2], e[3], edge_gids])

        # update is_boundary_edge_local
        for k in range(3):
            self.is_boundary_edge_local[k, adj_cells] = self.is_boundary_edge[
                self.cells["edges"][adj_cells, k]
            ]

        # Update the edge->cells relationship. We need to update edges_cells info for
        # all five edges.
        # First update the flipped edge; it's always interior.
        idx = self.edges_cells_idx[edge_gids]
        # cell ids
        self.edges_cells["interior"][1, idx] = adj_cells[0]
        self.edges_cells["interior"][2, idx] = adj_cells[1]
        # local edge ids; see self.cells["edges"]
        self.edges_cells["interior"][3, idx] = 1
        self.edges_cells["interior"][4, idx] = 2
        #
        # Now handle the four surrounding edges
        conf = [
            # The data is:
            # (1) edge id
            # (2) previous adjacent cell (adj_cells[0] or adj_cells[1])
            # (3) new adjacent cell (adj_cells[0] or adj_cells[1])
            # (4) local edge index in the new adjacent cell
            (e[0], 0, 0, 2),
            (e[1], 1, 0, 0),
            (e[2], 1, 1, 0),
            (e[3], 0, 1, 1),
        ]
        for edge, prev_adj_idx, new__adj_idx, new_local_edge_index in conf:
            prev_adj = adj_cells[prev_adj_idx]
            new__adj = adj_cells[new__adj_idx]
            idx = self.edges_cells_idx[edge]
            # boundary...
            is_boundary = self.is_boundary_edge[edge]
            idx_bou = idx[is_boundary]
            prev_adjacent = prev_adj[is_boundary]
            new__adjacent = new__adj[is_boundary]
            # The assertion just makes sure we're doing the right thing. It should never
            # trigger.
            assert np.all(prev_adjacent == self.edges_cells["boundary"][1, idx_bou])
            self.edges_cells["boundary"][1, idx_bou] = new__adjacent
            self.edges_cells["boundary"][2, idx_bou] = new_local_edge_index
            # ...or interior?
            prev_adjacent = prev_adj[~is_boundary]
            new__adjacent = new__adj[~is_boundary]
            idx_int = idx[~is_boundary]
            # Interior edges have two neighboring cells in no particular order. Find out
            # if the adj_cell if the flipped edge comes first or second.
            is_first = prev_adjacent == self.edges_cells["interior"][1, idx_int]
            # The following is just a safety net. We could as well take ~is_first.
            is_secnd = prev_adjacent == self.edges_cells["interior"][2, idx_int]
            assert np.all(np.logical_xor(is_first, is_secnd))
            # actually set the data
            idx_first = idx_int[is_first]
            self.edges_cells["interior"][1, idx_first] = new__adjacent[is_first]
            self.edges_cells["interior"][3, idx_first] = new_local_edge_index
            # likewise for when the cell appears in the second column
            idx_secnd = idx_int[~is_first]
            self.edges_cells["interior"][2, idx_secnd] = new__adjacent[~is_first]
            self.edges_cells["interior"][4, idx_secnd] = new_local_edge_index

        # Schedule the cell ids for data updates
        update_cell_ids = np.unique(adj_cells.T.flat)
        # Same for edge ids
        update_edge_gids = self.cells["edges"][update_cell_ids].flat
        edge_cell_idx = self.edges_cells_idx[update_edge_gids]
        update_interior_edge_ids = np.unique(
            edge_cell_idx[self.is_interior_edge[update_edge_gids]]
        )

        self._update_cell_values(update_cell_ids, update_interior_edge_ids)

    def _update_cell_values(self, cell_ids, interior_edge_ids):
        """Updates all sorts of cell information for the given cell IDs."""
        # update idx_hierarchy
        nds = self.cells["points"][cell_ids].T
        self.idx_hierarchy[..., cell_ids] = nds[self.local_idx]

        # update self.half_edge_coords
        self.half_edge_coords[:, cell_ids, :] = np.moveaxis(
            self.points[self.idx_hierarchy[1, ..., cell_ids]]
            - self.points[self.idx_hierarchy[0, ..., cell_ids]],
            0,
            1,
        )

        # update self.ei_dot_ei
        e = self.half_edge_coords[:, cell_ids]
        self.ei_dot_ei[:, cell_ids] = np.einsum("...k,...k->...", e, e)

        # update self.ei_dot_ej
        self._ei_dot_ej[:, cell_ids] = (
            self.ei_dot_ei[:, cell_ids]
            - np.sum(self.ei_dot_ei[:, cell_ids], axis=0) / 2
        )

        # update cell_volumes, ce_ratios_per_half_edge
        cv = compute_tri_areas(self.ei_dot_ej[:, cell_ids])
        self.cell_volumes[cell_ids] = cv

        if self._ce_ratios is not None:
            ce = compute_ce_ratios(self.ei_dot_ej[:, cell_ids], cv)
            self._ce_ratios[:, cell_ids] = ce

        if self._interior_ce_ratios is not None:
            self._interior_ce_ratios[interior_edge_ids] = 0.0
            edge_gids = self.interior_edges[interior_edge_ids]
            adj_cells = self.edges_cells["interior"][1:3, interior_edge_ids].T

            is_edge = np.array(
                [
                    self.cells["edges"][adj_cells[:, 0]][:, k] == edge_gids
                    for k in range(3)
                ]
            )
            assert np.all(np.sum(is_edge, axis=0) == 1)
            for k in range(3):
                self._interior_ce_ratios[
                    interior_edge_ids[is_edge[k]]
                ] += self.ce_ratios[k, adj_cells[is_edge[k], 0]]

            is_edge = np.array(
                [
                    self.cells["edges"][adj_cells[:, 1]][:, k] == edge_gids
                    for k in range(3)
                ]
            )
            assert np.all(np.sum(is_edge, axis=0) == 1)
            for k in range(3):
                self._interior_ce_ratios[
                    interior_edge_ids[is_edge[k]]
                ] += self.ce_ratios[k, adj_cells[is_edge[k], 1]]

        if self._is_boundary_cell is not None:
            self._is_boundary_cell[cell_ids] = np.any(
                self.is_boundary_edge_local[:, cell_ids], axis=0
            )

        # TODO update those values
        self._cell_centroids = None
        self._edge_lengths = None
        self._cell_circumcenters = None
        self._control_volumes = None
        self._cell_partitions = None
        self._cv_centroids = None
        self._signed_cell_areas = None
        self.subdomains = {}
