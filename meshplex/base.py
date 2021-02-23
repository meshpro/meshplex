import math
import warnings

import meshio
import numpy as np

from .exceptions import MeshplexError
from .helpers import compute_tri_areas, grp_start_len, unique_rows

__all__ = ["_SimplexMesh"]


class _SimplexMesh:
    def __init__(self, points, cells, sort_cells=False):
        if sort_cells:
            # Sort cells, first every row, then the rows themselves. This helps in many
            # downstream applications, e.g., when constructing linear systems with the
            # cells/edges. (When converting to CSR format, the I/J entries must be
            # sorted.) Don't use cells.sort(axis=1) to avoid
            # ```
            # ValueError: sort array is read-only
            # ```
            cells = np.sort(cells, axis=1)
            cells = cells[cells[:, 0].argsort()]

        points = np.asarray(points)
        cells = np.asarray(cells)
        assert len(points.shape) == 2, f"Illegal point coordinates shape {points.shape}"
        assert len(cells.shape) == 2, f"Illegal cells shape {cells.shape}"
        self.n = cells.shape[1]
        assert self.n in [3, 4], f"Illegal cells shape {cells.shape}"

        # Assert that all vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = np.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # points = points[uvertices]
        # ```
        # helps.
        # is_used = np.zeros(len(points), dtype=bool)
        # is_used[cells] = True
        # assert np.all(is_used), "There are {} dangling points in the mesh".format(
        #     np.sum(~is_used)
        # )

        self._points = np.asarray(points)
        # prevent accidental override of parts of the array
        self._points.setflags(write=False)

        self.cells = {"points": np.asarray(cells)}
        nds = self.cells["points"].T

        if cells.shape[1] == 3:
            # Create the idx_hierarchy (points->edges->cells), i.e., the value of
            # `self.idx_hierarchy[0, 2, 27]` is the index of the point of cell 27, edge
            # 2, point 0. The shape of `self.idx_hierarchy` is `(2, 3, n)`, where `n` is
            # the number of cells. Make sure that the k-th edge is opposite of the k-th
            # point in the triangle.
            self.local_idx = np.array([[1, 2], [2, 0], [0, 1]]).T
        else:
            assert cells.shape[1] == 4
            # Arrange the point_face_cells such that point k is opposite of face k in
            # each cell.
            idx = np.array([[1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]]).T
            self.point_face_cells = nds[idx]

            # Arrange the idx_hierarchy (point->edge->face->cells) such that
            #
            #   * point k is opposite of edge k in each face,
            #   * duplicate edges are in the same spot of the each of the faces,
            #   * all points are in domino order ([1, 2], [2, 3], [3, 1]),
            #   * the same edges are easy to find:
            #      - edge 0: face+1, edge 2
            #      - edge 1: face+2, edge 1
            #      - edge 2: face+3, edge 0
            #   * opposite edges are easy to find, too:
            #      - edge 0  <-->  (face+2, edge 0)  equals  (face+3, edge 2)
            #      - edge 1  <-->  (face+1, edge 1)  equals  (face+3, edge 1)
            #      - edge 2  <-->  (face+1, edge 0)  equals  (face+2, edge 2)
            #
            self.local_idx = np.array(
                [
                    [[2, 3], [3, 1], [1, 2]],
                    [[3, 0], [0, 2], [2, 3]],
                    [[0, 1], [1, 3], [3, 0]],
                    [[1, 2], [2, 0], [0, 1]],
                ]
            ).T

        # Map idx back to the points. This is useful if quantities which are in idx
        # shape need to be added up into points (e.g., equation system rhs).
        self.idx_hierarchy = nds[self.local_idx]

        # The inverted local index.
        # This array specifies for each of the three points which edge endpoints
        # correspond to it. For triangles, the above local_idx should give
        #
        #    [[(1, 1), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (0, 1)]]
        #
        self.local_idx_inv = [
            [tuple(i) for i in zip(*np.where(self.local_idx == k))]
            for k in range(self.n)
        ]

        self._is_point_used = None

        self._edge_lengths = None
        self._facet_areas = None
        self._signed_cell_volumes = None
        self._cell_volumes = None
        self._heights = None

        # only used for tetra
        self._zeta = None

        self._is_boundary_facet = None
        self._is_boundary_facet_local = None
        self.edges = None
        self._boundary_facets = None
        self._interior_facets = None
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_cell = None

    def __repr__(self):
        name = {
            3: "triangle",
            4: "tetra",
        }[self.cells["points"].shape[1]]
        num_points = len(self.points)
        num_cells = len(self.cells["points"])
        string = f"<meshplex {name} mesh, {num_points} points, {num_cells} cells>"
        return string

    # prevent overriding points without adapting the other mesh data
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points):
        new_points = np.asarray(new_points)
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
    def half_edge_coords(self):
        if self._half_edge_coords is None:
            p = self.points[self.idx_hierarchy]
            self._half_edge_coords = p[1] - p[0]
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
            # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
            # TODO reorganize the data?
            self._ei_dot_ei = np.einsum(
                "...k, ...k->...", self.half_edge_coords, self.half_edge_coords
            )
        return self._ei_dot_ei

    @property
    def ei_dot_ej(self):
        if self._ei_dot_ej is None:
            self._ei_dot_ej = self.ei_dot_ei - np.sum(self.ei_dot_ei, axis=0) / 2
            # An alternative is
            # ```
            # self._ei_dot_ej = np.einsum(
            #     "...k, ...k->...",
            #     self.half_edge_coords[[1, 2, 0]],
            #     self.half_edge_coords[[2, 0, 1]],
            # )
            # ```
            # but this is slower, cf.
            # <https://gist.github.com/nschloe/d9c1d872a3ab8b47ff22d97d103be266>.
        return self._ei_dot_ej

    def compute_centroids(self, idx=slice(None)):
        return np.sum(self.points[self.cells["points"][idx]], axis=1) / self.n

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all
        simplices."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_centroids()
        return self._cell_centroids

    @property
    def heights(self):
        if self._heights is None:
            # compute the distance between the base (n-1)-simplex and the left-over
            # point
            # See <https://math.stackexchange.com/a/4025438/36678>.
            cp = self.cells["points"]
            n = cp.shape[1]
            base_idx = np.moveaxis(
                np.array([cp[:, np.arange(n) != k] for k in range(n)]),
                -1,
                0,
            )
            base = self.points[base_idx]
            tip = self.points[cp.T]

            A = base - tip
            assert A.shape == base.shape
            ATA = np.einsum("i...j,k...j->...ik", A, A)
            e = np.ones(ATA.shape[:-1])
            self._heights = np.sqrt(1 / np.sum(np.linalg.solve(ATA, e), axis=-1))

        return self._heights

    @property
    def cell_barycenters(self):
        """See cell_centroids."""
        return self.cell_centroids

    @property
    def is_point_used(self):
        # Check which vertices are used.
        # If there are vertices which do not appear in the cells list, this
        # ```
        # uvertices, uidx = np.unique(cells, return_inverse=True)
        # cells = uidx.reshape(cells.shape)
        # points = points[uvertices]
        # ```
        # helps.
        if self._is_point_used is None:
            self._is_point_used = np.zeros(len(self.points), dtype=bool)
            self._is_point_used[self.cells["points"]] = True
        return self._is_point_used

    def write(self, filename, point_data=None, cell_data=None, field_data=None):
        if self.points.shape[1] == 2:
            n = len(self.points)
            a = np.ascontiguousarray(np.column_stack([self.points, np.zeros(n)]))
        else:
            a = self.points

        if self.cells["points"].shape[1] == 3:
            cell_type = "triangle"
        else:
            assert (
                self.cells["points"].shape[1] == 4
            ), "Only triangles/tetrahedra supported"
            cell_type = "tetra"

        meshio.write_points_cells(
            filename,
            a,
            {cell_type: self.cells["points"]},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    @property
    def edge_lengths(self):
        if self._edge_lengths is None:
            self._edge_lengths = np.sqrt(self.ei_dot_ei)
        return self._edge_lengths

    @property
    def facet_areas(self):
        if self._facet_areas is None:
            if self.n == 3:
                self._facet_areas = self.edge_lengths
            else:
                assert self.n == 4
                self._facet_areas = compute_tri_areas(self.ei_dot_ej)

        return self._facet_areas

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return np.s_[:]
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]["vertices"]

    def get_edge_mask(self, subdomain=None):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return np.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over the first index
        is_inside = np.all(is_in, axis=tuple(range(1)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet

        return is_inside

    def get_face_mask(self, subdomain):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return np.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last two (face_ids, cell_ids).
        n = len(is_in.shape)
        is_inside = np.all(is_in, axis=tuple(range(n - 2)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet_local

        return is_inside

    def get_cell_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return np.s_[:]

        if subdomain.is_boundary_only:
            # There are no boundary cells
            return np.array([])

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last one (cell_ids).
        n = len(is_in.shape)
        return np.all(is_in, axis=tuple(range(n - 1)))

    def _mark_vertices(self, subdomain):
        """Mark faces/edges which are fully in subdomain."""
        if subdomain is None:
            is_inside = np.ones(len(self.points), dtype=bool)
        else:
            is_inside = subdomain.is_inside(self.points.T).T

            if subdomain.is_boundary_only:
                # Filter boundary
                self.mark_boundary()
                is_inside = is_inside & self.is_boundary_point

        self.subdomains[subdomain] = {"vertices": is_inside}

    @property
    def signed_cell_volumes(self):
        """Signed volumes of an n-simplex in nD."""
        if self._signed_cell_volumes is None:
            self._signed_cell_volumes = self.compute_signed_cell_volumes()
        return self._signed_cell_volumes

    def compute_signed_cell_volumes(self, idx=slice(None)):
        n = self.points.shape[1]
        assert n == self.cells["points"].shape[1] - 1, (
            "Signed areas only make sense for n-simplices in in nD. "
            f"Got {n}D points."
        )
        if n == 2:
            # On <https://stackoverflow.com/q/50411583/353337>, we have a number of
            # alternatives computing the oriented area, but it's fastest with the
            # half-edges.
            x = self.half_edge_coords
            out = (x[0, idx, 1] * x[2, idx, 0] - x[0, idx, 0] * x[2, idx, 1]) / 2
        else:
            # https://en.wikipedia.org/wiki/Simplex#Volume
            cp = self.points[self.cells["points"]]
            # append ones
            cp1 = np.concatenate([cp, np.ones(cp.shape[:-1] + (1,))], axis=-1)
            out = np.linalg.det(cp1) / math.factorial(n)
        return out

    @property
    def zeta(self):
        assert self.n == 4
        ee = self.ei_dot_ej
        self._zeta = (
            -ee[2, [1, 2, 3, 0]] * ee[1] * ee[2]
            - ee[1, [2, 3, 0, 1]] * ee[2] * ee[0]
            - ee[0, [3, 0, 1, 2]] * ee[0] * ee[1]
            + ee[0] * ee[1] * ee[2]
        )
        return self._zeta

    @property
    def cell_volumes(self):
        if self._cell_volumes is None:
            if self.n == 3:
                self._cell_volumes = compute_tri_areas(self.ei_dot_ej)
            else:
                assert self.n == 4
                # sum(self.circumcenter_face_distances * face_areas / 3) = cell_volumes
                # =>
                # cell_volumes = np.sqrt(sum(zeta / 72))
                self._cell_volumes = np.sqrt(np.sum(self.zeta, axis=0) / 72.0)

        # For higher-dimensional volumes, check out the Cayley-Menger determinant
        # <http://mathworld.wolfram.com/Cayley-MengerDeterminant.html> or the
        # computation via heights.
        return self._cell_volumes

    @property
    def cell_incenters(self):
        """Get the midpoints of the inspheres."""
        # https://en.wikipedia.org/wiki/Incenter#Barycentric_coordinates
        # https://math.stackexchange.com/a/2864770/36678
        abc = self.facet_areas / np.sum(self.facet_areas, axis=0)
        return np.einsum("ij,jik->jk", abc, self.points[self.cells["points"]])

    @property
    def cell_inradius(self):
        """Get the inradii of all cells"""
        # See <http://mathworld.wolfram.com/Incircle.html>.
        # https://en.wikipedia.org/wiki/Tetrahedron#Inradius
        return (self.n - 1) * self.cell_volumes / np.sum(self.facet_areas, axis=0)

    def create_facets(self):
        """Set up facet->point and facet->cell relations."""
        s = self.idx_hierarchy.shape
        idx_zero = tuple([0] * (len(s) - 3))
        idx = self.idx_hierarchy[idx_zero]
        # Reshape into individual edges.
        # Reshape into individual faces, and take the first point per edge. (The
        # face is fully characterized by it.) Sort the columns to make it possible
        # for `unique()` to identify individual faces.
        idx = idx.reshape(idx.shape[0], -1)

        a = np.sort(idx.T)

        # exit(1)
        # a = np.sort(self.idx_hierarchy.reshape(s[0], -1).T)
        b = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
        )
        # Sort the columns to make it possible for `unique()` to identify
        # individual edges.
        # a_unique, inv, cts = unique_rows(a)
        a_unique, inv, cts = np.unique(b, return_inverse=True, return_counts=True)
        a_unique = a_unique.view(a.dtype).reshape(-1, a.shape[1])

        if np.any(cts >= 3):
            cts = cts[cts >= 3]
            num_excess_cells = np.sum(cts >= 3)
            raise MeshplexError(
                "No facet (edge for triangles, face in tetrahedra) "
                "should have more than two cells, "
                f"but found {num_excess_cells} such edges. "
                "Are cells listed twice?"
            )

        self._is_boundary_facet_local = (cts[inv] == 1).reshape(s[self.n - 2 :])
        self._is_boundary_facet = cts == 1

        if self.n == 3:
            self.edges = {"points": a_unique}

            # cell->edges relationship
            self.cells["edges"] = inv.reshape(3, -1).T

            self._facets_cells = None
            self._facets_cells_idx = None
        else:
            assert self.n == 4
            self.faces = {"points": a_unique}

            # cell->faces relationship
            num_cells = len(self.cells["points"])
            self.cells["faces"] = inv.reshape([4, -1]).T

            # save for create_edge_cells
            self._inv_faces = inv

    @property
    def is_boundary_facet_local(self):
        if self._is_boundary_facet_local is None:
            self.create_facets()
        return self._is_boundary_facet_local

    @property
    def is_boundary_facet(self):
        if self._is_boundary_facet is None:
            self.create_facets()
        return self._is_boundary_facet

    @property
    def is_interior_facet(self):
        return ~self.is_boundary_facet

    @property
    def is_boundary_cell(self):
        if self._is_boundary_cell is None:
            assert self.is_boundary_facet_local is not None
            self._is_boundary_cell = np.any(self.is_boundary_facet_local, axis=0)
        return self._is_boundary_cell

    @property
    def boundary_facets(self):
        if self._boundary_facets is None:
            self._boundary_facets = np.where(self.is_boundary_facet)[0]
        return self._boundary_facets

    @property
    def interior_facets(self):
        if self._interior_facets is None:
            self._interior_facets = np.where(~self.is_boundary_facet)[0]
        return self._interior_facets

    @property
    def is_boundary_point(self):
        if self._is_boundary_point is None:
            self._is_boundary_point = np.zeros(len(self.points), dtype=bool)
            self._is_boundary_point[
                self.idx_hierarchy[..., self.is_boundary_facet_local]
            ] = True
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self._is_interior_point = self.is_point_used & ~self.is_boundary_point
        return self._is_interior_point

    def mark_boundary(self):
        warnings.warn(
            "mark_boundary() does nothing. "
            "Boundary entities are computed on the fly.",
            DeprecationWarning,
        )

    @property
    def facets_cells(self):
        if self._facets_cells is None:
            self._compute_facets_cells()
        return self._facets_cells

    def _compute_facets_cells(self):
        """This creates edge->cells relations. While it's not necessary for many
        applications, it sometimes does come in handy, for example for mesh
        manipulation.
        """
        if self.edges is None:
            self.create_facets()

        # num_edges = len(self.edges["points"])
        # count = np.bincount(self.cells["edges"].flat, minlength=num_edges)

        # <https://stackoverflow.com/a/50395231/353337>
        edges_flat = self.cells["edges"].flat
        idx_sort = np.argsort(edges_flat)
        sorted_edges = edges_flat[idx_sort]
        idx_start, count = grp_start_len(sorted_edges)

        # count is redundant with is_boundary/interior_edge
        assert np.all((count == 1) == self.is_boundary_facet)
        assert np.all((count == 2) == self.is_interior_facet)

        idx_start_count_1 = idx_start[self.is_boundary_facet]
        idx_start_count_2 = idx_start[self.is_interior_facet]
        res1 = idx_sort[idx_start_count_1]
        res2 = idx_sort[np.array([idx_start_count_2, idx_start_count_2 + 1])]

        edge_id_boundary = sorted_edges[idx_start_count_1]
        edge_id_interior = sorted_edges[idx_start_count_2]

        # It'd be nicer if we could organize the data differently, e.g., as a structured
        # array or as a dict. Those possibilities are slower, unfortunately, for some
        # operations in remove_cells() (and perhaps elsewhere).
        # <https://github.com/numpy/numpy/issues/17850>
        self._facets_cells = {
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

        self._facets_cells_idx = None

    @property
    def facets_cells_idx(self):
        if self._facets_cells_idx is None:
            if self._facets_cells is None:
                self._compute_facets_cells()
            assert self.is_boundary_facet is not None
            # For each edge, store the index into the respective edge array.
            num_edges = len(self.edges["points"])
            self._facets_cells_idx = np.empty(num_edges, dtype=int)
            num_b = np.sum(self.is_boundary_facet)
            num_i = np.sum(self.is_interior_facet)
            self._facets_cells_idx[self.facets_cells["boundary"][0]] = np.arange(num_b)
            self._facets_cells_idx[self.facets_cells["interior"][0]] = np.arange(num_i)
        return self._facets_cells_idx
