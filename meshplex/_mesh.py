import math
import pathlib
import warnings

import meshio
import npx
import numpy as np
from numpy.typing import ArrayLike

from ._exceptions import MeshplexError
from ._helpers import _dot, _multiply, grp_start_len

__all__ = ["Mesh"]


class Mesh:
    def __init__(self, points, cells, sort_cells: bool = False):
        points = np.asarray(points)
        cells = np.asarray(cells)

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

        # assert len(points.shape) <= 2, f"Illegal point coordinates shape {points.shape}"
        assert len(cells.shape) == 2, f"Illegal cells shape {cells.shape}"
        self.n = cells.shape[1]

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

        # Initialize the idx hierarchy. The first entry, idx[0], is the cells->points
        # relationship, shape [3, numcells] for triangles and [4, numcells] for
        # tetrahedra. idx[1] is the (half-)facet->points to relationship, shape [2, 3,
        # numcells] for triangles and [3, 4, numcells] for tetrahedra, for example. The
        # indexing is chosen such the point idx[0][k] is opposite of the facet idx[1][:,
        # k]. This indexing keeps going until idx[-1] is of shape [2, 3, ..., numcells].
        self.idx = [np.asarray(cells).T]
        for _ in range(1, self.n - 1):
            m = len(self.idx[-1])
            r = np.arange(m)
            k = np.array([np.roll(r, -i) for i in range(1, m)])
            self.idx.append(self.idx[-1][k])

        self._is_point_used = None

        self._is_boundary_facet = None
        self._is_boundary_facet_local = None
        self.facets = None
        self._boundary_facets = None
        self._interior_facets = None
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_cell = None
        self._cells_facets = None

        self.subdomains = {}

        self._reset_point_data()

    def _reset_point_data(self):
        """Reset all data that changes when point coordinates changes."""
        self._half_edge_coords = None
        self._ei_dot_ei = None
        self._cell_centroids = None
        self._volumes = None
        self._integral_x = None
        self._signed_cell_volumes = None
        self._circumcenters = None
        self._cell_circumradii = None
        self._cell_heights = None
        self._ce_ratios = None
        self._cell_partitions = None
        self._control_volumes = None
        self._signed_circumcenter_distances = None
        self._circumcenter_facet_distances = None

        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._cv_cell_mask = None

    def __repr__(self):
        name = {
            2: "line",
            3: "triangle",
            4: "tetra",
        }[self.cells("points").shape[1]]
        num_points = len(self.points)
        num_cells = len(self.cells("points"))
        string = f"<meshplex {name} mesh, {num_points} points, {num_cells} cells>"
        return string

    # prevent overriding points without adapting the other mesh data
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, new_points: ArrayLike):
        new_points = np.asarray(new_points)
        assert new_points.shape == self._points.shape
        self._points = new_points
        # reset all computed values
        self._reset_point_data()

    def set_points(self, new_points: ArrayLike, idx=slice(None)):
        self.points.setflags(write=True)
        self.points[idx] = new_points
        self.points.setflags(write=False)
        self._reset_point_data()

    def cells(self, which):
        if which == "points":
            return self.idx[0].T
        elif which == "facets":
            assert self._cells_facets is not None
            return self._cells_facets

        assert which == "edges"
        assert self.n == 3
        return self._cells_facets

    @property
    def half_edge_coords(self):
        if self._half_edge_coords is None:
            self._compute_cell_values()
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            self._compute_cell_values()
        return self._ei_dot_ei

    @property
    def cell_heights(self):
        if self._cell_heights is None:
            self._compute_cell_values()
        return self._cell_heights

    @property
    def edge_lengths(self):
        if self._volumes is None:
            self._compute_cell_values()
        return self._volumes[0]

    @property
    def facet_areas(self):
        if self.n == 2:
            return np.ones(len(self.facets["points"]))
        if self._volumes is None:
            self._compute_cell_values()
        return self._volumes[-2]

    @property
    def cell_volumes(self):
        if self._volumes is None:
            self._compute_cell_values()
        return self._volumes[-1]

    @property
    def cell_circumcenters(self):
        """Get the center of the circumsphere of each cell."""
        if self._circumcenters is None:
            self._compute_cell_values()
        return self._circumcenters[-1]

    @property
    def cell_circumradius(self):
        """Get the circumradii of all cells"""
        if self._cell_circumradii is None:
            self._compute_cell_values()
        return self._cell_circumradii

    @property
    def cell_partitions(self):
        """Each simplex can be subdivided into parts that a closest to each corner.
        This method gives those parts, like ce_ratios associated with each edge.
        """
        if self._cell_partitions is None:
            self._compute_cell_values()
        return self._cell_partitions

    @property
    def circumcenter_facet_distances(self) -> ArrayLike:
        if self._circumcenter_facet_distances is None:
            self._compute_cell_values()
        return self._circumcenter_facet_distances

    def get_control_volume_centroids(self, cell_mask=None):
        """The centroid of any volume V is given by

        .. math::
          c = \\int_V x / \\int_V 1.

        The denominator is the control volume. The numerator can be computed by making
        use of the fact that the control volume around any vertex is composed of right
        triangles, two for each adjacent cell.

        Optionally disregard the contributions from particular cells. This is useful,
        for example, for temporarily disregarding flat cells on the boundary when
        performing Lloyd mesh optimization.
        """
        if self._cv_centroids is None or np.any(cell_mask != self._cvc_cell_mask):
            if self._integral_x is None:
                self._compute_cell_values()

            if cell_mask is None:
                idx = Ellipsis
            else:
                cell_mask = np.asarray(cell_mask)
                assert cell_mask.dtype == bool
                assert cell_mask.shape == (self.idx[-1].shape[-1],)
                # Use ":" for the first n-1 dimensions, then cell_mask
                idx = tuple((self.n - 1) * [slice(None)] + [~cell_mask])

            # TODO this can be improved by first summing up all components per cell
            integral_p = npx.sum_at(
                self._integral_x[idx], self.idx[-1][idx], len(self.points)
            )

            # Divide by the control volume
            cv = self.get_control_volumes(cell_mask)
            self._cv_centroids = (integral_p.T / cv).T
            self._cvc_cell_mask = cell_mask

        return self._cv_centroids

    @property
    def control_volume_centroids(self):
        return self.get_control_volume_centroids()

    @property
    def ce_ratios(self):
        """The covolume-edgelength ratios."""
        # There are many ways for computing the ratio of the covolume and the edge
        # length. For triangles, for example, there is
        #
        #   ce_ratios = -<ei, ej> / cell_volume / 4,
        #
        # for tetrahedra,
        #
        #   zeta = (
        #       + ei_dot_ej[0, 2] * ei_dot_ej[3, 5] * ei_dot_ej[5, 4]
        #       + ei_dot_ej[0, 1] * ei_dot_ej[3, 5] * ei_dot_ej[3, 4]
        #       + ei_dot_ej[1, 2] * ei_dot_ej[3, 4] * ei_dot_ej[4, 5]
        #       + self.ei_dot_ej[0] * self.ei_dot_ej[1] * self.ei_dot_ej[2]
        #       ).
        #
        # Since we have detailed cell partitions at hand, though, the easiest and
        # fastest is via those.
        if self._ce_ratios is None:
            if self._cell_partitions is None:
                self._compute_cell_values()

            self._ce_ratios = (
                self._cell_partitions[0] / self.ei_dot_ei * 2 * (self.n - 1)
            )

        return self._ce_ratios

    @property
    def signed_circumcenter_distances(self):
        if self._signed_circumcenter_distances is None:
            if self._cells_facets is None:
                self.create_facets()

            self._signed_circumcenter_distances = npx.sum_at(
                self.circumcenter_facet_distances.T,
                self.cells("facets"),
                self.facets["points"].shape[0],
            )[self.is_interior_facet]

        return self._signed_circumcenter_distances

    def _compute_cell_values(self, mask=slice(None)):
        """Computes the volumes of all edges, facets, cells etc. in the mesh. It starts
        off by computing the (squared) edge lengths, then complements the edge with one
        vertex to form face. It computes an orthogonal basis of the face (with modified
        Gram-Schmidt), and from that gets the height of all faces. From this, the area
        of the face is computed. Then, it complements again to form the 3-simplex,
        again forms an orthogonal basis with Gram-Schmidt, and so on.
        """
        e = self.points[self.idx[-1][..., mask]]
        e0 = e[0]
        diff = e[1] - e[0]

        orthogonal_basis = np.array([diff])

        volumes2 = [_dot(diff, self.n - 1)]

        circumcenters = [0.5 * (e[0] + e[1])]

        vv = _dot(diff, self.n - 1)
        circumradii2 = 0.25 * vv
        sqrt_vv = np.sqrt(vv)
        lmbda = 0.5 * np.sqrt(vv)

        sumx = np.array(e + circumcenters[-1])

        partitions = 0.5 * np.sqrt(np.array([vv, vv]))

        norms2 = np.array(volumes2)
        for kk, idx in enumerate(self.idx[:-1][::-1]):
            # Use the orthogonal bases of all sides to get a vector `v` orthogonal to
            # the side, pointing towards the additional point `p0`.
            p0 = self.points[idx][:, mask]
            v = p0 - e0
            # modified gram-schmidt
            for w, ww in zip(orthogonal_basis, norms2):
                alpha = np.einsum("...k,...k->...", w, v) / ww
                v -= _multiply(w, alpha, self.n - 1 - kk)

            vv = np.einsum("...k,...k->...", v, v)

            # Form the orthogonal basis for the next iteration by choosing one side
            # `k0`. <https://gist.github.com/nschloe/3922801e200cf82aec2fb53c89e1c578>
            # shows that it doesn't make a difference which point-facet combination we
            # choose.
            k0 = 0
            e0 = e0[k0]
            orthogonal_basis = np.row_stack([orthogonal_basis[:, k0], [v[k0]]])
            norms2 = np.row_stack([norms2[:, k0], [vv[k0]]])

            # The squared volume is the squared volume of the face times the squared
            # height divided by (n+1) ** 2.
            volumes2.append(volumes2[-1][0] * vv[k0] / (kk + 2) ** 2)

            # get the distance to the circumcenter; used in cell partitions and
            # circumcenter/-radius computation
            c = circumcenters[-1]

            p0c2 = _dot(p0 - c, self.n - 1 - kk)
            #
            sigma = 0.5 * (p0c2 - circumradii2) / vv
            lmbda2 = sigma ** 2 * vv

            # circumcenter, squared circumradius
            # <https://math.stackexchange.com/a/4064749/36678>
            #
            circumradii2 = lmbda2[k0] + circumradii2[k0]
            circumcenters.append(c[k0] + _multiply(v[k0], sigma[k0], self.n - 2 - kk))

            sumx += circumcenters[-1]

            # cell partitions
            # don't use sqrt(lmbda2) here; lmbda can be negative
            sqrt_vv = np.sqrt(vv)
            lmbda = sigma * sqrt_vv
            partitions *= lmbda / (kk + 2)

        # The integral of x,
        #
        #   \\int_V x,
        #
        # over all atomic wedges, i.e., areas cornered by a point, an edge midpoint, and
        # the subsequent circumcenters.
        # The integral of any linear function over a triangle is the average of the
        # values of the function in each of the three corners, times the area of the
        # triangle.
        integral_x = _multiply(sumx, partitions / self.n, self.n)

        if np.all(mask == slice(None)):
            # set new values
            self._ei_dot_ei = volumes2[0]
            self._half_edge_coords = diff
            self._volumes = [np.sqrt(v2) for v2 in volumes2]
            self._circumcenter_facet_distances = lmbda
            self._cell_heights = sqrt_vv
            self._cell_circumradii = np.sqrt(circumradii2)
            self._circumcenters = circumcenters
            self._cell_partitions = partitions
            self._integral_x = integral_x
        else:
            # update existing values
            assert self._ei_dot_ei is not None
            self._ei_dot_ei[:, mask] = volumes2[0]

            assert self._half_edge_coords is not None
            self._half_edge_coords[:, mask] = diff

            assert self._volumes is not None
            for k in range(len(self._volumes)):
                self._volumes[k][..., mask] = np.sqrt(volumes2[k])

            assert self._circumcenter_facet_distances is not None
            self._circumcenter_facet_distances[..., mask] = lmbda

            assert self._cell_heights is not None
            self._cell_heights[..., mask] = sqrt_vv

            assert self._cell_circumradii is not None
            self._cell_circumradii[mask] = np.sqrt(circumradii2)

            assert self._circumcenters is not None
            for k in range(len(self._circumcenters)):
                self._circumcenters[k][..., mask, :] = circumcenters[k]

            assert self._cell_partitions is not None
            self._cell_partitions[..., mask] = partitions

            assert self._integral_x is not None
            self._integral_x[..., mask, :] = integral_x

    @property
    def signed_cell_volumes(self):
        """Signed volumes of an n-simplex in nD."""
        if self._signed_cell_volumes is None:
            self._signed_cell_volumes = self.compute_signed_cell_volumes()
        return self._signed_cell_volumes

    def compute_signed_cell_volumes(self, idx=slice(None)):
        n = self.points.shape[1]
        assert self.n == self.points.shape[1] + 1, (
            "Signed areas only make sense for n-simplices in in nD. "
            f"Got {n}D points."
        )
        if self.n == 3:
            # On <https://stackoverflow.com/q/50411583/353337>, we have a number of
            # alternatives computing the oriented area, but it's fastest with the
            # half-edges.
            x = self.half_edge_coords
            return (x[0, idx, 1] * x[2, idx, 0] - x[0, idx, 0] * x[2, idx, 1]) / 2

        # https://en.wikipedia.org/wiki/Simplex#Volume
        cp = self.points[self.cells("points")]
        # append ones; this appends a column instead of a row as suggested by
        # wikipedia, but that doesn't change the determinant
        cp1 = np.concatenate([cp, np.ones(cp.shape[:-1] + (1,))], axis=-1)
        return np.linalg.det(cp1) / math.factorial(n)

    def compute_cell_centroids(self, idx=slice(None)):
        return np.sum(self.points[self.cells("points")[idx]], axis=1) / self.n

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all
        simplices."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_cell_centroids()
        return self._cell_centroids

    cell_barycenters = cell_centroids

    @property
    def cell_incenters(self):
        """Get the midpoints of the inspheres."""
        # https://en.wikipedia.org/wiki/Incenter#Barycentric_coordinates
        # https://math.stackexchange.com/a/2864770/36678
        abc = self.facet_areas / np.sum(self.facet_areas, axis=0)
        return np.einsum("ij,jik->jk", abc, self.points[self.cells("points")])

    @property
    def cell_inradius(self):
        """Get the inradii of all cells"""
        # See <http://mathworld.wolfram.com/Incircle.html>.
        # https://en.wikipedia.org/wiki/Tetrahedron#Inradius
        return (self.n - 1) * self.cell_volumes / np.sum(self.facet_areas, axis=0)

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
            self._is_point_used[self.cells("points")] = True
        return self._is_point_used

    def write(
        self,
        filename: str,
        point_data=None,
        cell_data=None,
        field_data=None,
    ):
        if self.points.shape[1] == 2:
            n = len(self.points)
            a = np.ascontiguousarray(np.column_stack([self.points, np.zeros(n)]))
        else:
            a = self.points

        if self.cells("points").shape[1] == 3:
            cell_type = "triangle"
        else:
            assert (
                self.cells("points").shape[1] == 4
            ), "Only triangles/tetrahedra supported"
            cell_type = "tetra"

        meshio.Mesh(
            a,
            {cell_type: self.cells("points")},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        ).write(filename)

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return slice(None)
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]["vertices"]

    def get_edge_mask(self, subdomain=None):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return slice(None)

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx[-1]]
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
            return slice(None)

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx[-1]]
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
            return slice(None)

        if subdomain.is_boundary_only:
            # There are no boundary cells
            return np.array([])

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        is_in = self.subdomains[subdomain]["vertices"][self.idx[-1]]
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
                is_inside = is_inside & self.is_boundary_point

        self.subdomains[subdomain] = {"vertices": is_inside}

    def create_facets(self):
        """Set up facet->point and facet->cell relations."""
        if self.n == 2:
            # Too bad that the need a specializaiton here. Could be avoided if the
            # idx hierarchy would be of shape (1,2,...,n), not (2,...,n), but not sure
            # if that's worth the change.
            idx = self.idx[0].flatten()
        else:
            idx = self.idx[1]
            idx = idx.reshape(idx.shape[0], -1)

        # Sort the columns to make it possible for `unique()` to identify individual
        # facets.
        idx = np.sort(idx, axis=0).T
        a_unique, inv, cts = npx.unique_rows(
            idx, return_inverse=True, return_counts=True
        )

        if np.any(cts > 2):
            num_weird_edges = np.sum(cts > 2)
            msg = (
                f"Found {num_weird_edges} facets with more than two neighboring cells. "
                "Something is not right."
            )
            # check if cells are identical, list them
            _, inv, cts = npx.unique_rows(
                np.sort(self.cells("points")), return_inverse=True, return_counts=True
            )
            if np.any(cts > 1):
                msg += " The following cells are equal:\n"
                for multiple_idx in np.where(cts > 1)[0]:
                    msg += str(np.where(inv == multiple_idx)[0])
            raise MeshplexError(msg)

        self._is_boundary_facet_local = (cts[inv] == 1).reshape(self.idx[0].shape)
        self._is_boundary_facet = cts == 1

        self.facets = {"points": a_unique}
        # cell->facets relationship
        self._cells_facets = inv.reshape(self.n, -1).T

        if self.n == 3:
            self.edges = self.facets
            self._facets_cells = None
            self._facets_cells_idx = None
        elif self.n == 4:
            self.faces = self.facets

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
            if self.n == 2:
                self._is_boundary_point[
                    self.idx[0][self.is_boundary_facet_local]
                ] = True
            else:
                self._is_boundary_point[
                    self.idx[1][:, self.is_boundary_facet_local]
                ] = True
        return self._is_boundary_point

    @property
    def is_interior_point(self):
        if self._is_interior_point is None:
            self._is_interior_point = self.is_point_used & ~self.is_boundary_point
        return self._is_interior_point

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
        if self.facets is None:
            self.create_facets()

        # num_edges = len(self.edges["points"])
        # count = np.bincount(self.cells("edges").flat, minlength=num_edges)

        # <https://stackoverflow.com/a/50395231/353337>
        edges_flat = self.cells("edges").flat
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
            num_edges = len(self.facets["points"])
            self._facets_cells_idx = np.empty(num_edges, dtype=int)
            num_b = np.sum(self.is_boundary_facet)
            num_i = np.sum(self.is_interior_facet)
            self._facets_cells_idx[self.facets_cells["boundary"][0]] = np.arange(num_b)
            self._facets_cells_idx[self.facets_cells["interior"][0]] = np.arange(num_i)
        return self._facets_cells_idx

    def remove_dangling_points(self):
        """Remove all points which aren't part of an array"""
        is_part_of_cell = np.zeros(self.points.shape[0], dtype=bool)
        is_part_of_cell[self.cells("points").flat] = True

        new_point_idx = np.cumsum(is_part_of_cell) - 1

        self._points = self._points[is_part_of_cell]

        for k in range(len(self.idx)):
            self.idx[k] = new_point_idx[self.idx[k]]

        if self._control_volumes is not None:
            self._control_volumes = self._control_volumes[is_part_of_cell]

        if self._cv_centroids is not None:
            self._cv_centroids = self._cv_centroids[is_part_of_cell]

        if self.facets is not None:
            self.facets["points"] = new_point_idx[self.facets["points"]]

        if self._is_interior_point is not None:
            self._is_interior_point = self._is_interior_point[is_part_of_cell]

        if self._is_boundary_point is not None:
            self._is_boundary_point = self._is_boundary_point[is_part_of_cell]

        if self._is_point_used is not None:
            self._is_point_used = self._is_point_used[is_part_of_cell]

    @property
    def q_radius_ratio(self):
        """Ratio of incircle and circumcircle ratios times (n-1). ("Normalized shape
        ratio".) Is 1 for the equilateral simplex, and is often used a quality measure
        for the cell.
        """
        # There are other sensible possiblities of defining cell quality, e.g.:
        #   * inradius to longest edge
        #   * shortest to longest edge
        #   * minimum dihedral angle
        #   * ...
        # See
        # <http://eidors3d.sourceforge.net/doc/index.html?eidors/meshing/calc_mesh_quality.html>.
        if self.n == 3:
            # q = 2 * r_in / r_out
            #   = (-a+b+c) * (+a-b+c) * (+a+b-c) / (a*b*c),
            #
            # where r_in is the incircle radius and r_out the circumcircle radius
            # and a, b, c are the edge lengths.
            a, b, c = self.edge_lengths
            return (-a + b + c) * (a - b + c) * (a + b - c) / (a * b * c)

        return (self.n - 1) * self.cell_inradius / self.cell_circumradius

    def remove_cells(self, remove_array: ArrayLike):
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
            keep = np.ones(len(self.cells("points")), dtype=bool)
            keep[remove_array] = False
        else:
            assert remove_array.dtype == bool
            keep = ~remove_array

        assert len(keep) == len(self.cells("points")), "Wrong length of index array."

        if np.all(keep):
            return 0

        # handle facet; this is a bit messy
        if self._cells_facets is not None:
            # updating the boundary data is a lot easier with facets_cells
            if self._facets_cells is None:
                self._compute_facets_cells()

            # Set facet to is_boundary_facet_local=True if it is adjacent to a removed
            # cell.
            facet_ids = self.cells("facets")[~keep].flatten()
            # only consider interior facets
            facet_ids = facet_ids[self.is_interior_facet[facet_ids]]
            idx = self.facets_cells_idx[facet_ids]
            cell_id = self.facets_cells["interior"][1:3, idx].T
            local_facet_id = self.facets_cells["interior"][3:5, idx].T
            self._is_boundary_facet_local[local_facet_id, cell_id] = True
            # now remove the entries corresponding to the removed cells
            self._is_boundary_facet_local = self._is_boundary_facet_local[:, keep]

            if self._is_boundary_cell is not None:
                self._is_boundary_cell[cell_id] = True
                self._is_boundary_cell = self._is_boundary_cell[keep]

            # update facets_cells
            keep_b_ec = keep[self.facets_cells["boundary"][1]]
            keep_i_ec0, keep_i_ec1 = keep[self.facets_cells["interior"][1:3]]
            # move ec from interior to boundary if exactly one of the two adjacent cells
            # was removed

            keep_i_0 = keep_i_ec0 & ~keep_i_ec1
            keep_i_1 = keep_i_ec1 & ~keep_i_ec0
            self._facets_cells["boundary"] = np.array(
                [
                    # facet id
                    np.concatenate(
                        [
                            self._facets_cells["boundary"][0, keep_b_ec],
                            self._facets_cells["interior"][0, keep_i_0],
                            self._facets_cells["interior"][0, keep_i_1],
                        ]
                    ),
                    # cell id
                    np.concatenate(
                        [
                            self._facets_cells["boundary"][1, keep_b_ec],
                            self._facets_cells["interior"][1, keep_i_0],
                            self._facets_cells["interior"][2, keep_i_1],
                        ]
                    ),
                    # local facet id
                    np.concatenate(
                        [
                            self._facets_cells["boundary"][2, keep_b_ec],
                            self._facets_cells["interior"][3, keep_i_0],
                            self._facets_cells["interior"][4, keep_i_1],
                        ]
                    ),
                ]
            )

            keep_i = keep_i_ec0 & keep_i_ec1

            # this memory copy isn't too fast
            self._facets_cells["interior"] = self._facets_cells["interior"][:, keep_i]

            num_facets_old = len(self.facets["points"])
            adjacent_facets, counts = np.unique(
                self.cells("facets")[~keep].flat, return_counts=True
            )
            # remove facet entirely either if 2 adjacent cells are removed or if it is a
            # boundary facet and 1 adjacent cells are removed
            is_facet_removed = (counts == 2) | (
                (counts == 1) & self._is_boundary_facet[adjacent_facets]
            )

            # set the new boundary facet
            self._is_boundary_facet[adjacent_facets[~is_facet_removed]] = True
            # Now actually remove the facets. This includes a reindexing.
            assert self._is_boundary_facet is not None
            keep_facets = np.ones(len(self._is_boundary_facet), dtype=bool)
            keep_facets[adjacent_facets[is_facet_removed]] = False

            # make sure there is only facets["points"], not facets["cells"] etc.
            assert self.facets is not None
            assert len(self.facets) == 1
            self.facets["points"] = self.facets["points"][keep_facets]
            self._is_boundary_facet = self._is_boundary_facet[keep_facets]

            # update facet and cell indices
            self._cells_facets = self.cells("facets")[keep]
            new_index_facets = np.arange(num_facets_old) - np.cumsum(~keep_facets)
            self._cells_facets = new_index_facets[self.cells("facets")]
            num_cells_old = len(self.cells("points"))
            new_index_cells = np.arange(num_cells_old) - np.cumsum(~keep)

            # this takes fairly long
            ec = self._facets_cells
            ec["boundary"][0] = new_index_facets[ec["boundary"][0]]
            ec["boundary"][1] = new_index_cells[ec["boundary"][1]]
            ec["interior"][0] = new_index_facets[ec["interior"][0]]
            ec["interior"][1:3] = new_index_cells[ec["interior"][1:3]]

            # simply set those to None; their reset is cheap
            self._facets_cells_idx = None
            self._boundary_facets = None
            self._interior_facets = None

        for k in range(len(self.idx)):
            self.idx[k] = self.idx[k][..., keep]

        if self._volumes is not None:
            for k in range(len(self._volumes)):
                self._volumes[k] = self._volumes[k][..., keep]

        if self._ce_ratios is not None:
            self._ce_ratios = self._ce_ratios[:, keep]

        if self._half_edge_coords is not None:
            self._half_edge_coords = self._half_edge_coords[:, keep]

        if self._ei_dot_ei is not None:
            self._ei_dot_ei = self._ei_dot_ei[:, keep]

        if self._cell_centroids is not None:
            self._cell_centroids = self._cell_centroids[keep]

        if self._circumcenters is not None:
            for k in range(len(self._circumcenters)):
                self._circumcenters[k] = self._circumcenters[k][..., keep, :]

        if self._cell_partitions is not None:
            self._cell_partitions = self._cell_partitions[..., keep]

        if self._signed_cell_volumes is not None:
            self._signed_cell_volumes = self._signed_cell_volumes[keep]

        if self._integral_x is not None:
            self._integral_x = self._integral_x[..., keep, :]

        if self._circumcenter_facet_distances is not None:
            self._circumcenter_facet_distances = self._circumcenter_facet_distances[
                ..., keep
            ]

        # TODO These could also be updated, but let's implement it when needed
        self._signed_circumcenter_distances = None
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
        The input criterion is a callback that must return an array of length
        `sum(mesh.is_boundary_cell)`.

        This helps, for example, in the following scenario.
        When points are moving around, flip_until_delaunay() makes sure the mesh remains
        a Delaunay mesh. This does not work on boundaries where very flat cells can
        still occur or cells may even 'invert'. (The interior point moves outside.) In
        this case, the boundary cell can be removed, and the newly outward node is made
        a boundary node."""
        num_removed = 0
        while True:
            num_boundary_cells = np.sum(self.is_boundary_cell)
            crit = criterion(self.is_boundary_cell)

            if ~np.any(crit):
                break

            if not isinstance(crit, np.ndarray) or crit.shape != (num_boundary_cells,):
                raise ValueError(
                    "criterion() callback must return a Boolean NumPy array "
                    f"of shape {(num_boundary_cells,)}, got {crit.shape}."
                )

            idx = self.is_boundary_cell.copy()
            idx[idx] = crit
            n = self.remove_cells(idx)
            num_removed += n
            if n == 0:
                break
        return num_removed

    def remove_duplicate_cells(self):
        sorted_cells = np.sort(self.cells("points"))
        _, inv, cts = npx.unique_rows(
            sorted_cells, return_inverse=True, return_counts=True
        )

        remove = np.zeros(len(self.cells("points")), dtype=bool)
        for k in np.where(cts > 1)[0]:
            rem = inv == k
            # don't remove first occurence
            first_idx = np.where(rem)[0][0]
            rem[first_idx] = False
            remove |= rem

        return self.remove_cells(remove)

    def get_control_volumes(self, cell_mask=None):
        """The control volumes around each vertex. Optionally disregard the
        contributions from particular cells. This is useful, for example, for
        temporarily disregarding flat cells on the boundary when performing Lloyd mesh
        optimization.
        """
        if self._cv_centroids is None or np.any(cell_mask != self._cvc_cell_mask):
            # Sum up the contributions according to how self.idx is constructed.
            # roll = np.array([np.roll(np.arange(kk + 3), -i) for i in range(1, kk + 3)])
            # vols = npx.sum_at(vols, roll, kk + 3)
            # v = self.cell_partitions[..., idx]

            if cell_mask is None:
                idx = slice(None)
            else:
                idx = ~cell_mask

            # TODO this can be improved by first summing up all components per cell
            self._control_volumes = npx.sum_at(
                self.cell_partitions[..., idx],
                self.idx[-1][..., idx],
                len(self.points),
            )

            self._cv_cell_mask = cell_mask
        return self._control_volumes

    control_volumes = property(get_control_volumes)

    @property
    def is_delaunay(self):
        return self.num_delaunay_violations == 0

    @property
    def num_delaunay_violations(self):
        """Number of interior facets where the Delaunay condition is violated."""
        # Delaunay violations are present exactly on the interior facets where the
        # signed circumcenter distance is negative. Count those.
        return np.sum(self.signed_circumcenter_distances < 0.0)

    @property
    def idx_hierarchy(self):
        warnings.warn(
            "idx_hierarchy is deprecated, use idx[-1] instead", DeprecationWarning
        )
        return self.idx[-1]

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
        """Save the mesh to a file, either as a PNG/SVG or a mesh file"""
        if pathlib.Path(filename).suffix in [".png", ".svg"]:
            import matplotlib.pyplot as plt

            self.plot(*args, **kwargs)
            plt.savefig(filename, transparent=True, bbox_inches="tight")
            plt.close()
        else:
            self.write(filename)

    def plot(self, *args, **kwargs):
        if self.n == 2:
            self._plot_line(*args, **kwargs)
        else:
            assert self.n == 3
            self._plot_tri(*args, **kwargs)

    def _plot_line(self):
        import matplotlib.pyplot as plt

        if len(self.points.shape) == 1:
            x = self.points
            y = np.zeros(self.points.shape[0])
        else:
            assert len(self.points.shape) == 2 and self.points.shape[1] == 2
            x, y = self.points.T

        plt.plot(x, y, "-o")

    def _plot_tri(
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
                self.create_facets()
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
                self.cells("points"),
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
                Polygon(self.points[self.cells("points")[idx]]) for idx in mark_cells
            ]
            p = PatchCollection(patches, facecolor="C1")
            ax.add_collection(p)

        if self.edges is None:
            self.create_facets()

        # Get edges, cut off z-component.
        e = self.points[self.edges["points"]][:, :, :2]

        if nondelaunay_edge_color is None:
            line_segments0 = LineCollection(e, color=mesh_color)
            ax.add_collection(line_segments0)
        else:
            # Plot regular edges, mark those with negative ce-ratio red.
            is_pos = np.zeros(len(self.edges["points"]), dtype=bool)
            is_pos[self.interior_facets[self.signed_circumcenter_distances >= 0]] = True

            # Mark Delaunay-conforming boundary edges
            is_pos_boundary = self.ce_ratios[self.is_boundary_facet_local] >= 0
            is_pos[self.boundary_facets[is_pos_boundary]] = True

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
                [cc[:, :2], edge_midpoints[self.cells("edges")[:, 0], :2]], axis=1
            )
            b = np.stack(
                [cc[:, :2], edge_midpoints[self.cells("edges")[:, 1], :2]], axis=1
            )
            c = np.stack(
                [cc[:, :2], edge_midpoints[self.cells("edges")[:, 2], :2]], axis=1
            )

            line_segments = LineCollection(
                np.concatenate([a, b, c]), color=comesh_color
            )
            ax.add_collection(line_segments)

        if boundary_edge_color:
            e = self.points[self.edges["points"][self.is_boundary_facet]][:, :, :2]
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
