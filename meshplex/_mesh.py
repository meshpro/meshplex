import math
import warnings

import meshio
import npx
import numpy as np

from ._exceptions import MeshplexError
from ._helpers import _dot, _multiply, compute_ce_ratios, grp_start_len

__all__ = ["Mesh"]


class Mesh:
    def __init__(self, points, cells, sort_cells=False):
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

        self.cells = {"points": np.asarray(cells)}

        # Initialize the idx hierarchy. The first entry, idx[0], is the cells->points
        # relationship, shape [3, numcells] for triangles and [4, numcells] for
        # tetrahedra. idx[1] is the (half-)facet->points to relationship, shape [2, 3,
        # numcells] for triangles and [3, 4, numcells] for tetrahedra, for example. The
        # indexing is chosen such the point idx[0][k] is opposite of the facet idx[1][:,
        # k]. This indexing keeps going until idx[-1] is of shape [2, 3, ..., numcells].
        self.idx = [self.cells["points"].T]
        for _ in range(1, self.n - 1):
            m = len(self.idx[-1])
            r = np.arange(m)
            k = np.array([np.roll(r, -i) for i in range(1, m)])
            self.idx.append(self.idx[-1][k])

        self._is_point_used = None

        self._is_boundary_facet = None
        self._is_boundary_facet_local = None
        self.edges = None
        self._boundary_facets = None
        self._interior_facets = None
        self._is_interior_point = None
        self._is_boundary_point = None
        self._is_boundary_cell = None

        self.subdomains = {}

        self._reset_point_data()

    def _reset_point_data(self):
        """Reset all data that changes when point coordinates changes."""
        self._half_edge_coords = None
        self._ei_dot_ei = None
        self._ei_dot_ej = None
        self._cell_centroids = None
        self._volumes = None
        self._integral_x = None
        self._signed_cell_volumes = None
        self._circumcenters = None
        self._circumradii2 = None
        self._heights = None
        self._ce_ratios = None
        self._partitions = None
        self._control_volumes = None
        self._interior_ce_ratios = None

        self._cv_centroids = None
        self._cvc_cell_mask = None
        self._cv_cell_mask = None

    def __repr__(self):
        name = {
            2: "line",
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
            p = self.points[self.idx[-1]]
            self._half_edge_coords = p[1] - p[0]
        return self._half_edge_coords

    @property
    def ei_dot_ei(self):
        if self._ei_dot_ei is None:
            self._ei_dot_ei = _dot(self.half_edge_coords, self.n - 1)
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

    def compute_cell_centroids(self, idx=slice(None)):
        return np.sum(self.points[self.cells["points"][idx]], axis=1) / self.n

    @property
    def cell_centroids(self):
        """The centroids (barycenters, midpoints of the circumcircles) of all
        simplices."""
        if self._cell_centroids is None:
            self._cell_centroids = self.compute_cell_centroids()
        return self._cell_centroids

    cell_barycenters = cell_centroids

    @property
    def cell_heights(self):
        # TODO remove or get from _compute_things()
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

    def _compute_things(self):
        """Computes the volumes of all edges, facets, cells etc. in the mesh. It starts
        off by computing the (squared) edge lengths, then complements the edge with one
        vertex to form face. It computes an orthogonal basis of the face (with modified
        Gram-Schmidt), and from that gets the height of all faces. From this, the area
        of the face is computed. Then, it complements again to form the 3-simplex,
        again forms an orthogonal basis with Gram-Schmidt, and so on.
        """
        e = self.points[self.idx[-1]]
        e0 = e[0]
        diff = e[1] - e[0]
        orthogonal_basis = np.array([diff])

        volumes2 = [_dot(orthogonal_basis[0], self.n - 1)]
        self._circumcenters = [0.5 * (e[0] + e[1])]

        sumx = np.array(e + self._circumcenters[-1])

        dd = _dot(diff, self.n - 1)
        self._circumradii2 = [0.25 * dd]

        self._partitions = [0.5 * np.sqrt(np.array([dd, dd]))]

        norms2 = np.array(volumes2)
        for kk, idx in enumerate(self.idx[:-1][::-1]):
            # Use the orthogonal bases of all sides to get a vector `v` orthogonal to
            # the side, pointing towards the additional point `p0`.
            p0 = self.points[idx]
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
            c = self._circumcenters[-1]
            cr2 = self._circumradii2[-1]

            p0c2 = _dot(p0 - c, self.n - 1 - kk)
            #
            sigma = 0.5 * (p0c2 - cr2) / vv
            lmbda2 = sigma ** 2 * vv

            # circumcenter, squared circumradius
            # <https://math.stackexchange.com/a/4064749/36678>
            #
            self._circumradii2.append(lmbda2[k0] + cr2[k0])
            self._circumcenters.append(
                c[k0] + _multiply(v[k0], sigma[k0], self.n - 2 - kk)
            )

            sumx += self._circumcenters[-1]

            # cell partitions
            # don't use sqrt(lmbda2) here; lmbda can be negative
            lmbda = sigma * np.sqrt(vv)
            vols = self._partitions[-1] * lmbda / (kk + 2)
            self._partitions.append(vols)

        self._volumes = [np.sqrt(v2) for v2 in volumes2]

        # The integral of x,
        #
        #   \\int_V x,
        #
        # over all atomic wedges, i.e., areas cornered by a point, an edge midpoint, and
        # the subsequent circumcenters.
        # The integral of any linear function over a triangle is the average of the
        # values of the function in each of the three corners, times the area of the
        # triangle.
        self._integral_x = _multiply(sumx, self._partitions[-1] / self.n, self.n)

    @property
    def edge_lengths(self):
        if self._volumes is None:
            self._compute_things()
        return self._volumes[0]

    @property
    def facet_areas(self):
        if self.n == 2:
            return np.ones(len(self.facets["points"]))

        if self._volumes is None:
            self._compute_things()

        return self._volumes[-2]

    @property
    def cell_volumes(self):
        if self._volumes is None:
            self._compute_things()
        return self._volumes[-1]

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
        idx = self.idx[1]
        # reshape the last two dimensions into one
        idx = idx.reshape(idx.shape[0], -1)

        # Sort the columns to make it possible for `unique()` to identify individual
        # facets.
        idx = np.sort(idx.T, axis=1)
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
                np.sort(self.cells["points"]), return_inverse=True, return_counts=True
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
        self.cells["facets"] = inv.reshape(self.n, -1).T

        if self.n == 2:
            pass
        elif self.n == 3:
            self.edges = self.facets
            self.cells["edges"] = self.cells["facets"]

            self._facets_cells = None
            self._facets_cells_idx = None
        else:
            assert self.n == 4
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
            self._is_boundary_point[self.idx[1][:, self.is_boundary_facet_local]] = True
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

    def remove_dangling_points(self):
        """Remove all points which aren't part of an array"""
        is_part_of_cell = np.zeros(self.points.shape[0], dtype=bool)
        is_part_of_cell[self.cells["points"].flat] = True

        new_point_idx = np.cumsum(is_part_of_cell) - 1

        self._points = self._points[is_part_of_cell]
        self.cells["points"] = new_point_idx[self.cells["points"]]
        for k in range(len(self.idx)):
            self.idx[k] = new_point_idx[self.idx[k]]

        if self._control_volumes is not None:
            self._control_volumes = self._control_volumes[is_part_of_cell]

        if self._cv_centroids is not None:
            self._cv_centroids = self._cv_centroids[is_part_of_cell]

        if self.edges is not None:
            self.edges["points"] = new_point_idx[self.edges["points"]]

        if self._is_interior_point is not None:
            self._is_interior_point = self._is_interior_point[is_part_of_cell]

        if self._is_boundary_point is not None:
            self._is_boundary_point = self._is_boundary_point[is_part_of_cell]

        if self._is_point_used is not None:
            self._is_point_used = self._is_point_used[is_part_of_cell]

    @property
    def cell_circumcenters(self):
        """Get the center of the circumsphere of each cell."""
        if self._circumcenters is None:
            self._compute_things()
        return self._circumcenters[-1]

    @property
    def cell_circumradius(self):
        """Get the circumradii of all cells"""
        if self._circumradii2 is None:
            self._compute_things()
        return np.sqrt(self._circumradii2[-1])

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

    @property
    def cell_quality(self):
        warnings.warn(
            "Use `q_radius_ratio`. This method will be removed in a future release."
        )
        return self.q_radius_ratio

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
            # updating the boundary data is a lot easier with facets_cells
            if self._facets_cells is None:
                self._compute_facets_cells()

            # Set edge to is_boundary_facet_local=True if it is adjacent to a removed
            # cell.
            facet_ids = self.cells["edges"][~keep].flatten()
            # only consider interior edges
            facet_ids = facet_ids[self.is_interior_facet[facet_ids]]
            idx = self.facets_cells_idx[facet_ids]
            cell_id = self.facets_cells["interior"][1:3, idx].T
            local_edge_id = self.facets_cells["interior"][3:5, idx].T
            self._is_boundary_facet_local[local_edge_id, cell_id] = True
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
                    # edge id
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
                    # local edge id
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

            num_edges_old = len(self.edges["points"])
            adjacent_edges, counts = np.unique(
                self.cells["edges"][~keep].flat, return_counts=True
            )
            # remove edge entirely either if 2 adjacent cells are removed or if it is a
            # boundary edge and 1 adjacent cells are removed
            is_facet_removed = (counts == 2) | (
                (counts == 1) & self._is_boundary_facet[adjacent_edges]
            )

            # set the new boundary edges
            self._is_boundary_facet[adjacent_edges[~is_facet_removed]] = True
            # Now actually remove the edges. This includes a reindexing.
            assert self._is_boundary_facet is not None
            keep_edges = np.ones(len(self._is_boundary_facet), dtype=bool)
            keep_edges[adjacent_edges[is_facet_removed]] = False

            # make sure there is only edges["points"], not edges["cells"] etc.
            assert self.edges is not None
            assert len(self.edges) == 1
            self.edges["points"] = self.edges["points"][keep_edges]
            self._is_boundary_facet = self._is_boundary_facet[keep_edges]

            # update edge and cell indices
            self.cells["edges"] = self.cells["edges"][keep]
            new_index_edges = np.arange(num_edges_old) - np.cumsum(~keep_edges)
            self.cells["edges"] = new_index_edges[self.cells["edges"]]
            num_cells_old = len(self.cells["points"])
            new_index_cells = np.arange(num_cells_old) - np.cumsum(~keep)

            # this takes fairly long
            ec = self._facets_cells
            ec["boundary"][0] = new_index_edges[ec["boundary"][0]]
            ec["boundary"][1] = new_index_cells[ec["boundary"][1]]
            ec["interior"][0] = new_index_edges[ec["interior"][0]]
            ec["interior"][1:3] = new_index_cells[ec["interior"][1:3]]

            # simply set those to None; their reset is cheap
            self._facets_cells_idx = None
            self._boundary_facets = None
            self._interior_facets = None

        self.cells["points"] = self.cells["points"][keep]

        for k in range(len(self.idx)):
            self.idx[k] = self.idx[k][..., keep]

        if self._volumes is not None:
            for k in range(len(self._volumes)):
                self._volumes[k] = self._volumes[k][..., keep]

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

        if self._circumcenters is not None:
            for k in range(len(self._circumcenters)):
                self._circumcenters[k] = self._circumcenters[k][..., keep, :]

        if self._partitions is not None:
            for k in range(len(self._partitions)):
                self._partitions[k] = self._partitions[k][..., keep]

        if self._signed_cell_volumes is not None:
            self._signed_cell_volumes = self._signed_cell_volumes[keep]

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

    def remove_duplicate_cells(self):
        sorted_cells = np.sort(self.cells["points"])
        _, inv, cts = npx.unique_rows(
            sorted_cells, return_inverse=True, return_counts=True
        )

        remove = np.zeros(len(self.cells["points"]), dtype=bool)
        for k in np.where(cts > 1)[0]:
            rem = inv == k
            # don't remove first occurence
            first_idx = np.where(rem)[0][0]
            rem[first_idx] = False
            remove |= rem

        return self.remove_cells(remove)

    @property
    def cell_partitions(self):
        """Each simplex can be subdivided into parts that a closest to each corner.
        This method gives those parts, like ce_ratios associated with each edge.
        """
        # if self._partitions is None:
        #     # self._compute_things()
        #     # The volume of the pyramid is
        #     #
        #     # edge_length ** 2 / 2 * covolume / edgelength / (n-1)
        #     # = edgelength / 2 * covolume / (n - 1)
        #     # TODO keep this for computing ce_ratios
        #     self._cell_partitions = self.ei_dot_ei / 2 * self.ce_ratios / (self.n - 1)
        # return self._cell_partitions
        if self._partitions is None:
            self._compute_things()
        return self._partitions[-1]

    def get_control_volumes(self, idx=slice(None)):
        """The control volumes around each vertex. Optionally disregard the
        contributions from particular cells. This is useful, for example, for
        temporarily disregarding flat cells on the boundary when performing Lloyd mesh
        optimization.
        """
        if self._control_volumes is None or np.any(idx != self._cv_cell_mask):
            # Sum up the contributions according to how self.idx is constructed.
            # roll = np.array([np.roll(np.arange(kk + 3), -i) for i in range(1, kk + 3)])
            # vols = npx.sum_at(vols, roll, kk + 3)
            # v = self.cell_partitions[..., idx]

            # TODO this can be improved by first summing up all components per cell
            self._control_volumes = npx.sum_at(
                self.cell_partitions[:, idx],
                self.idx[-1][:, idx],
                len(self.points),
            )

            # # Explicitly sum up contributions per cell first. Makes sum_at faster.
            # if self.n == 2:
            #     v = np.array([v, v])
            # elif self.n == 3:
            #     v = np.array([v[1] + v[2], v[2] + v[0], v[0] + v[1]])
            # else:
            #     assert self.n == 4
            #     # For every point k (range(4)), check for which edges k appears in
            #     # local_idx, and sum() up the v's from there.
            #     v = np.array(
            #         [
            #             v[0, 2] + v[1, 1] + v[2, 3] + v[0, 1] + v[1, 3] + v[2, 2],
            #             v[0, 3] + v[1, 2] + v[2, 0] + v[0, 2] + v[1, 0] + v[2, 3],
            #             v[0, 0] + v[1, 3] + v[2, 1] + v[0, 3] + v[1, 1] + v[2, 0],
            #             v[0, 1] + v[1, 0] + v[2, 2] + v[0, 0] + v[1, 2] + v[2, 1],
            #         ]
            #     )

            # # sum all the vals into self._control_volumes at ids
            # self._control_volumes = npx.sum_at(
            #     v,
            #     self.cells["points"][~cell_mask].T,
            #     len(self.points),
            # )

            self._cv_cell_mask = idx
        return self._control_volumes

    @property
    def control_volumes(self):
        """The control volumes around each vertex."""
        if self._control_volumes is None:
            return self.get_control_volumes()

        return self._control_volumes

    @property
    def ce_ratios(self):
        """The covolume-edgelength ratios."""
        if self._ce_ratios is None:
            if self.n == 2:
                self._ce_ratios = 1.0 / np.sqrt(self.ei_dot_ei)
            elif self.n == 3:
                self._ce_ratios = compute_ce_ratios(self.ei_dot_ej, self.cell_volumes)
            else:
                assert self.n == 4
                self._ce_ratios = self._compute_ce_ratios_geometric()
        return self._ce_ratios

    @property
    def ce_ratios_per_interior_facet(self):
        if self._interior_ce_ratios is None:
            if "edges" not in self.cells:
                self.create_facets()

            ce_ratios = npx.sum_at(
                self.ce_ratios.T,
                self.cells["edges"],
                self.edges["points"].shape[0],
            )
            self._interior_ce_ratios = ce_ratios[self.is_interior_facet]

        return self._interior_ce_ratios

    # Question:
    # We're looking for an explicit expression for the algebraic c/e ratios. Might it be
    # that, analogous to the triangle dot product, the "triple product" has something to
    # do with it?
    # "triple product": Project one edge onto the plane spanned by the two others.
    #
    # def _compute_ce_ratios_algebraic(self):
    #     # Precompute edges.
    #     half_edges = (
    #         self.points[self.idx[-1][1]]
    #         - self.points[self.idx[-1][0]]
    #     )

    #     # Build the equation system:
    #     # The equation
    #     #
    #     # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>
    #     #
    #     # has to hold for all vectors u in the plane spanned by the edges,
    #     # particularly by the edges themselves.
    #     # A = np.empty(3, 4, half_edges.shape[2], 3, 3)
    #     A = np.einsum("j...k,l...k->jl...", half_edges, half_edges)
    #     A = A ** 2

    #     # Compute the RHS  cell_volume * <edge, edge>.
    #     # The dot product <edge, edge> is also on the diagonals of A (before squaring),
    #     # but simply computing it again is cheaper than extracting it from A.
    #     edge_dot_edge = np.einsum("...i,...j->...", half_edges, half_edges)
    #     # TODO cell_volumes
    #     self.cell_volumes = np.random.rand(2951)
    #     rhs = edge_dot_edge * self.cell_volumes

    #     # Solve all k-by-k systems at once ("broadcast"). (`k` is the number of edges
    #     # per simplex here.)
    #     # If the matrix A is (close to) singular if and only if the cell is (close to
    #     # being) degenerate. Hence, it has volume 0, and so all the edge coefficients
    #     # are 0, too. Hence, do nothing.
    #     ce_ratios = np.linalg.solve(A, rhs)

    #     return ce_ratios

    def _compute_ce_ratios_geometric(self):
        # For triangles, the covolume/edgelength ratios are
        #
        #   [1]   ce_ratios = -<ei, ej> / cell_volume / 4;
        #
        # for tetrahedra, is somewhat more tricky. This is the reference expression:
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
        # for the face [1, 2, 3] (with edges [3, 4, 5]), where points and edges are
        # ordered like
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
        # This is not a too obvious extension of -<ei, ej> in [1]. However, consider the
        # fact that this contains all pairwise dot-products of edges not part of the
        # respective face (<e0, e1>, <e1, e2>, <e2, e0>), each of them weighted with
        # dot-products of edges in the respective face.
        #
        # Note that, to retrieve the covolume-edge ratio, one divides by
        #
        #       alpha = (
        #           + ei_dot_ej[3, 5] * ei_dot_ej[5, 4]
        #           + ei_dot_ej[3, 5] * ei_dot_ej[3, 4]
        #           + ei_dot_ej[3, 4] * ei_dot_ej[4, 5]
        #           )
        #
        # (which is the square of the face area). It's funny that there should be no
        # further simplification in zeta/alpha, but nothing has been found here yet.
        #

        # From base.py, but spelled out here since we can avoid one sqrt when computing
        # the c/e ratios for the faces.
        alpha = (
            +self.ei_dot_ej[2] * self.ei_dot_ej[0]
            + self.ei_dot_ej[0] * self.ei_dot_ej[1]
            + self.ei_dot_ej[1] * self.ei_dot_ej[2]
        )
        # face_ce_ratios = -self.ei_dot_ej * 0.25 / face_areas[None]
        face_ce_ratios_div_face_areas = -self.ei_dot_ej / alpha

        ee = self.ei_dot_ej
        zeta = (
            -ee[2, [1, 2, 3, 0]] * ee[1] * ee[2]
            - ee[1, [2, 3, 0, 1]] * ee[2] * ee[0]
            - ee[0, [3, 0, 1, 2]] * ee[0] * ee[1]
            + ee[0] * ee[1] * ee[2]
        )

        #
        # self.circumcenter_face_distances =
        #    zeta / (24.0 * face_areas) / self.cell_volumes[None]
        # ce_ratios = \
        #     0.5 * face_ce_ratios * self.circumcenter_face_distances[None],
        #
        # so
        ce_ratios = (
            zeta / 48.0 * face_ce_ratios_div_face_areas / self.cell_volumes[None]
        )

        # Distances of the cell circumcenter to the faces.
        face_areas = 0.5 * np.sqrt(alpha)
        self.circumcenter_face_distances = (
            zeta / (24.0 * face_areas) / self.cell_volumes[None]
        )

        return ce_ratios

    def num_delaunay_violations(self):
        """Number of edges where the Delaunay condition is violated."""
        # Delaunay violations are present exactly on the interior edges where the
        # ce_ratio is negative. Count those.
        if self.n == 3:
            return np.sum(self.ce_ratios_per_interior_facet < 0.0)

        assert self.n == 4

        # Delaunay violations are present exactly on the interior faces where the sum of
        # the signed distances between face circumcenter and tetrahedron circumcenter is
        # negative.
        if self.circumcenter_face_distances is None:
            self._compute_ce_ratios_geometric()
            # self._compute_ce_ratios_algebraic()

        if "facets" not in self.cells:
            self.create_facets()

        num_facets = self.facets["points"].shape[0]
        sums = npx.sum_at(
            self.circumcenter_face_distances,
            self.cells["facets"].T,
            num_facets,
        )
        return np.sum(sums[self.is_interior_facet] < 0.0)

    def get_control_volume_centroids(self, idx=slice(None)):
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
        assert self.n == 3

        if self._cv_centroids is None or np.any(idx != self._cvc_cell_mask):
            if self._integral_x is None:
                self._compute_things()

            # TODO this can be improved by first summing up all components per cell
            integral_p = npx.sum_at(
                self._integral_x[..., idx, :],
                self.idx[-1][:, idx],
                len(self.points),
            )

            # Divide by the control volume
            cv = self.get_control_volumes(idx)
            self._cv_centroids = (integral_p.T / cv).T

            # _, v = self._compute_integral_x()
            # v = v[:, :, idx, :]

            # # Again, make use of the fact that edge k is opposite of point k in every
            # # cell. Adding the arrays first makes the work for sum_at lighter.
            # ids = self.cells["points"][idx].T
            # vals = np.array([v[1, 1] + v[0, 2], v[1, 2] + v[0, 0], v[1, 0] + v[0, 1]])

            # # add it all up
            # self._cv_centroids = npx.sum_at(vals, ids, self.points.shape[0])

            # # Divide by the control volume
            # cv = self.get_control_volumes(idx=idx)
            # # self._cv_centroids /= np.where(cv > 0.0, cv, 1.0)
            # self._cv_centroids = (self._cv_centroids.T / cv).T
            self._cvc_cell_mask = idx
            assert np.all(idx == self._cv_cell_mask)

        return self._cv_centroids

    @property
    def control_volume_centroids(self):
        return self.get_control_volume_centroids()

    # def _compute_integral_x(self):
    #     # Computes the integral of x,
    #     #
    #     #   \\int_V x,
    #     #
    #     # over all atomic "triangles", i.e., areas cornered by a point, an edge
    #     # midpoint, and a circumcenter.
    #     #
    #     # The integral of any linear function over a triangle is the average of the
    #     # values of the function in each of the three corners, times the area of the
    #     # triangle.
    #     assert self.n == 3
    #     right_triangle_vols = self.cell_partitions

    #     point_edges = self.idx[-1]

    #     corner = self.points[point_edges]
    #     edge_midpoints = 0.5 * (corner[0] + corner[1])
    #     cc = self.cell_circumcenters

    #     average = (corner + edge_midpoints[None] + cc[None, None]) / 3.0

    #     contribs = right_triangle_vols[None, :, :, None] * average
    #     return point_edges, contribs
