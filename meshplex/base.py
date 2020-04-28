import numpy

import meshio

from .exceptions import MeshplexError

__all__ = []


def compute_tri_areas(ei_dot_ej):
    vol2 = 0.25 * (
        ei_dot_ej[2] * ei_dot_ej[0]
        + ei_dot_ej[0] * ei_dot_ej[1]
        + ei_dot_ej[1] * ei_dot_ej[2]
    )
    # vol2 is the squared volume, but can be slightly negative if it comes to round-off
    # errors. Corrrect those.
    assert numpy.all(vol2 > -1.0e-14)
    vol2[vol2 < 0] = 0.0
    return numpy.sqrt(vol2)


def compute_ce_ratios(ei_dot_ej, tri_areas):
    """Given triangles (specified by their mutual edge projections and the area), this
    routine will return ratio of the signed distances of the triangle circumcenters to
    the edge midpoints and the edge lengths.
    """
    # The input argument are the dot products
    #
    #   <e1, e2>
    #   <e2, e0>
    #   <e0, e1>
    #
    # of the edges
    #
    #   e0: x1->x2,
    #   e1: x2->x0,
    #   e2: x0->x1.
    #
    # Note that edge e_i is opposite of node i and the edges add up to 0.
    # (Those quantities can be shared between numerous methods, so share them.)
    #

    # There are multiple ways of deriving a closed form for the
    # covolume-edgelength ratios.
    #
    #   * The covolume-edge ratios for the edges of each cell is the solution
    #     of the equation system
    #
    #       |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>,
    #
    #     where alpha_i are the covolume contributions for the edges. This
    #     equation system holds for all vectors u in the plane spanned by the
    #     edges, particularly by the edges themselves.
    #
    #     For triangles, the exact solution of the system is
    #
    #       x_1 = <e_2, e_3> / <e1 x e2, e1 x e3> * |simplex|;
    #
    #     see <https://math.stackexchange.com/a/1855380/36678>.
    #
    #   * In trilinear coordinates
    #     <https://en.wikipedia.org/wiki/Trilinear_coordinates>, the
    #     circumcenter is
    #
    #         cos(alpha0) : cos(alpha1) : cos(alpha2)
    #
    #     where the alpha_i are the angles opposite of the respective edge.
    #     With
    #
    #       cos(alpha0) = <e1, e2> / ||e1|| / ||e2||
    #
    #     and the conversion formula to Cartesian coordinates, ones gets the
    #     expression
    #
    #         ce0 = <e1, e2> * 0.5 / sqrt(alpha)
    #
    #     with
    #
    #         alpha = <e1, e2> * <e0, e1>
    #               + <e2, e0> * <e1, e2>
    #               + <e0, e1> * <e2, e0>.
    #
    # Note that some simplifications are achieved by virtue of
    #
    #   e1 + e2 + e3 = 0.
    #
    if not numpy.all(tri_areas > 0.0):
        raise MeshplexError("Degenerate cell.")

    return -ei_dot_ej * 0.25 / tri_areas[None]


def compute_triangle_circumcenters(X, ei_dot_ei, ei_dot_ej):
    """Computes the circumcenters of all given triangles.
    """
    # The input argument are the dot products
    #
    #   <e1, e2>
    #   <e2, e0>
    #   <e0, e1>
    #
    # of the edges
    #
    #   e0: x1->x2,
    #   e1: x2->x0,
    #   e2: x0->x1.
    #
    # Note that edge e_i is opposite of node i and the edges add up to 0.

    # The trilinear coordinates of the circumcenter are
    #
    #   cos(alpha0) : cos(alpha1) : cos(alpha2)
    #
    # where alpha_k is the angle at point k, opposite of edge k. The Cartesian
    # coordinates are (see
    # <https://en.wikipedia.org/wiki/Trilinear_coordinates#Between_Cartesian_and_trilinear_coordinates>)
    #
    #     C = sum_i ||e_i|| * cos(alpha_i)/beta * P_i
    #
    # with
    #
    #     beta = sum ||e_i||*cos(alpha_i)
    #
    # Incidentally, the cosines are
    #
    #    cos(alpha0) = <e1, e2> / ||e1|| / ||e2||,
    #
    # so in total
    #
    #    C = <e_0, e0> <e1, e2> / sum_i (<e_i, e_i> <e{i+1}, e{i+2}>) P0
    #      + ... P1
    #      + ... P2.
    #
    # An even nicer formula is given on
    # <https://en.wikipedia.org/wiki/Circumscribed_circle#Barycentric_coordinates>: The
    # barycentric coordinates of the circumcenter are
    #
    #   a^2 (b^2 + c^2 - a^2) : b^2 (c^2 + a^2 - b^2) : c^2 (a^2 + b^2 - c^2).
    #
    # This is only using the squared edge lengths, too!
    #
    alpha = ei_dot_ei * ei_dot_ej
    alpha_sum = alpha[0] + alpha[1] + alpha[2]
    beta = alpha / alpha_sum[None]
    a = X * beta[..., None]
    cc = a[0] + a[1] + a[2]

    # alpha = numpy.array([
    #     ei_dot_ei[0] * (ei_dot_ei[1] + ei_dot_ei[2] - ei_dot_ei[0]),
    #     ei_dot_ei[1] * (ei_dot_ei[2] + ei_dot_ei[0] - ei_dot_ei[1]),
    #     ei_dot_ei[2] * (ei_dot_ei[0] + ei_dot_ei[1] - ei_dot_ei[2]),
    # ])
    # alpha /= numpy.sum(alpha, axis=0)
    # cc = (X[0].T * alpha[0] + X[1].T * alpha[1] + X[2].T * alpha[2]).T
    return cc


class _base_mesh:
    def __init__(self, nodes, cells_nodes):
        self.node_coords = nodes
        self._edge_lengths = None

    def write(self, filename, point_data=None, cell_data=None, field_data=None):
        if self.node_coords.shape[1] == 2:
            n = len(self.node_coords)
            a = numpy.ascontiguousarray(
                numpy.column_stack([self.node_coords, numpy.zeros(n)])
            )
        else:
            a = self.node_coords

        if self.cells["nodes"].shape[1] == 3:
            cell_type = "triangle"
        else:
            assert (
                self.cells["nodes"].shape[1] == 4
            ), "Only triangles/tetrahedra supported"
            cell_type = "tetra"

        meshio.write_points_cells(
            filename,
            a,
            {cell_type: self.cells["nodes"]},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    @property
    def edge_lengths(self):
        if self._edge_lengths is None:
            self._edge_lengths = numpy.sqrt(self.ei_dot_ei)
        return self._edge_lengths

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]["vertices"]

    def get_edge_mask(self, subdomain=None):
        """Get faces which are fully in subdomain.
        """
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its nodes are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over the first index
        is_inside = numpy.all(is_in, axis=tuple(range(1)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_edge

        return is_inside

    def get_face_mask(self, subdomain):
        """Get faces which are fully in subdomain.
        """
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its nodes are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last two (face_ids, cell_ids).
        n = len(is_in.shape)
        is_inside = numpy.all(is_in, axis=tuple(range(n - 2)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet

        return is_inside

    def get_cell_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain.is_boundary_only:
            # There are no boundary cells
            return numpy.array([])

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last one (cell_ids).
        n = len(is_in.shape)
        return numpy.all(is_in, axis=tuple(range(n - 1)))

    def _mark_vertices(self, subdomain):
        """Mark faces/edges which are fully in subdomain.
        """
        if subdomain is None:
            is_inside = numpy.ones(len(self.node_coords), dtype=bool)
        else:
            is_inside = subdomain.is_inside(self.node_coords.T).T

            if subdomain.is_boundary_only:
                # Filter boundary
                self.mark_boundary()
                is_inside = is_inside & self.is_boundary_node

        self.subdomains[subdomain] = {"vertices": is_inside}
        return
