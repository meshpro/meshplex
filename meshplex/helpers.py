import math

import numpy

from .exceptions import MeshplexError


def get_signed_simplex_volumes(cells, pts):
    """Signed volume of a simplex in nD. Note that signing only makes sense for
    n-simplices in R^n.
    """
    n = pts.shape[1]
    assert cells.shape[1] == n + 1

    p = pts[cells]
    p = numpy.concatenate([p, numpy.ones(list(p.shape[:2]) + [1])], axis=-1)
    return numpy.linalg.det(p) / math.factorial(n)


def grp_start_len(a):
    """Given a sorted 1D input array `a`, e.g., [0 0, 1, 2, 3, 4, 4, 4], this routine
    returns the indices where the blocks of equal integers start and how long the blocks
    are.
    """
    # https://stackoverflow.com/a/50394587/353337
    m = numpy.concatenate([[True], a[:-1] != a[1:], [True]])
    idx = numpy.flatnonzero(m)
    return idx[:-1], numpy.diff(idx)


def unique_rows(a):
    # The numpy alternative `numpy.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    b = numpy.ascontiguousarray(a).view(
        numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
    )
    a_unique, inv, cts = numpy.unique(b, return_inverse=True, return_counts=True)
    a_unique = a_unique.view(a.dtype).reshape(-1, a.shape[1])
    return a_unique, inv, cts


def compute_tri_areas(ei_dot_ej):
    vol2 = 0.25 * (
        ei_dot_ej[2] * ei_dot_ej[0]
        + ei_dot_ej[0] * ei_dot_ej[1]
        + ei_dot_ej[1] * ei_dot_ej[2]
    )
    # vol2 is the squared volume, but can be slightly negative if it comes to round-off
    # errors. Correct those.
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
    if numpy.any(tri_areas <= 0.0):
        raise MeshplexError("Degenerate cells.")

    return -ei_dot_ej * 0.25 / tri_areas[None]


def compute_triangle_circumcenters(X, ei_dot_ei, ei_dot_ej):
    """Computes the circumcenters of all given triangles."""
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
    #     C = sum_i ||e_i|| * cos(alpha_i) / beta * P_i
    #
    # with
    #
    #     beta = sum ||e_i|| * cos(alpha_i)
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
    alpha = ei_dot_ei * ei_dot_ej
    alpha_sum = alpha[0] + alpha[1] + alpha[2]
    beta = alpha / alpha_sum[None]
    a = X * beta[..., None]
    cc = a[0] + a[1] + a[2]

    # An even nicer formula is given on
    # <https://en.wikipedia.org/wiki/Circumscribed_circle#Barycentric_coordinates>: The
    # barycentric coordinates of the circumcenter are
    #
    #   a^2 (b^2 + c^2 - a^2) : b^2 (c^2 + a^2 - b^2) : c^2 (a^2 + b^2 - c^2).
    #
    # This is only using the squared edge lengths, too!
    # TODO make this happen, perhaps with something like
    #    ei_dot_ei * (numpy.sum(ei_dot_ei, axis=0) - 2 * ei_dot_ei)
    #
    # alpha = numpy.array([
    #     ei_dot_ei[0] * (ei_dot_ei[1] + ei_dot_ei[2] - ei_dot_ei[0]),
    #     ei_dot_ei[1] * (ei_dot_ei[2] + ei_dot_ei[0] - ei_dot_ei[1]),
    #     ei_dot_ei[2] * (ei_dot_ei[0] + ei_dot_ei[1] - ei_dot_ei[2]),
    # ])
    # alpha /= numpy.sum(alpha, axis=0)
    # cc = (X[0].T * alpha[0] + X[1].T * alpha[1] + X[2].T * alpha[2]).T
    return cc
