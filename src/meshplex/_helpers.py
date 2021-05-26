import math

import numpy as np


def get_signed_simplex_volumes(cells, pts):
    """Signed volume of a simplex in nD. Note that signing only makes sense for
    n-simplices in R^n.
    """
    n = pts.shape[1]
    assert cells.shape[1] == n + 1

    p = pts[cells]
    p = np.concatenate([p, np.ones(list(p.shape[:2]) + [1])], axis=-1)
    return np.linalg.det(p) / math.factorial(n)


def grp_start_len(a):
    """Given a sorted 1D input array `a`, e.g., [0 0, 1, 2, 3, 4, 4, 4], this routine
    returns the indices where the blocks of equal integers start and how long the blocks
    are.
    """
    # https://stackoverflow.com/a/50394587/353337
    m = np.concatenate([[True], a[:-1] != a[1:], [True]])
    idx = np.flatnonzero(m)
    return idx[:-1], np.diff(idx)


def _dot(a, n):
    """Dot product, preserve the leading n dimensions."""
    # einsum is faster if the tail survives, e.g., ijk,ijk->jk.
    # <https://gist.github.com/nschloe/8bc015cc1a9e5c56374945ddd711df7b>
    # TODO reorganize the data?
    assert n <= len(a.shape)
    # Would use -1 as second argument, but <https://github.com/numpy/numpy/issues/18519>
    b = a.reshape(*a.shape[:n], np.prod(a.shape[n:]).astype(int))
    return np.einsum("...i,...i->...", b, b)


def _multiply(a, b, n):
    """Multiply the along the first n dimensions of a and b. For example,
    a.shape == (5,6,3), b.shape == (5, 6), n = 2, will return an array c of a.shape with
    c[i,j,k] = a[i,j,k] * b[i,j].
    """
    aa = a.reshape(np.prod(a.shape[:n]), *a.shape[n:])
    bb = b.reshape(np.prod(b.shape[:n]), *b.shape[n:])
    cc = (aa.T * bb).T
    c = cc.reshape(*a.shape)
    return c
