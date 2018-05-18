# -*- coding: utf-8 -*-
#
import math

import numpy


def get_signed_simplex_volumes(cells, pts):
    '''Signed volume of a simplex in nD. Note that signing only makes sense for
    n-simplices in R^n.
    '''
    n = pts.shape[1]
    assert cells.shape[1] == n+1

    p = pts[cells]
    p = numpy.concatenate([p, numpy.ones(list(p.shape[:2]) + [1])], axis=-1)
    return numpy.linalg.det(p) / math.factorial(n)


def grp_start_len(a):
    '''Given a sorted 1D input array `a`, e.g., [0 0, 1, 2, 3, 4, 4, 4], this
    routine returns the indices where the blocks of equal integers start and
    how long the blocks are.
    '''
    # https://stackoverflow.com/a/50394587/353337
    m = numpy.concatenate([[True], a[:-1] != a[1:], [True]])
    idx = numpy.flatnonzero(m)
    return idx[:-1], numpy.diff(idx)
