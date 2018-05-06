# -*- coding: utf-8 -*-
#
import math

import numpy


def get_signed_tri_areas(cells, pts):
    '''Signed area of a triangle in 2D.
    '''
    assert cells.shape[1] == 3
    assert pts.shape[1] == 2
    # http://mathworld.wolfram.com/TriangleArea.html
    p = pts[cells].T
    p = numpy.moveaxis(p, 0, 1)
    return (
        - p[1][0]*p[0][1] + p[2][0]*p[0][1] + p[0][0]*p[1][1]
        - p[2][0]*p[1][1] - p[0][0]*p[2][1] + p[1][0]*p[2][1]
        ) / 2


def get_signed_simplex_volumes(cells, pts):
    '''Signed volume of a simplex in nD. Note that signing only makes sense for
    n-simplices in R^n.
    '''
    n = pts.shape[1]
    assert cells.shape[1] == n+1

    p = pts[cells]
    p = numpy.concatenate([p, numpy.ones(list(p.shape[:2]) + [1])], axis=-1)
    return numpy.linalg.det(p) / math.factorial(n)
