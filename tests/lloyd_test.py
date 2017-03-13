# -*- coding: utf-8 -*-
#
from helpers import download_mesh
import meshio
import voropy

import numpy


def test_pacman_lloyd():
    filename = download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    X, cells, _, _, _ = meshio.read(filename)

    mesh = voropy.smoothing.lloyd(
            X, cells['triangle'],
            1.0e-2,
            max_steps=1000,
            fcc_type='boundary',
            flip_frequency=1,
            verbose=False
            # output_filetype='png'
            )

    # Test if we're dealing with the mesh we expect.
    nc = mesh.node_coords.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    # assert abs(norm1 - 1944.49523751269) < tol
    # assert abs(norm2 - 76.097893244864181) < tol
    assert abs(norm1 - 1939.1198108068188) < tol
    assert abs(norm2 - 75.949652079323229) < tol
    assert abs(normi - 5.0) < tol

    assert mesh.num_delaunay_violations() == 0

    return
