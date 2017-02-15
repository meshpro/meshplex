# -*- coding: utf-8 -*-
#
import fetch_data
import voropy

import numpy


def test_pacman_lloyd():
    filename = fetch_data.download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    mesh, _, _, _ = voropy.read(filename, flat_cell_correction='boundary')

    mesh = voropy.mesh_tri.lloyd_smoothing(
            mesh,
            1.0e-2,
            fcc_type='boundary',
            flip_frequency=1,
            verbose=False
            )

    # Test if we're dealing with the mesh we expect.
    nc = mesh.node_coords.flatten()
    norm1 = numpy.linalg.norm(nc, ord=1)
    norm2 = numpy.linalg.norm(nc, ord=2)
    normi = numpy.linalg.norm(nc, ord=numpy.inf)

    tol = 1.0e-12
    assert abs(norm1 - 1939.1198108068188) < tol
    assert abs(norm2 - 75.949652079323229) < tol
    assert abs(normi - 5.0) < tol

    assert mesh.num_delaunay_violations() == 0

    return
