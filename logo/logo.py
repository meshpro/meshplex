#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pygmsh as pg
import voropy


def generate():
    geom = pg.Geometry()

    lcar = 0.1
    circle = geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            lcar,
            num_sections=4,
            compound=True,
            make_surface=False
            )

    coords_v = numpy.array([
        [+0.2, -0.6, 0.0],
        [+0.65, +0.5, 0.0],
        [+0.25, +0.5, 0.0],
        [+0.0, -0.15, 0.0],
        [-0.25, +0.5, 0.0],
        [-0.65, +0.5, 0.0],
        [-0.2, -0.6, 0.0],
        ])
    hole = geom.add_polygon(coords_v, lcar, make_surface=False)

    geom.add_plane_surface(circle.line_loop, holes=[hole])

    return geom


if __name__ == '__main__':
    X, cells, _, _, _ = pg.generate_mesh(generate())
    # single out nodes that are actually used
    uvertices, uidx = numpy.unique(cells['triangle'], return_inverse=True)
    cells = uidx.reshape(cells['triangle'].shape)
    X = X[uvertices]

    mesh = voropy.smoothing.lloyd(X, cells, 1.0e-10)
    mesh.show(
            show_centroids=False,
            mesh_color=[0.8, 0.8, 0.8],
            comesh_color='k',
            boundary_edge_color='k'
            )
