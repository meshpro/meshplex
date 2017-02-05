#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pygmsh as pg


def generate():
    geom = pg.Geometry()

    lcar = 0.1
    circle = geom.add_circle(
            [0.0, 0.0, 0.0],
            1.0,
            lcar,
            num_sections=4,
            compound=True
            )

    ll = geom.add_line_loop(circle)

    coords_v = np.array([
        [+0.2, -0.6, 0.0],
        [+0.65, +0.5, 0.0],
        [+0.25, +0.5, 0.0],
        [+0.0, -0.15, 0.0],
        [-0.25, +0.5, 0.0],
        [-0.65, +0.5, 0.0],
        [-0.2, -0.6, 0.0],
        ])
    hole = geom.add_polygon_loop(coords_v, lcar)

    geom.add_plane_surface([ll, hole])

    return geom


if __name__ == '__main__':
    import meshio
    points, cells = pg.generate_mesh(generate())
    meshio.write('logo.vtu', points, cells)
