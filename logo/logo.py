#! /usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
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
    points, cells = pg.generate_mesh(generate())
    mesh = voropy.mesh_tri.MeshTri(points, cells['triangle'])
    mesh = voropy.mesh_tri.lloyd_smoothing(mesh, 1.0e-10)
    mesh.show(
            show_centroids=False,
            mesh_color=[0.8,0.8,0.8],
            comesh_color='k',
            boundary_edge_color='k'
            )
    plt.show()
