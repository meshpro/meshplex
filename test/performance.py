"""
"""
import random

import numpy as np
import perfplot
from scipy.spatial import Delaunay

import meshplex


def setup(n):
    radius = 1.0
    k = np.arange(n)
    boundary_pts = radius * np.column_stack(
        [np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)]
    )

    # Compute the number of interior points such that all triangles can be somewhat
    # equilateral.
    edge_length = 2 * np.pi * radius / n
    domain_area = np.pi - n * (radius ** 2 / 2 * (edge_length - np.sin(edge_length)))
    cell_area = np.sqrt(3) / 4 * edge_length ** 2
    target_num_cells = domain_area / cell_area
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    # <=>
    # num_interior_points ~= 0.5 * (num_cells + num_boundary_edges) + 1 - num_boundary_points
    m = int(0.5 * (target_num_cells + n) + 1 - n)

    # generate random points in circle;
    # <http://mathworld.wolfram.com/DiskPointPicking.html>
    for seed in range(0, 255):
        np.random.seed(seed)
        r = np.random.rand(m)
        alpha = 2 * np.pi * np.random.rand(m)

        interior_pts = np.column_stack(
            [np.sqrt(r) * np.cos(alpha), np.sqrt(r) * np.sin(alpha)]
        )

        pts = np.concatenate([boundary_pts, interior_pts])

        tri = Delaunay(pts)

        # Make sure there are exactly `n` boundary points
        mesh0 = meshplex.MeshTri(pts, tri.simplices)
        mesh1 = meshplex.MeshTri(pts, tri.simplices)
        if np.sum(mesh0.is_boundary_point) == n:
            break

    mesh0.create_edges()
    mesh1.create_edges()

    num_interior_edges = np.sum(mesh0.is_interior_edge)
    idx = random.sample(range(num_interior_edges), n // 10)
    print(num_interior_edges, len(idx), len(idx) / num_interior_edges)

    # # move interior points a little bit such that we have edges to flip
    # max_step = np.min(mesh.cell_inradius) / 2
    # mesh.points = mesh.points + max_step * np.random.rand(*mesh.points.shape)
    # print(mesh.num_delaunay_violations)
    return mesh0, mesh1, idx


def flip_old(data):
    mesh0, mesh1, idx = data
    mesh0.flip_interior_edges_old(idx)


def flip_new(data):
    mesh0, mesh1, idx = data
    mesh1.flip_interior_edges(idx)


perfplot.show(
    setup=setup,
    kernels=[flip_old, flip_new],
    n_range=[2 ** k for k in range(5, 13)],
    equality_check=None,
    # set target time to 0 to avoid more than one repetition
    target_time_per_measurement=0.0,
)
