"""
"""
import perfplot
import numpy
from scipy.spatial import Delaunay

import meshplex


def setup(n):
    radius = 1.0
    k = numpy.arange(n)
    boundary_pts = radius * numpy.column_stack(
        [numpy.cos(2 * numpy.pi * k / n), numpy.sin(2 * numpy.pi * k / n)]
    )

    # Compute the number of interior points such that all triangles can be somewhat
    # equilateral.
    edge_length = 2 * numpy.pi * radius / n
    domain_area = numpy.pi - n * (
        radius ** 2 / 2 * (edge_length - numpy.sin(edge_length))
    )
    cell_area = numpy.sqrt(3) / 4 * edge_length ** 2
    target_num_cells = domain_area / cell_area
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    # <=>
    # num_interior_points ~= 0.5 * (num_cells + num_boundary_edges) + 1 - num_boundary_points
    m = int(0.5 * (target_num_cells + n) + 1 - n)

    # generate random points in circle;
    # <http://mathworld.wolfram.com/DiskPointPicking.html>
    for seed in range(0, 255):
        numpy.random.seed(seed)
        r = numpy.random.rand(m)
        alpha = 2 * numpy.pi * numpy.random.rand(m)

        interior_pts = numpy.column_stack(
            [numpy.sqrt(r) * numpy.cos(alpha), numpy.sqrt(r) * numpy.sin(alpha)]
        )

        pts = numpy.concatenate([boundary_pts, interior_pts])

        tri = Delaunay(pts)

        # Make sure there are exactly `n` boundary points
        mesh = meshplex.MeshTri(pts, tri.simplices)
        if numpy.sum(mesh.is_boundary_point) == n:
            break

    # move interior points a little bit such that we have edges to flip
    max_step = numpy.min(mesh.cell_inradius) / 2
    mesh.points = mesh.points + max_step * numpy.random.rand(*mesh.points.shape)
    print(mesh.num_delaunay_violations())
    return mesh


def flip_old(mesh):
    mesh.flip_until_delaunay_old()


def flip_new(mesh):
    mesh.flip_until_delaunay()


perfplot.show(
    setup=setup,
    kernels=[flip_old],
    n_range=[2 ** k for k in range(5, 13)],
    equality_check=None
)
