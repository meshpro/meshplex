import os
import pathlib
import tempfile

import meshio
import numpy
import pytest
from helpers import near_equal, run

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def _compute_polygon_area(pts):
    # shoelace formula
    return (
        numpy.abs(
            numpy.dot(pts[0], numpy.roll(pts[1], -1))
            - numpy.dot(numpy.roll(pts[0], -1), pts[1])
        )
        / 2
    )


@pytest.mark.parametrize(
    "cells_dtype",
    [
        numpy.int64
        # depends on <https://github.com/numpy/numpy/issues/17760>
        # numpy.uint64
    ],
)
def test_unit_triangle(cells_dtype):
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = numpy.array([[0, 1, 2]], dtype=cells_dtype)
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    assert (mesh.local_idx.T == [[1, 2], [2, 0], [0, 1]]).all()
    assert mesh.local_idx_inv == [[(0, 2), (1, 1)], [(0, 0), (1, 2)], [(0, 1), (1, 0)]]

    # ce_ratios
    assert near_equal(mesh.ce_ratios.T, [0.0, 0.5, 0.5], tol)

    # control volumes
    assert near_equal(mesh.control_volumes, [0.25, 0.125, 0.125], tol)

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert near_equal(mesh.cell_circumcenters, [0.5, 0.5], tol)

    # centroids
    assert near_equal(mesh.cell_centroids, [1.0 / 3.0, 1.0 / 3.0], tol)
    assert near_equal(mesh.cell_barycenters, [1.0 / 3.0, 1.0 / 3.0], tol)

    # control volume centroids
    assert near_equal(
        mesh.control_volume_centroids,
        [[0.25, 0.25], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
        tol,
    )

    # incenter
    assert near_equal(
        mesh.cell_incenters, [[(2 - numpy.sqrt(2)) / 2, (2 - numpy.sqrt(2)) / 2]], tol
    )

    # circumcenter
    assert near_equal(mesh.cell_circumcenters, [[0.5, 0.5]], tol)

    assert mesh.num_delaunay_violations() == 0

    assert mesh.genus == 0.5

    mesh.get_cell_mask()
    mesh.get_edge_mask()
    mesh.get_vertex_mask()

    # dummy subdomain marker test
    class Subdomain:
        is_boundary_only = False

        def is_inside(self, X):
            return numpy.ones(X.shape[1:], dtype=bool)

    cell_mask = mesh.get_cell_mask(Subdomain())
    assert sum(cell_mask) == 1

    # save
    _, filename = tempfile.mkstemp(suffix=".png")
    mesh.save(filename)
    os.remove(filename)
    _, filename = tempfile.mkstemp(suffix=".vtk")
    mesh.save(filename)
    os.remove(filename)


def test_regular_tri_additional_points():
    points = numpy.array(
        [
            [0.0, 3.4, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.3, 4.4, 0.0],
        ]
    )
    cells = numpy.array([[1, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    assert numpy.array_equal(mesh.is_point_used, [False, True, True, True, False])
    assert numpy.array_equal(mesh.is_boundary_point, [False, True, True, True, False])
    assert numpy.array_equal(
        mesh.is_interior_point, [False, False, False, False, False]
    )

    tol = 1.0e-14

    assert numpy.array_equal(mesh.cells["points"], [[1, 2, 3]])

    mesh.create_edges()
    assert numpy.array_equal(mesh.cells["edges"], [[2, 1, 0]])
    assert numpy.array_equal(mesh.edges["points"], [[1, 2], [1, 3], [2, 3]])

    # ce_ratios
    assert near_equal(mesh.ce_ratios.T, [0.0, 0.5, 0.5], tol)

    # control volumes
    assert near_equal(mesh.control_volumes, [0.0, 0.25, 0.125, 0.125, 0.0], tol)

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert near_equal(mesh.cell_circumcenters, [0.5, 0.5, 0.0], tol)

    # Centroids.
    # Nans appear here as the some points aren't part of any cell and hence have no
    # control volume.
    cvc = mesh.control_volume_centroids
    assert numpy.all(numpy.isnan(cvc[0]))
    assert numpy.all(numpy.isnan(cvc[4]))
    assert near_equal(
        cvc[1:4],
        [[0.25, 0.25, 0.0], [2.0 / 3.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 2.0 / 3.0, 0.0]],
        tol,
    )
    assert mesh.num_delaunay_violations() == 0


def test_regular_tri_order():
    points = numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cells = numpy.array([[0, 1, 2]])

    mesh = meshplex.MeshTri(points, cells)
    assert all((mesh.cells["points"] == [0, 1, 2]).flat)

    tol = 1.0e-14

    # ce_ratios
    assert near_equal(mesh.ce_ratios.T, [0.5, 0.0, 0.5], tol)

    # control volumes
    assert near_equal(mesh.control_volumes, [0.125, 0.25, 0.125], tol)

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert near_equal(mesh.cell_circumcenters, [0.5, 0.5, 0.0], tol)

    # centroids
    assert near_equal(
        mesh.control_volume_centroids,
        [[1.0 / 6.0, 2.0 / 3.0, 0.0], [0.25, 0.25, 0.0], [2.0 / 3.0, 1.0 / 6.0, 0.0]],
        tol,
    )

    assert mesh.num_delaunay_violations() == 0


@pytest.mark.parametrize("a", [1.0, 2.0])
def test_regular_tri2(a):
    points = (
        numpy.array(
            [
                [-0.5, -0.5 * numpy.sqrt(3.0), 0],
                [-0.5, +0.5 * numpy.sqrt(3.0), 0],
                [1, 0, 0],
            ]
        )
        / numpy.sqrt(3)
        * a
    )
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    # ce_ratios
    val = 0.5 / numpy.sqrt(3.0)
    assert near_equal(mesh.ce_ratios, [val, val, val], tol)

    # control volumes
    vol = numpy.sqrt(3.0) / 4 * a ** 2
    assert near_equal(mesh.control_volumes, [vol / 3.0, vol / 3.0, vol / 3.0], tol)

    # cell volumes
    assert near_equal(mesh.cell_volumes, [vol], tol)

    # circumcenters
    assert near_equal(mesh.cell_circumcenters, [0.0, 0.0, 0.0], tol)


# def test_degenerate_small0():
#     h = 1.0e-3
#     points = numpy.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0.5, h, 0.0],
#         ])
#     cells = numpy.array([[0, 1, 2]])
#     mesh = meshplex.MeshTri(
#             points,
#             cells,
#             allow_negative_volumes=True
#             )

#     tol = 1.0e-14

#     # ce_ratios
#     alpha = 0.5 * h - 1.0 / (8*h)
#     beta = 1.0 / (4*h)
#     assertAlmostEqual(mesh.get_ce_ratios_per_edge()[0], alpha, delta=tol)
#     self.assertAlmostEqual(mesh.get_ce_ratios_per_edge()[1], beta, delta=tol)
#     self.assertAlmostEqual(mesh.get_ce_ratios_per_edge()[2], beta, delta=tol)

#     # control volumes
#     alpha1 = 0.0625 * (3*h - 1.0/(4*h))
#     alpha2 = 0.125 * (h + 1.0 / (4*h))
#     assert near_equal(
#         mesh.get_control_volumes(),
#         [alpha1, alpha1, alpha2],
#         tol
#         )

#     # cell volumes
#     self.assertAlmostEqual(mesh.cell_volumes[0], 0.5 * h, delta=tol)

#     # surface areas
#     edge_length = numpy.sqrt(0.5**2 + h**2)
#     # circumference = 1.0 + 2 * edge_length
#     alpha = 0.5 * (1.0 + edge_length)
#     self.assertAlmostEqual(mesh.surface_areas[0], alpha, delta=tol)
#     self.assertAlmostEqual(mesh.surface_areas[1], alpha, delta=tol)
#     self.assertAlmostEqual(mesh.surface_areas[2], edge_length, delta=tol)

#     # centroids
#     alpha = -41.666666669333345
#     beta = 0.58333199998399976
#      self.assertAlmostEqual(
#              mesh.centroids[0][0],
#              0.416668000016,
#              delta=tol
#              )
#     self.assertAlmostEqual(mesh.centroids[0][1], alpha, delta=tol)
#     self.assertAlmostEqual(mesh.centroids[0][2], 0.0, delta=tol)

#     self.assertAlmostEqual(mesh.centroids[1][0], beta, delta=tol)
#     self.assertAlmostEqual(mesh.centroids[1][1], alpha, delta=tol)
#     self.assertAlmostEqual(mesh.centroids[1][2], 0.0, delta=tol)

#     self.assertAlmostEqual(mesh.centroids[2][0], 0.5, delta=tol)
#     self.assertAlmostEqual(mesh.centroids[2][1], -41.666, delta=tol)
#     self.assertAlmostEqual(mesh.centroids[2][2], 0.0, delta=tol)

#     self.assertEqual(mesh.num_delaunay_violations(), 0)


@pytest.mark.parametrize(
    "h",
    # TODO [1.0e0, 1.0e-1]
    [1.0e0],
)
def test_degenerate_small0b(h):
    points = numpy.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells, sort_cells=True)  # test sort_cells, too

    tol = 1.0e-14

    # edge lengths
    el = numpy.sqrt(0.5 ** 2 + h ** 2)
    assert near_equal(mesh.edge_lengths.T, [el, el, 1.0], tol)

    # ce_ratios
    ce0 = 0.5 / h * (h ** 2 - 0.25)
    ce12 = 0.25 / h
    assert near_equal(mesh.ce_ratios.T, [ce12, ce12, ce0], tol)

    # control volumes
    cv12 = 0.25 * (1.0 ** 2 * ce0 + (0.25 + h ** 2) * ce12)
    cv0 = 0.5 * (0.25 + h ** 2) * ce12
    assert near_equal(mesh.control_volumes, [cv12, cv12, cv0], tol)

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # circumcenters
    assert near_equal(mesh.cell_circumcenters, [0.5, 0.375, 0.0], tol)

    assert mesh.num_delaunay_violations() == 0


# # TODO parametrize with flat boundary correction
# def test_degenerate_small0b_fcc():
#     h = 1.0e-3
#     points = numpy.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0]])
#     cells = numpy.array([[0, 1, 2]])
#     mesh = meshplex.MeshTri(points, cells)
#
#     tol = 1.0e-14
#
#     # edge lengths
#     el = numpy.sqrt(0.5 ** 2 + h ** 2)
#     assert near_equal(mesh.edge_lengths.T, [el, el, 1.0], tol)
#
#     # ce_ratios
#     ce = h
#     assert near_equal(mesh.ce_ratios.T, [ce, ce, 0.0], tol)
#
#     # control volumes
#     cv = ce * el
#     alpha = 0.25 * el * cv
#     beta = 0.5 * h - 2 * alpha
#     assert near_equal(mesh.control_volumes, [alpha, alpha, beta], tol)
#
#     # cell volumes
#     assert near_equal(mesh.cell_volumes, [0.5 * h], tol)
#
#     # surface areas
#     g = numpy.sqrt((0.5 * el) ** 2 + (ce * el) ** 2)
#     alpha = 0.5 * el + g
#     beta = el + (1.0 - 2 * g)
#     assert near_equal(mesh.surface_areas, [alpha, alpha, beta], tol)
#
#     # centroids
#     centroids = mesh.control_volume_centroids
#     alpha = 1.0 / 6000.0
#     gamma = 0.00038888918518558031
#     assert near_equal(centroids[0], [0.166667, alpha, 0.0], tol)
#     assert near_equal(centroids[1], [0.833333, alpha, 0.0], tol)
#     assert near_equal(centroids[2], [0.5, gamma, 0.0], tol)

#     assert mesh.num_delaunay_violations() == 0


@pytest.mark.parametrize("h, a", [(1.0e-3, 0.3)])
def test_degenerate_small1(h, a):
    points = numpy.array([[0, 0, 0], [1, 0, 0], [a, h, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-12

    # edge lengths
    el0 = numpy.sqrt((1.0 - a) ** 2 + h ** 2)
    el1 = numpy.sqrt(a ** 2 + h ** 2)
    el2 = 1.0
    assert near_equal(mesh.edge_lengths.T, [[el0, el1, el2]], tol)

    # ce_ratios
    ce0 = 0.5 * a / h
    ce1 = 0.5 * (1 - a) / h
    ce2 = 0.5 * (h - (1 - a) * a / h) / el2
    assert near_equal(mesh.ce_ratios[:, 0], [ce0, ce1, ce2], 1.0e-8)

    # # control volumes
    # cv1 = ce1 * el1
    # alpha1 = 0.25 * el1 * cv1
    # cv2 = ce2 * el2
    # alpha2 = 0.25 * el2 * cv2
    # beta = 0.5 * h - (alpha1 + alpha2)
    # assert near_equal(mesh.control_volumes, [alpha1, alpha2, beta], tol)
    # assert abs(sum(mesh.control_volumes) - 0.5 * h) < tol

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # # surface areas
    # b1 = numpy.sqrt((0.5 * el1) ** 2 + cv1 ** 2)
    # alpha0 = b1 + 0.5 * el1
    # b2 = numpy.sqrt((0.5 * el2) ** 2 + cv2 ** 2)
    # alpha1 = b2 + 0.5 * el2
    # total = 1.0 + el1 + el2
    # alpha2 = total - alpha0 - alpha1
    # assert near_equal(mesh.surface_areas, [alpha0, alpha1, alpha2], tol)

    assert mesh.num_delaunay_violations() == 0


@pytest.mark.parametrize("h", [1.0e-2])
def test_degenerate_small2(h):
    points = numpy.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0], [0.5, -h, 0.0]])
    cells = numpy.array([[0, 1, 2], [0, 1, 3]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-11

    # ce_ratios
    alpha = h - 1.0 / (4 * h)
    beta = 1.0 / (4 * h)
    assert near_equal(mesh.ce_ratios_per_interior_edge, [alpha], tol)

    alpha2 = (h - 1.0 / (4 * h)) / 2
    assert near_equal(
        mesh.ce_ratios, [[beta, beta], [beta, beta], [alpha2, alpha2]], tol
    )

    # control volumes
    alpha1 = 0.125 * (3 * h - 1.0 / (4 * h))
    alpha2 = 0.125 * (h + 1.0 / (4 * h))
    assert near_equal(mesh.control_volumes, [alpha1, alpha1, alpha2, alpha2], tol)

    # circumcenters
    assert near_equal(
        mesh.cell_circumcenters, [[0.5, -12.495, 0.0], [0.5, +12.495, 0.0]], tol
    )

    # cell volumes
    assert near_equal(mesh.cell_volumes, [0.5 * h, 0.5 * h], tol)

    assert mesh.num_delaunay_violations() == 1


def test_rectanglesmall():
    points = numpy.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 3]])

    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    assert near_equal(mesh.ce_ratios_per_interior_edge, [0.0], tol)

    assert near_equal(mesh.ce_ratios, [[5.0, 0.05], [0.0, 5.0], [0.05, 0.0]], tol)
    assert near_equal(mesh.control_volumes, [2.5, 2.5, 2.5, 2.5], tol)
    assert near_equal(mesh.cell_volumes, [5.0, 5.0], tol)
    assert mesh.num_delaunay_violations() == 0


def test_pacman():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman.vtk")

    run(
        mesh,
        73.64573933105898,
        [3.596101914906618, 0.26638548094154696],
        [379.275476266239, 1.2976923100235962],
        [2.6213234038171014, 0.13841739494523228],
    )

    assert mesh.num_delaunay_violations() == 0


def test_shell():
    points = numpy.array(
        [
            [+0.0, +0.0, +1.0],
            [+1.0, +0.0, +0.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
            [+0.0, -1.0, +0.0],
        ]
    )
    cells = numpy.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 4]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14
    ce_ratios = 0.5 / numpy.sqrt(3.0) * numpy.ones((4, 3))
    assert near_equal(mesh.ce_ratios.T, ce_ratios, tol)

    cv = numpy.array([2.0, 1.0, 1.0, 1.0, 1.0]) / numpy.sqrt(3.0)
    assert near_equal(mesh.control_volumes, cv, tol)

    cell_vols = numpy.sqrt(3.0) / 2.0 * numpy.ones(4)
    assert near_equal(mesh.cell_volumes, cell_vols, tol)

    assert mesh.num_delaunay_violations() == 0


def test_sphere():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "sphere.vtk")
    run(
        mesh,
        12.273645818711595,
        [1.0177358705967492, 0.10419690304323895],
        [366.3982135866799, 1.7062353589387327],
        [0.72653362732751214, 0.05350373815413411],
    )
    # assertEqual(mesh.num_delaunay_violations(), 60)


def test_update_point_coordinates():
    mesh = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")
    assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-15)

    mesh1 = meshplex.MeshTri(mesh.points, mesh.get_cells_type("triangle"))

    numpy.random.seed(123)
    X2 = mesh.points + 1.0e-2 * numpy.random.rand(*mesh.points.shape)
    mesh2 = meshplex.MeshTri(X2, mesh.get_cells_type("triangle"))

    mesh1.points = X2

    tol = 1.0e-12
    assert near_equal(mesh1.ei_dot_ej, mesh2.ei_dot_ej, tol)
    assert near_equal(mesh1.cell_volumes, mesh2.cell_volumes, tol)


def test_inradius():
    # 3-4-5 triangle
    points = numpy.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    assert near_equal(mesh.cell_inradius, [1.0], tol)

    # 30-60-90 triangle
    a = 1.0
    points = numpy.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * numpy.sqrt(3.0), 0.0]]
    )
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    assert near_equal(mesh.cell_inradius, [a / 4 * (numpy.sqrt(3) - 1)], tol)


def test_circumradius():
    # 3-4-5 triangle
    points = numpy.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    assert near_equal(mesh.cell_circumradius, [2.5], tol)

    # 30-60-90 triangle
    a = 1.0
    points = numpy.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * numpy.sqrt(3.0), 0.0]]
    )
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    assert near_equal(mesh.cell_circumradius, [a / 2], tol)


def test_quality():
    # 3-4-5 triangle
    points = numpy.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    q = mesh.q_radius_ratio
    assert near_equal(q, 2 * mesh.cell_inradius / mesh.cell_circumradius, tol)

    # 30-60-90 triangle
    a = 1.0
    points = numpy.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * numpy.sqrt(3.0), 0.0]]
    )
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    q = mesh.q_radius_ratio
    assert near_equal(q, 2 * mesh.cell_inradius / mesh.cell_circumradius, tol)


def test_angles():
    # 3-4-5 triangle
    points = numpy.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    assert near_equal(
        mesh.angles,
        [[numpy.pi / 2], [numpy.arcsin(4.0 / 5.0)], [numpy.arcsin(3.0 / 5.0)]],
        tol,
    )

    # 30-60-90 triangle
    a = 1.0
    points = numpy.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * numpy.sqrt(3.0), 0.0]]
    )
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    ic = mesh.angles / numpy.pi * 180
    assert near_equal(ic, [[90], [60], [30]], tol)


def test_flat_boundary():
    #
    #  3___________2
    #  |\_   2   _/|
    #  |  \_   _/  |
    #  | 3  \4/  1 |
    #  |   _/ \_   |
    #  | _/     \_ |
    #  |/    0    \|
    #  0-----------1
    #
    x = 0.4
    y = 0.5
    X = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [x, y, 0.0],
        ]
    )
    cells = numpy.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    mesh = meshplex.MeshTri(X, cells)

    # Inspect the covolumes in left cell.
    edge_length = numpy.sqrt(x ** 2 + y ** 2)
    ref = numpy.array([edge_length, edge_length, 1.0])
    assert numpy.all(numpy.abs(mesh.edge_lengths[:, 3] - ref) < 1.0e-12)
    #
    alpha = 0.5 / x * y * numpy.sqrt(y ** 2 + x ** 2)
    beta = 0.5 / x * (x ** 2 - y ** 2)
    ref = [alpha, alpha, beta]
    covolumes = mesh.ce_ratios[:, 3] * mesh.edge_lengths[:, 3]
    assert numpy.all(numpy.abs(covolumes - ref) < 1.0e-12)

    #
    beta = numpy.sqrt(alpha ** 2 + 0.2 ** 2 + 0.25 ** 2)
    control_volume_corners = numpy.array(
        [
            mesh.cell_circumcenters[0][:2],
            mesh.cell_circumcenters[1][:2],
            mesh.cell_circumcenters[2][:2],
            mesh.cell_circumcenters[3][:2],
        ]
    )
    ref_area = _compute_polygon_area(control_volume_corners.T)

    assert numpy.abs(mesh.control_volumes[4] - ref_area) < 1.0e-12

    cv = numpy.zeros(X.shape[0])
    for edges, ce_ratios in zip(mesh.idx_hierarchy.T, mesh.ce_ratios.T):
        for i, ce in zip(edges, ce_ratios):
            ei = mesh.points[i[1]] - mesh.points[i[0]]
            cv[i] += 0.25 * ce * numpy.dot(ei, ei)

    assert numpy.all(numpy.abs(cv - mesh.control_volumes) < 1.0e-12 * cv)


def test_set_points():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.set_points([0.1, 0.1], [0])
    ref = mesh.cell_volumes.copy()

    mesh2 = meshplex.MeshTri(mesh.points, mesh.cells["points"])
    assert numpy.all(numpy.abs(ref - mesh2.cell_volumes)) < 1.0e-10


if __name__ == "__main__":
    test_set_points()