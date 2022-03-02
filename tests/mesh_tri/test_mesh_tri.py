import os
import pathlib
import platform
import tempfile

import meshio
import meshzoo
import numpy as np
import pytest

import meshplex

from ..helpers import assert_norms, is_near_equal, run

this_dir = pathlib.Path(__file__).resolve().parent


def _compute_polygon_area(pts):
    # shoelace formula
    return (
        np.abs(
            np.dot(pts[0], np.roll(pts[1], -1)) - np.dot(np.roll(pts[0], -1), pts[1])
        )
        / 2
    )


# The dtype restriction is because of np.bincount.
# See  <https://github.com/numpy/numpy/issues/17760> and
# <https://github.com/nschloe/meshplex/issues/90>.
cell_dtypes = []
cell_dtypes += [
    np.int32,
]
if platform.architecture()[0] == "64bit":
    cell_dtypes += [
        np.uint32,  # when numpy is fixed, this can go to all arches
        np.int64,
        # np.uint64  # depends on the numpy fix
    ]


@pytest.mark.parametrize("cells_dtype", cell_dtypes)
def test_unit_triangle(cells_dtype):
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2]], dtype=cells_dtype)
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    # ce_ratios
    assert is_near_equal(mesh.ce_ratios.T, [0.0, 0.5, 0.5], tol)

    # control volumes
    assert is_near_equal(mesh.control_volumes, [0.25, 0.125, 0.125], tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert is_near_equal(mesh.cell_circumcenters, [0.5, 0.5], tol)

    # centroids
    assert is_near_equal(mesh.cell_centroids, [1.0 / 3.0, 1.0 / 3.0], tol)
    assert is_near_equal(mesh.cell_barycenters, [1.0 / 3.0, 1.0 / 3.0], tol)

    # control volume centroids
    print(mesh.control_volume_centroids)
    assert is_near_equal(
        mesh.control_volume_centroids,
        [[0.25, 0.25], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
        tol,
    )

    # incenter
    assert is_near_equal(
        mesh.cell_incenters, [[(2 - np.sqrt(2)) / 2, (2 - np.sqrt(2)) / 2]], tol
    )

    # circumcenter
    assert is_near_equal(mesh.cell_circumcenters, [[0.5, 0.5]], tol)

    assert mesh.num_delaunay_violations == 0

    assert mesh.genus == 0

    mesh.get_cell_mask()
    mesh.get_edge_mask()
    mesh.get_vertex_mask()

    # dummy subdomain marker test
    class Subdomain:
        is_boundary_only = False

        def is_inside(self, X):
            return np.ones(X.shape[1:], dtype=bool)

    cell_mask = mesh.get_cell_mask(Subdomain())
    assert np.sum(cell_mask) == 1

    # save
    _, filename = tempfile.mkstemp(suffix=".png")
    mesh.save(filename)
    os.remove(filename)
    _, filename = tempfile.mkstemp(suffix=".vtk")
    mesh.save(filename)
    os.remove(filename)


def test_regular_tri_additional_points():
    points = np.array(
        [
            [0.0, 3.4, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.3, 4.4, 0.0],
        ]
    )
    cells = np.array([[1, 2, 3]])
    mesh = meshplex.MeshTri(points, cells)

    assert np.array_equal(mesh.is_point_used, [False, True, True, True, False])
    assert np.array_equal(mesh.is_boundary_point, [False, True, True, True, False])
    assert np.array_equal(mesh.is_interior_point, [False, False, False, False, False])

    tol = 1.0e-14

    assert np.array_equal(mesh.cells("points"), [[1, 2, 3]])

    mesh.create_facets()
    assert np.array_equal(mesh.cells("edges"), [[2, 1, 0]])
    assert np.array_equal(mesh.edges["points"], [[1, 2], [1, 3], [2, 3]])

    # ce_ratios
    assert is_near_equal(mesh.ce_ratios.T, [0.0, 0.5, 0.5], tol)

    # control volumes
    assert is_near_equal(mesh.control_volumes, [0.0, 0.25, 0.125, 0.125, 0.0], tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert is_near_equal(mesh.cell_circumcenters, [0.5, 0.5, 0.0], tol)

    # Centroids.
    # Nans appear here as the some points aren't part of any cell and hence have no
    # control volume.
    cvc = mesh.control_volume_centroids
    assert np.all(np.isnan(cvc[0]))
    assert np.all(np.isnan(cvc[4]))
    assert is_near_equal(
        cvc[1:4],
        [[0.25, 0.25, 0.0], [2.0 / 3.0, 1.0 / 6.0, 0.0], [1.0 / 6.0, 2.0 / 3.0, 0.0]],
        tol,
    )
    assert mesh.num_delaunay_violations == 0


def test_regular_tri_order():
    points = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cells = np.array([[0, 1, 2]])

    mesh = meshplex.MeshTri(points, cells)
    assert all((mesh.cells("points") == [0, 1, 2]).flat)

    tol = 1.0e-14

    # ce_ratios
    assert is_near_equal(mesh.ce_ratios.T, [0.5, 0.0, 0.5], tol)

    # control volumes
    assert is_near_equal(mesh.control_volumes, [0.125, 0.25, 0.125], tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert is_near_equal(mesh.cell_circumcenters, [0.5, 0.5, 0.0], tol)

    # centroids
    assert is_near_equal(
        mesh.control_volume_centroids,
        [[1.0 / 6.0, 2.0 / 3.0, 0.0], [0.25, 0.25, 0.0], [2.0 / 3.0, 1.0 / 6.0, 0.0]],
        tol,
    )

    assert mesh.num_delaunay_violations == 0


@pytest.mark.parametrize("a", [1.0, 2.0])
def test_regular_tri2(a):
    points = (
        np.array(
            [
                [-0.5, -0.5 * np.sqrt(3.0), 0],
                [-0.5, +0.5 * np.sqrt(3.0), 0],
                [1, 0, 0],
            ]
        )
        / np.sqrt(3)
        * a
    )
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    # ce_ratios
    val = 0.5 / np.sqrt(3.0)
    assert is_near_equal(mesh.ce_ratios, [val, val, val], tol)

    # control volumes
    vol = np.sqrt(3.0) / 4 * a**2
    assert is_near_equal(mesh.control_volumes, [vol / 3.0, vol / 3.0, vol / 3.0], tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [vol], tol)

    # circumcenters
    assert is_near_equal(mesh.cell_circumcenters, [0.0, 0.0, 0.0], tol)


# def test_degenerate_small0():
#     h = 1.0e-3
#     points = np.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0.5, h, 0.0],
#         ])
#     cells = np.array([[0, 1, 2]])
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
#     assert is_near_equal(
#         mesh.get_control_volumes(),
#         [alpha1, alpha1, alpha2],
#         tol
#         )

#     # cell volumes
#     self.assertAlmostEqual(mesh.cell_volumes[0], 0.5 * h, delta=tol)

#     # surface areas
#     edge_length = np.sqrt(0.5**2 + h**2)
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

#     self.assertEqual(mesh.num_delaunay_violations, 0)


@pytest.mark.parametrize(
    "h",
    # TODO [1.0e0, 1.0e-1]
    [1.0e0],
)
def test_degenerate_small0b(h):
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells, sort_cells=True)  # test sort_cells, too

    tol = 1.0e-14

    # edge lengths
    el = np.sqrt(0.5**2 + h**2)
    assert is_near_equal(mesh.edge_lengths.T, [el, el, 1.0], tol)

    # ce_ratios
    ce0 = 0.5 / h * (h**2 - 0.25)
    ce12 = 0.25 / h
    assert is_near_equal(mesh.ce_ratios.T, [ce12, ce12, ce0], tol)

    # control volumes
    cv12 = 0.25 * (1.0**2 * ce0 + (0.25 + h**2) * ce12)
    cv0 = 0.5 * (0.25 + h**2) * ce12
    assert is_near_equal(mesh.control_volumes, [cv12, cv12, cv0], tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # circumcenters
    assert is_near_equal(mesh.cell_circumcenters, [0.5, 0.375, 0.0], tol)

    assert mesh.num_delaunay_violations == 0


# # TODO parametrize with flat boundary correction
# def test_degenerate_small0b_fcc():
#     h = 1.0e-3
#     points = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0]])
#     cells = np.array([[0, 1, 2]])
#     mesh = meshplex.MeshTri(points, cells)
#
#     tol = 1.0e-14
#
#     # edge lengths
#     el = np.sqrt(0.5 ** 2 + h ** 2)
#     assert is_near_equal(mesh.edge_lengths.T, [el, el, 1.0], tol)
#
#     # ce_ratios
#     ce = h
#     assert is_near_equal(mesh.ce_ratios.T, [ce, ce, 0.0], tol)
#
#     # control volumes
#     cv = ce * el
#     alpha = 0.25 * el * cv
#     beta = 0.5 * h - 2 * alpha
#     assert is_near_equal(mesh.control_volumes, [alpha, alpha, beta], tol)
#
#     # cell volumes
#     assert is_near_equal(mesh.cell_volumes, [0.5 * h], tol)
#
#     # surface areas
#     g = np.sqrt((0.5 * el) ** 2 + (ce * el) ** 2)
#     alpha = 0.5 * el + g
#     beta = el + (1.0 - 2 * g)
#     assert is_near_equal(mesh.surface_areas, [alpha, alpha, beta], tol)
#
#     # centroids
#     centroids = mesh.control_volume_centroids
#     alpha = 1.0 / 6000.0
#     gamma = 0.00038888918518558031
#     assert is_near_equal(centroids[0], [0.166667, alpha, 0.0], tol)
#     assert is_near_equal(centroids[1], [0.833333, alpha, 0.0], tol)
#     assert is_near_equal(centroids[2], [0.5, gamma, 0.0], tol)

#     assert mesh.num_delaunay_violations == 0


@pytest.mark.parametrize("h, a", [(1.0e-3, 0.3)])
def test_degenerate_small1(h, a):
    points = np.array([[0, 0, 0], [1, 0, 0], [a, h, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-12

    # edge lengths
    el0 = np.sqrt((1.0 - a) ** 2 + h**2)
    el1 = np.sqrt(a**2 + h**2)
    el2 = 1.0
    assert is_near_equal(mesh.edge_lengths.T, [[el0, el1, el2]], tol)

    # ce_ratios
    ce0 = 0.5 * a / h
    ce1 = 0.5 * (1 - a) / h
    ce2 = 0.5 * (h - (1 - a) * a / h) / el2
    assert is_near_equal(mesh.ce_ratios[:, 0], [ce0, ce1, ce2], 1.0e-8)

    # # control volumes
    # cv1 = ce1 * el1
    # alpha1 = 0.25 * el1 * cv1
    # cv2 = ce2 * el2
    # alpha2 = 0.25 * el2 * cv2
    # beta = 0.5 * h - (alpha1 + alpha2)
    # assert is_near_equal(mesh.control_volumes, [alpha1, alpha2, beta], tol)
    # assert abs(sum(mesh.control_volumes) - 0.5 * h) < tol

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # # surface areas
    # b1 = np.sqrt((0.5 * el1) ** 2 + cv1 ** 2)
    # alpha0 = b1 + 0.5 * el1
    # b2 = np.sqrt((0.5 * el2) ** 2 + cv2 ** 2)
    # alpha1 = b2 + 0.5 * el2
    # total = 1.0 + el1 + el2
    # alpha2 = total - alpha0 - alpha1
    # assert is_near_equal(mesh.surface_areas, [alpha0, alpha1, alpha2], tol)

    assert mesh.num_delaunay_violations == 0


@pytest.mark.parametrize("h", [1.0e-2])
def test_degenerate_small2(h):
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, h, 0.0], [0.5, -h, 0.0]])
    cells = np.array([[0, 1, 2], [0, 1, 3]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-11

    # ce_ratios
    alpha = h - 1.0 / (4 * h)
    beta = 1.0 / (4 * h)
    assert is_near_equal(mesh.signed_circumcenter_distances, [alpha], tol)

    alpha2 = (h - 1.0 / (4 * h)) / 2
    assert is_near_equal(
        mesh.ce_ratios, [[beta, beta], [beta, beta], [alpha2, alpha2]], tol
    )

    # control volumes
    alpha1 = 0.125 * (3 * h - 1.0 / (4 * h))
    alpha2 = 0.125 * (h + 1.0 / (4 * h))
    assert is_near_equal(mesh.control_volumes, [alpha1, alpha1, alpha2, alpha2], tol)

    # circumcenters
    assert is_near_equal(
        mesh.cell_circumcenters, [[0.5, -12.495, 0.0], [0.5, +12.495, 0.0]], tol
    )

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [0.5 * h, 0.5 * h], tol)

    assert mesh.num_delaunay_violations == 1


def test_rectanglesmall():
    points = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]])

    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    assert is_near_equal(mesh.signed_circumcenter_distances, [0.0], tol)

    assert is_near_equal(mesh.ce_ratios, [[5.0, 0.05], [0.0, 5.0], [0.05, 0.0]], tol)
    assert is_near_equal(mesh.control_volumes, [2.5, 2.5, 2.5, 2.5], tol)
    assert is_near_equal(mesh.cell_volumes, [5.0, 5.0], tol)
    assert mesh.num_delaunay_violations == 0


def test_pacman():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman.vtu")

    run(
        mesh,
        54.312974717523744,
        [1.9213504740523146, 0.07954185111555329],
        [403.5307055719196, 0.5512267577002408],
        [1.3816992621175055, 0.0443755870238773],
    )

    assert mesh.num_delaunay_violations == 0


def test_shell():
    points = np.array(
        [
            [+0.0, +0.0, +1.0],
            [+1.0, +0.0, +0.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
            [+0.0, -1.0, +0.0],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 1, 4]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14
    ce_ratios = 0.5 / np.sqrt(3.0) * np.ones((4, 3))
    assert is_near_equal(mesh.ce_ratios.T, ce_ratios, tol)

    cv = np.array([2.0, 1.0, 1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert is_near_equal(mesh.control_volumes, cv, tol)

    cell_vols = np.sqrt(3.0) / 2.0 * np.ones(4)
    assert is_near_equal(mesh.cell_volumes, cell_vols, tol)

    assert mesh.num_delaunay_violations == 0


def test_sphere():
    points, cells = meshzoo.icosa_sphere(5)
    mesh = meshplex.Mesh(points, cells)

    run(
        mesh,
        12.413437988936916,
        [0.7864027242108207, 0.05524648209283611],
        [128.70115197256447, 0.3605511489598192],
        [0.5593675314375034, 0.02963260270642986],
    )


def test_update_point_coordinates():
    mesh = meshio.read(this_dir / ".." / "meshes" / "pacman.vtu")
    assert np.all(np.abs(mesh.points[:, 2]) < 1.0e-15)

    mesh1 = meshplex.MeshTri(mesh.points, mesh.get_cells_type("triangle"))

    np.random.seed(123)
    X2 = mesh.points + 1.0e-2 * np.random.rand(*mesh.points.shape)
    mesh2 = meshplex.MeshTri(X2, mesh.get_cells_type("triangle"))

    mesh1.points = X2

    tol = 1.0e-12
    assert is_near_equal(mesh1.cell_volumes, mesh2.cell_volumes, tol)


def test_inradius():
    # 3-4-5 triangle
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    assert is_near_equal(mesh.cell_inradius, [1.0], tol)

    # 30-60-90 triangle
    a = 1.0
    points = np.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * np.sqrt(3.0), 0.0]]
    )
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    assert is_near_equal(mesh.cell_inradius, [a / 4 * (np.sqrt(3) - 1)], tol)


def test_circumradius():
    # 3-4-5 triangle
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    assert is_near_equal(mesh.cell_circumradius, [2.5], tol)

    # 30-60-90 triangle
    a = 1.0
    points = np.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * np.sqrt(3.0), 0.0]]
    )
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    assert is_near_equal(mesh.cell_circumradius, [a / 2], tol)


def test_quality():
    # 3-4-5 triangle
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-15

    q = mesh.q_radius_ratio
    assert is_near_equal(q, 2 * mesh.cell_inradius / mesh.cell_circumradius, tol)

    # 30-60-90 triangle
    a = 1.0
    points = np.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * np.sqrt(3.0), 0.0]]
    )
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    q = mesh.q_radius_ratio
    assert is_near_equal(q, 2 * mesh.cell_inradius / mesh.cell_circumradius, tol)


def test_angles():
    # 3-4-5 triangle
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    tol = 1.0e-14

    assert is_near_equal(
        mesh.angles,
        [[np.pi / 2], [np.arcsin(4.0 / 5.0)], [np.arcsin(3.0 / 5.0)]],
        tol,
    )

    # 30-60-90 triangle
    a = 1.0
    points = np.array(
        [[0.0, 0.0, 0.0], [a / 2, 0.0, 0.0], [0.0, a / 2 * np.sqrt(3.0), 0.0]]
    )
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    ic = mesh.angles / np.pi * 180
    assert is_near_equal(ic, [[90], [60], [30]], tol)


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
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [x, y, 0.0],
        ]
    )
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

    mesh = meshplex.MeshTri(X, cells)

    # Inspect the covolumes in left cell.
    edge_length = np.sqrt(x**2 + y**2)
    ref = np.array([edge_length, edge_length, 1.0])
    assert np.all(np.abs(mesh.edge_lengths[:, 3] - ref) < 1.0e-12)
    #
    alpha = 0.5 / x * y * np.sqrt(y**2 + x**2)
    beta = 0.5 / x * (x**2 - y**2)
    ref = [alpha, alpha, beta]
    covolumes = mesh.ce_ratios[:, 3] * mesh.edge_lengths[:, 3]
    assert np.all(np.abs(covolumes - ref) < 1.0e-12)

    #
    beta = np.sqrt(alpha**2 + 0.2**2 + 0.25**2)
    control_volume_corners = np.array(
        [
            mesh.cell_circumcenters[0][:2],
            mesh.cell_circumcenters[1][:2],
            mesh.cell_circumcenters[2][:2],
            mesh.cell_circumcenters[3][:2],
        ]
    )
    ref_area = _compute_polygon_area(control_volume_corners.T)

    assert np.abs(mesh.control_volumes[4] - ref_area) < 1.0e-12

    cv = np.zeros(X.shape[0])
    for edges, ce_ratios in zip(mesh.idx[1].T, mesh.ce_ratios.T):
        for i, ce in zip(edges, ce_ratios):
            ei = mesh.points[i[1]] - mesh.points[i[0]]
            cv[i] += 0.25 * ce * np.dot(ei, ei)

    assert np.all(np.abs(cv - mesh.control_volumes) < 1.0e-12 * cv)


def test_set_points():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    mesh.set_points([0.1, 0.1], [0])
    ref = mesh.cell_volumes.copy()

    mesh2 = meshplex.MeshTri(mesh.points, mesh.cells("points"))
    assert np.all(np.abs(ref - mesh2.cell_volumes) < 1.0e-10)


def test_reference_vals_pacman():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman.vtu")
    mesh = meshplex.MeshTri(mesh.points[:, :2], mesh.cells("points"))

    assert_norms(
        mesh.points,
        [3.0544932927920363e03, 8.8106015937625088e01, 4.2500000000000000e00],
        1.0e-15,
    )
    assert_norms(
        mesh.half_edge_coords,
        [1.6674048509514694e03, 1.9463913181988705e01, 3.4718650853971766e-01],
        1.0e-15,
    )
    assert_norms(
        mesh.ei_dot_ei,
        [3.7884391635599366e02, 5.5826366101867908e00, 1.2739502744897091e-01],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_partitions,
        [5.4312974717523744e01, 5.6678942198355931e-01, 8.5844237283573006e-03],
        1.0e-12,
    )
    assert_norms(
        mesh.cell_centroids,
        [5.5718766084264298e03, 1.1728553160741399e02, 4.1694621840309081e00],
        1.0e-15,
    )
    assert_norms(
        mesh.edge_lengths,
        [1.3250455158127431e03, 1.9463913181988705e01, 3.5692440018716975e-01],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_volumes,
        [5.4312974717523744e01, 1.3816992621175055e00, 4.4375587023877297e-02],
        1.0e-15,
    )
    assert_norms(
        mesh.ce_ratios,
        [1.3499477445918124e03, 2.0088073714816950e01, 5.5122675770024077e-01],
        1.0e-14,
    )
    assert_norms(
        mesh.control_volumes,
        [5.4312974717523744e01, 1.9213504740523146e00, 7.9541851115553286e-02],
        1.0e-15,
    )
    assert_norms(
        mesh.control_volume_centroids,
        [3.0478135855839828e03, 8.7829558499197603e01, 4.1869842124121526e00],
        1.0e-15,
    )
    assert_norms(
        mesh.signed_cell_volumes,
        [5.4312974717523744e01, 1.3816992621175055e00, 4.4375587023877297e-02],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_circumcenters,
        [5.5720855984960217e03, 1.1729075391802718e02, 4.1780370020570583e00],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_circumradius,
        [2.5571964535497142e02, 6.5009888666474742e00, 1.8757161840809547e-01],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_incenters,
        [5.5715778346847819e03, 1.1727887700257899e02, 4.1655515539293466e00],
        1.0e-15,
    )
    assert_norms(
        mesh.cell_inradius,
        [1.2685029853822100e02, 3.2249724655140719e00, 9.1724742697552028e-02],
        1.0e-15,
    )
    assert_norms(
        mesh.q_radius_ratio,
        [1.5359568026022387e03, 3.9044827334140905e01, 9.9999895608618172e-01],
        1.0e-15,
    )
