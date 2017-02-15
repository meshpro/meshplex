# -*- coding: utf-8 -*-
#
import fetch_data
import pytest
import voropy

from math import fsum
import numpy


def _near_equal(a, b, tol):
    return numpy.allclose(a, b, rtol=0.0, atol=tol)


def _run(
        mesh,
        volume,
        convol_norms, ce_ratio_norms, cellvol_norms,
        tol=1.0e-12
        ):
    # if mesh.cells['nodes'].shape[1] == 3:
    #     dim = 2
    # elif mesh.cells['nodes'].shape[1] == 4:
    #     dim = 3
    # else:
    #     raise ValueError('Can only handle triangles and tets.')

    # Check cell volumes.
    total_cellvolume = fsum(mesh.cell_volumes)
    assert abs(volume - total_cellvolume) < tol * volume
    norm2 = numpy.linalg.norm(mesh.cell_volumes, ord=2)
    norm_inf = numpy.linalg.norm(mesh.cell_volumes, ord=numpy.Inf)
    assert _near_equal(cellvol_norms, [norm2, norm_inf], tol)

    # If everything is Delaunay and the boundary elements aren't flat, the
    # volume of the domain is given by
    #   1/n * edge_lengths * ce_ratios.
    # Unfortunately, this isn't always the case.
    # ```
    # total_ce_ratio = \
    #     fsum(mesh.edge_lengths**2 * mesh.get_ce_ratios_per_edge() / dim)
    # self.assertAlmostEqual(volume, total_ce_ratio, delta=tol * volume)
    # ```
    # Check ce_ratio norms.
    # TODO reinstate
    alpha2 = fsum((mesh.get_ce_ratios()**2).flat)
    alpha_inf = max(abs(mesh.get_ce_ratios()).flat)
    assert _near_equal(ce_ratio_norms, [alpha2, alpha_inf], tol)

    # Check the volume by summing over the absolute value of the control
    # volumes.
    vol = fsum(mesh.get_control_volumes())
    assert abs(volume - vol) < tol*volume
    # Check control volume norms.
    norm2 = numpy.linalg.norm(mesh.get_control_volumes(), ord=2)
    norm_inf = numpy.linalg.norm(mesh.get_control_volumes(), ord=numpy.Inf)
    assert _near_equal(convol_norms, [norm2, norm_inf], tol)

    return


def test_regular_tri():
    points = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
        ])
    cells = numpy.array([[0, 1, 2]])
    mesh = voropy.mesh_tri.MeshTri(points, cells)

    tol = 1.0e-14

    # ce_ratios
    assert _near_equal(mesh.get_ce_ratios_per_edge(), [0.5, 0.5, 0.0], tol)

    # control volumes
    assert _near_equal(
        mesh.get_control_volumes(),
        [0.25, 0.125, 0.125],
        tol
        )

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [0.5], tol)

    # circumcenters
    assert _near_equal(
        mesh.get_cell_circumcenters(),
        [0.5, 0.5, 0.0],
        tol
        )

    # centroids
    centroids = mesh.get_control_volume_centroids()
    assert _near_equal(centroids[0], [0.25, 0.25, 0.0], tol)
    assert _near_equal(centroids[1], [2.0/3.0, 1.0/6.0, 0.0], tol)
    assert _near_equal(centroids[2], [1.0/6.0, 2.0/3.0, 0.0], tol)

    assert mesh.num_delaunay_violations() == 0

    return


@pytest.mark.parametrize(
        'a',
        [1.0, 2.0]
        )
def test_regular_tri2(a):
    points = numpy.array([
        [-0.5, -0.5 * numpy.sqrt(3.0), 0],
        [-0.5, +0.5 * numpy.sqrt(3.0), 0],
        [1, 0, 0]
        ]) / numpy.sqrt(3) * a
    cells = numpy.array([[0, 1, 2]])
    mesh = voropy.mesh_tri.MeshTri(points, cells)

    tol = 1.0e-14

    # ce_ratios
    val = 0.5 / numpy.sqrt(3.0)
    assert _near_equal(mesh.get_ce_ratios(), [val, val, val], tol)

    # control volumes
    vol = numpy.sqrt(3.0) / 4 * a**2
    assert _near_equal(
        mesh.get_control_volumes(),
        [vol / 3.0, vol / 3.0, vol / 3.0],
        tol
        )

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [vol], tol)

    # circumcenters
    assert _near_equal(
        mesh.get_cell_circumcenters(),
        [0.0, 0.0, 0.0],
        tol
        )

    return


# def test_degenerate_small0():
#     h = 1.0e-3
#     points = numpy.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0.5, h, 0.0],
#         ])
#     cells = numpy.array([[0, 1, 2]])
#     mesh = voropy.mesh_tri.MeshTri(
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
#     assert _near_equal(
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
#     return


@pytest.mark.parametrize(
        'h',
        # TODO [1.0e0, 1.0e-1]
        [1.0e0]
        )
def test_degenerate_small0b(h):
    points = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, h, 0.0],
        ])
    cells = numpy.array([[0, 1, 2]])
    mesh = voropy.mesh_tri.MeshTri(
            points,
            cells,
            flat_cell_correction=None
            )

    tol = 1.0e-14

    # edge lengths
    el = numpy.sqrt(0.5**2 + h**2)
    assert _near_equal(mesh.get_edge_lengths(), [[el, el, 1.0]], tol)

    # ce_ratios
    ce0 = 0.5/h * (h**2 - 0.25)
    ce12 = 0.25/h
    assert _near_equal(mesh.get_ce_ratios_per_edge(), [ce0, ce12, ce12], tol)

    # control volumes
    cv12 = 0.25 * (1.0**2 * ce0 + (0.25 + h**2) * ce12)
    cv0 = 0.5 * (0.25 + h**2) * ce12
    assert _near_equal(mesh.get_control_volumes(), [cv12, cv12, cv0], tol)

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # surface areas
    ids, vals = mesh.get_surface_areas()
    assert numpy.all(ids == [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    assert _near_equal(
        vals,
        [
            [0.0, 0.5*el, 0.5*el],
            [0.5*el, 0.0, 0.5*el],
            [0.5, 0.5, 0.0],
        ],
        tol
        )

    assert mesh.num_delaunay_violations() == 0
    return


# TODO parametrize with flat boundary correction
def test_degenerate_small0b_fcc():
    h = 1.0e-3
    points = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, h, 0.0],
        ])
    cells = numpy.array([[0, 1, 2]])
    mesh = voropy.mesh_tri.MeshTri(
            points,
            cells,
            flat_cell_correction='full'
            )

    tol = 1.0e-14

    # edge lengths
    el = numpy.sqrt(0.5**2 + h**2)
    assert _near_equal(mesh.get_edge_lengths(), [el, el, 1.0], tol)

    # ce_ratios
    ce = h
    assert _near_equal(mesh.get_ce_ratios(), [ce, ce, 0.0], tol)

    # control volumes
    cv = ce * el
    alpha = 0.25 * el * cv
    beta = 0.5*h - 2*alpha
    assert _near_equal(
        mesh.get_control_volumes(),
        [alpha, alpha, beta],
        tol
        )

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # surface areas
    g = numpy.sqrt((0.5 * el)**2 + (ce * el)**2)
    alpha = 0.5 * el + g
    beta = el + (1.0 - 2*g)
    assert _near_equal(mesh.get_surface_areas(), [alpha, alpha, beta], tol)

    # centroids
    centroids = mesh.get_control_volume_centroids()
    alpha = 1.0 / 6000.0
    gamma = 0.00038888918518558031
    assert _near_equal(centroids[0], [0.166667, alpha, 0.0], tol)
    assert _near_equal(centroids[1], [0.833333, alpha, 0.0], tol)
    assert _near_equal(centroids[2], [0.5, gamma, 0.0], tol)

    assert mesh.num_delaunay_violations() == 0
    return


@pytest.mark.parametrize('h, a', [(1.0e-3, 0.3)])
def test_degenerate_small1(h, a):
    points = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [a, h, 0.0],
        ])
    cells = numpy.array([[0, 1, 2]])
    mesh = voropy.mesh_tri.MeshTri(
            points,
            cells,
            flat_cell_correction='full'
            )

    tol = 1.0e-14

    # edge lengths
    el1 = numpy.sqrt(a**2 + h**2)
    el2 = numpy.sqrt((1.0 - a)**2 + h**2)
    assert _near_equal(mesh.get_edge_lengths(), [[el2, el1, 1.0]], tol)

    # ce_ratios
    ce1 = 0.5 * h / a
    ce2 = 0.5 * h / (1.0 - a)
    assert _near_equal(mesh.get_ce_ratios_per_edge(), [0.0, ce1, ce2], tol)

    # control volumes
    cv1 = ce1 * el1
    alpha1 = 0.25 * el1 * cv1
    cv2 = ce2 * el2
    alpha2 = 0.25 * el2 * cv2
    beta = 0.5*h - (alpha1 + alpha2)
    assert _near_equal(
        mesh.get_control_volumes(),
        [alpha1, alpha2, beta],
        tol
        )
    assert abs(sum(mesh.get_control_volumes()) - 0.5*h) < tol

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [0.5 * h], tol)

    # surface areas
    b1 = numpy.sqrt((0.5*el1)**2 + cv1**2)
    alpha0 = b1 + 0.5*el1
    b2 = numpy.sqrt((0.5*el2)**2 + cv2**2)
    alpha1 = b2 + 0.5*el2
    total = 1.0 + el1 + el2
    alpha2 = total - alpha0 - alpha1
    surf = mesh.get_surface_areas()
    assert _near_equal(surf, [alpha0, alpha1, alpha2], tol)

    assert mesh.num_delaunay_violations() == 0
    return


@pytest.mark.parametrize('h', [1.0e-2])
def test_degenerate_small2(h):
    points = numpy.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, h, 0.0],
        [0.5, -h, 0.0]
        ])
    cells = numpy.array([[0, 1, 2], [0, 1, 3]])
    mesh = voropy.mesh_tri.MeshTri(
            points,
            cells
            )

    tol = 1.0e-11

    # ce_ratios
    alpha = h - 1.0 / (4*h)
    beta = 1.0 / (4*h)
    assert _near_equal(
        mesh.get_ce_ratios_per_edge(),
        [alpha, beta, beta, beta, beta],
        tol
        )

    # control volumes
    alpha1 = 0.125 * (3*h - 1.0/(4*h))
    alpha2 = 0.125 * (h + 1.0 / (4*h))
    assert _near_equal(
        mesh.get_control_volumes(),
        [alpha1, alpha1, alpha2, alpha2],
        tol
        )

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [0.5*h, 0.5*h], tol)

    assert mesh.num_delaunay_violations() == 1

    return


@pytest.mark.parametrize(
        'a',  # edge length
        [1.0]
        )
def test_regular_tet0(a):
    points = numpy.array([
        [1.0, 0, 0],
        [-0.5,  numpy.sqrt(3.0) / 2.0, 0],
        [-0.5, -numpy.sqrt(3.0) / 2.0, 0],
        [0.0, 0.0, numpy.sqrt(2.0)],
        ]) / numpy.sqrt(3.0) * a
    cells = numpy.array([[0, 1, 2, 3]])
    mesh = voropy.mesh_tetra.MeshTetra(points, cells)

    mesh.show()
    mesh.show_edge(0)
    # from matplotlib import pyplot as plt
    # plt.show()

    tol = 1.0e-10

    z = a / numpy.sqrt(24.0)
    assert _near_equal(mesh.get_cell_circumcenters(), [0.0, 0.0, z], tol)

    mesh.compute_ce_ratios_geometric()
    assert _near_equal(mesh.circumcenter_face_distances, [z, z, z, z], tol)

    # covolume/edge length ratios
    # alpha = a / 12.0 / numpy.sqrt(2)
    alpha = a / 2 / numpy.sqrt(24) / numpy.sqrt(12)
    vals = mesh.get_ce_ratios()
    print(vals.shape)
    assert _near_equal(
        vals,
        [[
            [alpha, alpha, alpha],
            [alpha, alpha, alpha],
            [alpha, alpha, alpha],
            [alpha, alpha, alpha],
        ]],
        tol
        )

    # cell volumes
    vol = a**3 / 6.0 / numpy.sqrt(2)
    assert _near_equal(mesh.cell_volumes, [vol], tol)

    # control volumes
    val = vol / 4.0
    assert _near_equal(
        mesh.get_control_volumes(),
        [val, val, val, val],
        tol
        )

    return


# @pytest.mark.parametrize(
#         'a',  # basis edge length
#         [1.0]
#         )
# def test_regular_tet1_algebraic(a):
#     points = numpy.array([
#         [0, 0, 0],
#         [a, 0, 0],
#         [0, a, 0],
#         [0, 0, a]
#         ])
#     cells = numpy.array([[0, 1, 2, 3]])
#     tol = 1.0e-10
#
#     mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='algebraic')
#
#     assert _near_equal(
#         mesh.get_cell_circumcenters(),
#         [[a/2.0, a/2.0, a/2.0]],
#         tol
#         )
#
#     # covolume/edge length ratios
#     assert _near_equal(
#         mesh.get_ce_ratios_per_edge(),
#         [a/6.0, a/6.0, a/6.0, 0.0, 0.0, 0.0],
#         tol
#         )
#
#     # cell volumes
#     assert _near_equal(mesh.cell_volumes, [a**3/6.0], tol)
#
#     # control volumes
#     assert _near_equal(
#         mesh.get_control_volumes(),
#         [a**3/12.0, a**3/36.0, a**3/36.0, a**3/36.0],
#         tol
#         )
#
#     return


@pytest.mark.parametrize(
        'a',  # basis edge length
        [0.5, 1.0, 2.0]
        )
def test_regular_tet1_geometric(a):
    points = numpy.array([
        [0, 0, 0],
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
        ])
    cells = numpy.array([[0, 1, 2, 3]])
    tol = 1.0e-10

    mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='geometric')

    assert _near_equal(
        mesh.get_cell_circumcenters(),
        [a/2.0, a/2.0, a/2.0],
        tol
        )

    # covolume/edge length ratios
    assert _near_equal(
        mesh.get_ce_ratios(),
        [[
            [-a/24.0, -a/24.0, -a/24.0],
            [a/8.0, a/8.0, 0.0],
            [a/8.0, 0.0, a/8.0],
            [0.0, a/8.0, a/8.0],
        ]],
        tol
        )

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [a**3/6.0], tol)

    # control volumes
    assert _near_equal(
        mesh.get_control_volumes(),
        [a**3/8.0, a**3/72.0, a**3/72.0, a**3/72.0],
        tol
        )

    assert _near_equal(
        mesh.circumcenter_face_distances,
        [-0.5/numpy.sqrt(3)*a, 0.5*a, 0.5*a, 0.5*a],
        tol
        )

    return


# @pytest.mark.parametrize(
#         'h',
#         [1.0, 1.0e-2]
#         )
# def test_degenerate_tet0(h):
#     points = numpy.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0.5, 0.5, h],
#         ])
#     cells = numpy.array([[0, 1, 2, 3]])
#     mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='algebraic')
#
#     tol = 1.0e-12
#
#     z = 0.5 * h - 1.0 / (4*h)
#     assert _near_equal(
#         mesh.get_cell_circumcenters(),
#         [[0.5, 0.5, z]],
#         tol
#         )
#
#     # covolume/edge length ratios
#     print(h)
#     print(mesh.get_ce_ratios())
#     assert _near_equal(
#         mesh.get_ce_ratios(),
#         [[
#             [0.0, 0.0, 0.0],
#             [3.0/80.0, 3.0/40.0, 3.0/80.0],
#             [3.0/40.0, 3.0/80.0, 3.0/80.0],
#             [0.0, 1.0/16.0, 1.0/16.0],
#         ]],
#         tol
#         )
#     # [h / 6.0, h / 6.0, 0.0, -1.0/24/h, 1.0/12/h, 1.0/12/h],
#
#     # control volumes
#     ref = [
#         h / 18.0,
#         1.0/72.0 * (3*h - 1.0/(2*h)),
#         1.0/72.0 * (3*h - 1.0/(2*h)),
#         1.0/36.0 * (h + 1.0/(2*h))
#         ]
#     assert _near_equal(mesh.get_control_volumes(), ref, tol)
#
#     # cell volumes
#     assert _near_equal(mesh.cell_volumes, [h/6.0], tol)
#
#     return


# @pytest.mark.parametrize(
#         'h',
#         [1.0e-1]
#         )
# def test_degenerate_tet1(h):
#     points = numpy.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0.25, 0.25, h],
#         [0.25, 0.25, -h],
#         ])
#     cells = numpy.array([
#         [0, 1, 2, 3],
#         [0, 1, 2, 4]
#         ])
#     mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='algebraic')
#
#     total_vol = h / 3.0
#
#     _run(
#         mesh,
#         total_vol,
#         [0.18734818957173291, 77.0/720.0],
#         [2.420625, 5.0/6.0],
#         [1.0 / numpy.sqrt(2.0) / 30., 1.0/60.0]
#         )
#     return


def test_rectanglesmall():
    points = numpy.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
        ])
    cells = numpy.array([
        [0, 1, 2],
        [0, 2, 3]
        ])

    mesh = voropy.mesh_tri.MeshTri(points, cells)

    tol = 1.0e-14

    assert _near_equal(
        mesh.get_ce_ratios_per_edge(),
        [0.05, 0.0, 5.0, 5.0, 0.05],
        tol
        )
    assert _near_equal(
        mesh.get_control_volumes(),
        [2.5, 2.5, 2.5, 2.5],
        tol
        )
    assert _near_equal(mesh.cell_volumes, [5.0, 5.0], tol)
    assert mesh.num_delaunay_violations() == 0

    return


def test_cubesmall():
    points = numpy.array([
        [-0.5, -0.5, -5.0],
        [-0.5,  0.5, -5.0],
        [0.5, -0.5, -5.0],
        [-0.5, -0.5,  5.0],
        [0.5,  0.5, -5.0],
        [0.5,  0.5,  5.0],
        [-0.5,  0.5,  5.0],
        [0.5, -0.5,  5.0]
        ])
    cells = numpy.array([
        [0, 1, 2, 3],
        [1, 2, 4, 5],
        [1, 2, 3, 5],
        [1, 3, 5, 6],
        [2, 3, 5, 7]
        ])
    mesh = voropy.mesh_tetra.MeshTetra(points, cells)

    tol = 1.0e-14

    cv = numpy.ones(8) * 5.0 / 4.0
    cellvols = [5.0/3.0, 5.0/3.0, 10.0/3.0, 5.0/3.0, 5.0/3.0]

    assert _near_equal(mesh.get_control_volumes(), cv, tol)
    assert _near_equal(mesh.cell_volumes, cellvols, tol)

    cv_norms = [
        numpy.linalg.norm(cv, ord=2),
        numpy.linalg.norm(cv, ord=numpy.Inf),
        ]
    cellvol_norms = [
        numpy.linalg.norm(cellvols, ord=2),
        numpy.linalg.norm(cellvols, ord=numpy.Inf),
        ]
    _run(
        mesh,
        10.0,
        cv_norms,
        [28.095851618771825, 1.25],
        cellvol_norms,
        )
    return


# def test_arrow3d():
#     nodes = numpy.array([
#         [0.0,  0.0, 0.0],
#         [2.0, -1.0, 0.0],
#         [2.0,  0.0, 0.0],
#         [2.0,  1.0, 0.0],
#         [0.5,  0.0, -0.9],
#         [0.5,  0.0, 0.9]
#         ])
#     cellsNodes = numpy.array([
#         [1, 2, 4, 5],
#         [2, 3, 4, 5],
#         [0, 1, 4, 5],
#         [0, 3, 4, 5]
#         ])
#     mesh = voropy.mesh_tetra.MeshTetra(nodes, cellsNodes, mode='algebraic')
#
#     # # pull this to see what a negative ce_ratio looks like
#     # mesh.show()
#     # mesh.show_edge(12)
#     # from matplotlib import pyplot as plt
#     # plt.show()
#
#     _run(
#         mesh,
#         1.2,
#         [numpy.sqrt(0.30104), 0.354],
#         [14.281989026063275, 2.4],
#         [numpy.sqrt(0.45), 0.45]
#         )
#
#     assert mesh.num_delaunay_violations() == 2
#
#     return


def test_tetrahedron():
    filename = fetch_data.download_mesh(
            'tetrahedron.msh',
            '27a5d7e102e6613a1e58629c252cb293'
            )
    mesh, _, _, _ = voropy.read(filename)
    # mesh.show_edge(54)
    _run(
        mesh,
        64.1500299099584,
        [17.07120343309435, 7.5899731568813653],
        # [33.87181266432331, 1.6719101545282922],
        [6.898476155562042, 0.34400453539215242],
        [11.571692332290635, 2.9699087921277054]
        )
    return


def test_pacman():
    filename = fetch_data.download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    mesh, _, _, _ = voropy.read(filename, flat_cell_correction='boundary')

    _run(
        mesh,
        73.64573933105898,
        [3.5908322974649631, 0.26638548094154707],
        [669.3501944927655, 1.8142648825759053],
        [2.6213234038171014, 0.13841739494523228]
        )

    assert mesh.num_delaunay_violations() == 0

    return


def test_shell():
    points = numpy.array([
        [0.0,  0.0,  1.0],
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0]
        ])
    cells = numpy.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 1, 4]
        ])
    mesh = voropy.mesh_tri.MeshTri(points, cells)
    _run(
        mesh,
        2 * numpy.sqrt(3),
        [2 * numpy.sqrt(2.0/3.0), 2.0/numpy.sqrt(3.0)],
        [5.0 / 3.0, numpy.sqrt(1.0 / 3.0)],
        [numpy.sqrt(3.0), numpy.sqrt(3.0) / 2.0]
        )

    assert mesh.num_delaunay_violations() == 0

    return


def test_sphere():
    filename = fetch_data.download_mesh(
            'sphere.msh',
            '70a5dbf79c3b259ed993458ff4aa2e93'
            )
    mesh, _, _, _ = voropy.read(filename)
    _run(
        mesh,
        12.273645818711595,
        [1.0177358705967492, 0.10419690304323895],
        [729.9372898474035, 3.2706494490659366],
        [0.72653362732751214, 0.05350373815413411]
        )

    # assertEqual(mesh.num_delaunay_violations(), 60)

    return


# def test_toy_algebraic():
#     filename = fetch_data.download_mesh(
#         'toy.msh',
#         '1d125d3fa9f373823edd91ebae5f7a81'
#         )
#     mesh, _, _, _ = voropy.read(filename)
#
#     # Even if the input data has only a small error, the error in the
#     # ce_ratios can be magnitudes larger. This is demonstrated here: Take
#     # the same mesh from two different source files with a differnce of
#     # the order of 1e-16. The ce_ratios differ by up to 1e-7.
#     if False:
#         print(mesh.cells.keys())
#         pts = mesh.node_coords.copy()
#         pts += 1.0e-16 * numpy.random.rand(pts.shape[0], pts.shape[1])
#         mesh2 = voropy.mesh_tetra.MeshTetra(pts, mesh.cells['nodes'])
#         #
#         diff_coords = mesh.node_coords - mesh2.node_coords
#         max_diff_coords = max(diff_coords.flatten())
#         print('||coords_1 - coords_2||_inf  =  %e' % max_diff_coords)
#         diff_ce_ratios = mesh.get_ce_ratios_per_edge() - mesh2.ce_ratios
#         print(
#             '||ce_ratios_1 - ce_ratios_2||_inf  =  %e'
#             % max(diff_ce_ratios)
#             )
#         from matplotlib import pyplot as plt
#         plt.figure()
#         n = len(mesh.get_ce_ratios_per_edge())
#         plt.semilogy(range(n), diff_ce_ratios)
#         plt.show()
#         exit(1)
#
#     _run(
#         mesh,
#         volume=9.3875504672601107,
#         convol_norms=[0.20348466631551548, 0.010271101930468585],
#         ce_ratio_norms=[396.4116343366758, 3.4508458933423918],
#         cellvol_norms=[0.091903119589148916, 0.0019959463063558944],
#         tol=1.0e-6
#         )
#     return


def test_toy_geometric():
    filename = fetch_data.download_mesh(
        'toy.msh',
        '1d125d3fa9f373823edd91ebae5f7a81'
        )
    mesh, _, _, _ = voropy.read(filename)

    mesh = voropy.mesh_tetra.MeshTetra(
        mesh.node_coords,
        mesh.cells['nodes'],
        mode='geometric'
        )

    _run(
        mesh,
        volume=9.3875504672601107,
        convol_norms=[0.20175742659663737, 0.0093164692200450819],
        ce_ratio_norms=[76.7500558132087, 0.34008519731077325],
        cellvol_norms=[0.091903119589148916, 0.0019959463063558944],
        tol=1.0e-6
        )
    return
