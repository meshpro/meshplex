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
    ref = numpy.array([
        [[-a/24.0], [a/8.0], [a/8.0], [0.0]],
        [[-a/24.0], [a/8.0], [0.0], [a/8.0]],
        [[-a/24.0], [0.0], [a/8.0], [a/8.0]],
        ])
    assert _near_equal(mesh.get_ce_ratios(), ref, tol)

    # cell volumes
    assert _near_equal(mesh.cell_volumes, [a**3/6.0], tol)

    # control volumes
    assert _near_equal(
        mesh.get_control_volumes(),
        [a**3/8.0, a**3/72.0, a**3/72.0, a**3/72.0],
        tol
        )

    assert _near_equal(
        mesh.circumcenter_face_distances.T,
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

    cv = numpy.ones(8) * 1.25
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


def test_arrow3d():
    nodes = numpy.array([
        [0.0,  0.0, 0.0],
        [2.0, -1.0, 0.0],
        [2.0,  0.0, 0.0],
        [2.0,  1.0, 0.0],
        [0.5,  0.0, -0.9],
        [0.5,  0.0, 0.9]
        ])
    cellsNodes = numpy.array([
        [1, 2, 4, 5],
        [2, 3, 4, 5],
        [0, 1, 4, 5],
        [0, 3, 4, 5]
        ])
    mesh = voropy.mesh_tetra.MeshTetra(nodes, cellsNodes)

    _run(
        mesh,
        1.2,
        [0.58276428453480922, 0.459],
        [0.40826901831985885, 0.2295],
        [numpy.sqrt(0.45), 0.45]
        )

    assert mesh.num_delaunay_violations() == 2

    return


def test_tetrahedron():
    filename = fetch_data.download_mesh(
            'tetrahedron.msh',
            '27a5d7e102e6613a1e58629c252cb293'
            )
    mesh, _, _, _ = voropy.read(filename)

    print(mesh.cell_volumes)
    print(sum(mesh.cell_volumes))
    print(sum(mesh.get_control_volumes()))
    # mesh.show_edge(54)
    _run(
        mesh,
        64.1500299099584,
        [16.308991595922095, 7.0264329635751395],
        [15.622787827281265, 0.65212756309069242],
        [11.571692332290635, 2.9699087921277054]
        )
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
        ce_ratio_norms=[13.497977312281323, 0.42980191511570004],
        cellvol_norms=[0.091903119589148916, 0.0019959463063558944],
        tol=1.0e-6
        )

    cc = mesh.get_cell_circumcenters()
    cc_norm_2 = fsum(cc.flat)
    cc_norm_inf = max(cc.flat)
    assert abs(cc_norm_2 - 1103.7038287583791) < 1.0e-14
    assert abs(cc_norm_inf - 3.4234008596539662) < 1.0e-14
    return
