import pathlib
from math import fsum

import numpy as np
import pytest

import meshplex

from .helpers import is_near_equal, run

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("a", [0.5, 1.0, 1.33])  # edge length
def test_regular_tet0(a):
    points = (
        a
        * np.array(
            [
                [1.0, 0, 0],
                [-0.5, +np.sqrt(3.0) / 2.0, 0],
                [-0.5, -np.sqrt(3.0) / 2.0, 0],
                [0.0, 0.0, np.sqrt(2.0)],
            ]
        )
        / np.sqrt(3.0)
    )
    cells = np.array([[0, 1, 2, 3]])
    mesh = meshplex.MeshTetra(points, cells.copy())
    # test __repr__
    print(mesh)

    assert np.all(mesh.cells("points") == cells)

    mesh.show()
    mesh.show_edge(0)

    tol = 1.0e-14

    z = a / np.sqrt(24.0)
    assert is_near_equal(mesh.cell_circumcenters, [0.0, 0.0, z], tol)

    assert is_near_equal(mesh.circumcenter_facet_distances, [z, z, z, z], tol)

    # covolume/edge length ratios
    # alpha = a / 12.0 / np.sqrt(2)
    alpha = a / 2 / np.sqrt(24) / np.sqrt(12)
    vals = mesh.ce_ratios
    assert is_near_equal(
        vals,
        [
            [
                [alpha, alpha, alpha],
                [alpha, alpha, alpha],
                [alpha, alpha, alpha],
                [alpha, alpha, alpha],
            ]
        ],
        tol,
    )

    # cell volumes
    vol = a ** 3 / 6.0 / np.sqrt(2)
    assert is_near_equal(mesh.cell_volumes, [vol], tol)

    # control volumes
    val = vol / 4.0
    assert is_near_equal(mesh.control_volumes, [val, val, val, val], tol)

    # inradius
    # side_area = np.sqrt(3) / 4 * a ** 2
    # vol = a ** 3 / 6.0 / np.sqrt(2)
    # inradius = 3 * vol / (4 * side_area)
    inradius = a * np.sqrt(6) / 12
    assert is_near_equal(mesh.cell_inradius, [inradius], tol)

    # circumradius
    circumradius = a * np.sqrt(6) / 4
    assert is_near_equal(mesh.cell_circumradius, [circumradius], tol)

    # cell quality
    assert is_near_equal(mesh.q_radius_ratio, [1.0], tol)

    assert is_near_equal(mesh.cell_barycenters, mesh.cell_centroids, tol)
    assert is_near_equal(mesh.cell_barycenters, [[0.0, 0.0, a * np.sqrt(6) / 12]], tol)

    assert is_near_equal(mesh.cell_incenters, [[0.0, 0.0, a * np.sqrt(6) / 12]], tol)

    assert is_near_equal(mesh.q_min_sin_dihedral_angles, [1.0], tol)

    assert is_near_equal(mesh.q_vol_rms_edgelength3, [1.0], tol)


# @pytest.mark.parametrize(
#         'a',  # basis edge length
#         [1.0]
#         )
# def test_regular_tet1_algebraic(a):
#     points = np.array([
#         [0, 0, 0],
#         [a, 0, 0],
#         [0, a, 0],
#         [0, 0, a]
#         ])
#     cells = np.array([[0, 1, 2, 3]])
#     tol = 1.0e-14
#
#     mesh = meshplex.MeshTetra(points, cells, mode='algebraic')
#
#     assert is_near_equal(
#         mesh.cell_circumcenters,
#         [[a/2.0, a/2.0, a/2.0]],
#         tol
#         )
#
#     # covolume/edge length ratios
#     assert is_near_equal(
#         mesh.ce_ratios_per_edge,
#         [a/6.0, a/6.0, a/6.0, 0.0, 0.0, 0.0],
#         tol
#         )
#
#     # cell volumes
#     assert is_near_equal(mesh.cell_volumes, [a**3/6.0], tol)
#
#     # control volumes
#     assert is_near_equal(
#         mesh.control_volumes,
#         [a**3/12.0, a**3/36.0, a**3/36.0, a**3/36.0],
#         tol
#         )


@pytest.mark.parametrize(
    "a",
    [
        # 0.5,
        1.0,
        # 2.0
    ],
)  # basis edge length
def test_unit_tetrahedron_geometric(a):
    points = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]])
    cells = np.array([[0, 1, 2, 3]])
    tol = 1.0e-14

    mesh = meshplex.MeshTetra(points, cells)

    assert all((mesh.cells("points") == cells).flat)

    assert is_near_equal(mesh.cell_circumcenters, [a / 2.0, a / 2.0, a / 2.0], tol)

    # covolume/edge length ratios
    ref = np.array(
        [
            [[-a / 24.0], [a / 8.0], [a / 8.0], [0.0]],
            [[-a / 24.0], [a / 8.0], [0.0], [a / 8.0]],
            [[-a / 24.0], [0.0], [a / 8.0], [a / 8.0]],
        ]
    )
    assert is_near_equal(mesh.ce_ratios, ref, tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [a ** 3 / 6.0], tol)

    # control volumes
    assert is_near_equal(
        mesh.control_volumes,
        [a ** 3 / 8.0, a ** 3 / 72.0, a ** 3 / 72.0, a ** 3 / 72.0],
        tol,
    )

    assert is_near_equal(
        mesh.circumcenter_facet_distances.T,
        [-0.5 / np.sqrt(3) * a, 0.5 * a, 0.5 * a, 0.5 * a],
        tol,
    )

    # inradius
    ref = a * 0.2113248654051872
    assert is_near_equal(mesh.cell_inradius, [ref], tol)

    # circumradius
    ref = a * np.sqrt(3) / 2
    assert is_near_equal(mesh.cell_circumradius, [ref], tol)

    # cell quality
    ref = 7.320508075688774e-01
    assert is_near_equal(mesh.q_radius_ratio, [ref], tol)

    assert is_near_equal(mesh.q_min_sin_dihedral_angles, [np.sqrt(3) / 2], tol)

    assert is_near_equal(mesh.q_vol_rms_edgelength3, [4 * np.sqrt(3) / 9], tol)


def test_regular_tet1_geometric_order():
    a = 1.0
    points = np.array([[0, 0, a], [0, 0, 0], [a, 0, 0], [0, a, 0]])
    cells = np.array([[0, 1, 2, 3]])
    tol = 1.0e-14

    mesh = meshplex.MeshTetra(points, cells)

    assert all((mesh.cells("points") == [0, 1, 2, 3]).flat)

    assert is_near_equal(mesh.cell_circumcenters, [a / 2.0, a / 2.0, a / 2.0], tol)

    # covolume/edge length ratios
    ref = np.array(
        [
            [[0.0], [-a / 24.0], [a / 8.0], [a / 8.0]],
            [[a / 8.0], [-a / 24.0], [a / 8.0], [0.0]],
            [[a / 8.0], [-a / 24.0], [0.0], [a / 8.0]],
        ]
    )
    assert is_near_equal(mesh.ce_ratios, ref, tol)

    # cell volumes
    assert is_near_equal(mesh.cell_volumes, [a ** 3 / 6.0], tol)

    # control volumes
    assert is_near_equal(
        mesh.control_volumes,
        [a ** 3 / 72.0, a ** 3 / 8.0, a ** 3 / 72.0, a ** 3 / 72.0],
        tol,
    )

    assert is_near_equal(
        mesh.circumcenter_facet_distances.T,
        [0.5 * a, -0.5 / np.sqrt(3) * a, 0.5 * a, 0.5 * a],
        tol,
    )


# @pytest.mark.parametrize(
#         'h',
#         [1.0, 1.0e-2]
#         )
# def test_degenerate_tet0(h):
#     points = np.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0.5, 0.5, h],
#         ])
#     cells = np.array([[0, 1, 2, 3]])
#     mesh = meshplex.MeshTetra(points, cells, mode='algebraic')
#
#     tol = 1.0e-14
#
#     z = 0.5 * h - 1.0 / (4*h)
#     assert is_near_equal(
#         mesh.cell_circumcenters,
#         [[0.5, 0.5, z]],
#         tol
#         )
#
#     # covolume/edge length ratios
#     print(h)
#     print(mesh.ce_ratios)
#     assert is_near_equal(
#         mesh.ce_ratios,
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
#     assert is_near_equal(mesh.control_volumes, ref, tol)
#
#     # cell volumes
#     assert is_near_equal(mesh.cell_volumes, [h/6.0], tol)


# @pytest.mark.parametrize(
#         'h',
#         [1.0e-1]
#         )
# def test_degenerate_tet1(h):
#     points = np.array([
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0.25, 0.25, h],
#         [0.25, 0.25, -h],
#         ])
#     cells = np.array([
#         [0, 1, 2, 3],
#         [0, 1, 2, 4]
#         ])
#     mesh = meshplex.MeshTetra(points, cells, mode='algebraic')
#
#     total_vol = h / 3.0
#
#     run(
#         mesh,
#         total_vol,
#         [0.18734818957173291, 77.0/720.0],
#         [2.420625, 5.0/6.0],
#         [1.0 / np.sqrt(2.0) / 30., 1.0/60.0]
#         )


def test_cubesmall():
    points = np.array(
        [
            [-0.5, -0.5, -5.0],
            [-0.5, +0.5, -5.0],
            [+0.5, -0.5, -5.0],
            [-0.5, -0.5, +5.0],
            [+0.5, +0.5, -5.0],
            [+0.5, +0.5, +5.0],
            [-0.5, +0.5, +5.0],
            [+0.5, -0.5, +5.0],
        ]
    )
    cells = np.array(
        [[0, 1, 2, 3], [1, 2, 4, 5], [1, 2, 3, 5], [1, 3, 5, 6], [2, 3, 5, 7]]
    )
    mesh = meshplex.MeshTetra(points, cells)

    tol = 1.0e-14

    cv = np.ones(8) * 1.25
    cellvols = [5.0 / 3.0, 5.0 / 3.0, 10.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0]

    assert is_near_equal(mesh.control_volumes, cv, tol)
    assert is_near_equal(mesh.cell_volumes, cellvols, tol)

    cv_norms = [np.linalg.norm(cv, ord=2), np.linalg.norm(cv, ord=np.Inf)]
    cellvol_norms = [
        np.linalg.norm(cellvols, ord=2),
        np.linalg.norm(cellvols, ord=np.Inf),
    ]
    run(mesh, 10.0, cv_norms, [28.095851618771825, 1.25], cellvol_norms)


def test_arrow3d():
    points = np.array(
        [
            [+0.0, +0.0, +0.0],
            [+2.0, -1.0, +0.0],
            [+2.0, +0.0, +0.0],
            [+2.0, +1.0, +0.0],
            [+0.5, +0.0, -0.9],
            [+0.5, +0.0, +0.9],
        ]
    )
    cellsNodes = np.array([[1, 2, 4, 5], [2, 3, 4, 5], [0, 1, 4, 5], [0, 3, 4, 5]])
    mesh = meshplex.MeshTetra(points, cellsNodes)

    run(
        mesh,
        1.2,
        [0.58276428453480922, 0.459],
        [0.40826901831985885, 0.2295],
        [np.sqrt(0.45), 0.45],
    )

    assert mesh.num_delaunay_violations == 2


def test_tetrahedron():
    mesh = meshplex.read(this_dir / "meshes" / "tetrahedron.vtk")

    run(
        mesh,
        64.1500299099584,
        [16.308991595922095, 7.0264329635751395],
        [6.898476155562041, 0.34400453539215237],
        [11.571692332290635, 2.9699087921277054],
    )


# def test_toy_algebraic():
#     filename = download_mesh(
#         'toy.vtk',
#         'f48abda972822bab224b91a74d695573'
#         )
#     mesh, _, _, _ = meshplex.read(filename)
#
#     # Even if the input data has only a small error, the error in the
#     # ce_ratios can be magnitudes larger. This is demonstrated here: Take
#     # the same mesh from two different source files with a differnce of
#     # the order of 1e-16. The ce_ratios differ by up to 1e-7.
#     if False:
#         print(mesh.cells.keys())
#         pts = mesh.points.copy()
#         pts += 1.0e-16 * np.random.rand(pts.shape[0], pts.shape[1])
#         mesh2 = meshplex.MeshTetra(pts, mesh.cells['points'])
#         #
#         diff_coords = mesh.points - mesh2.points
#         max_diff_coords = max(diff_coords.flatten())
#         print('||coords_1 - coords_2||_inf  =  %e' % max_diff_coords)
#         diff_ce_ratios = mesh.ce_ratios_per_edge - mesh2.ce_ratios
#         print(
#             '||ce_ratios_1 - ce_ratios_2||_inf  =  %e'
#             % max(diff_ce_ratios)
#             )
#         from matplotlib import pyplot as plt
#         plt.figure()
#         n = len(mesh.ce_ratios_per_edge)
#         plt.semilogy(range(n), diff_ce_ratios)
#         plt.show()
#         exit(1)
#
#     run(
#         mesh,
#         volume=9.3875504672601107,
#         convol_norms=[0.20348466631551548, 0.010271101930468585],
#         ce_ratio_norms=[396.4116343366758, 3.4508458933423918],
#         cellvol_norms=[0.091903119589148916, 0.0019959463063558944],
#         tol=1.0e-6
#         )


def test_toy_geometric():
    mesh = meshplex.read(this_dir / "meshes" / "toy.vtk")

    mesh = meshplex.MeshTetra(mesh.points, mesh.cells("points"))

    run(
        mesh,
        volume=9.3875504672601107,
        convol_norms=[0.20175742659663737, 0.0093164692200450819],
        ce_ratio_norms=[13.497977312281323, 0.42980191511570004],
        cellvol_norms=[0.091903119589148916, 0.0019959463063558944],
        tol=1.0e-6,
    )

    cc = mesh.cell_circumcenters
    cc_norm_2 = fsum(cc.flat)
    cc_norm_inf = max(cc.flat)
    assert abs(cc_norm_2 - 1103.7038287583791) < 1.0e-11
    assert abs(cc_norm_inf - 3.4234008596539662) < 1.0e-11


def test_signed_volume():
    mesh = meshplex.read(this_dir / "meshes" / "toy.vtk")

    vols = meshplex.get_signed_simplex_volumes(mesh.cells("points"), mesh.points)
    # TODO see if this gets more accurate
    assert np.all(abs(abs(vols) - mesh.cell_volumes) < 1.0e-11 * mesh.cell_volumes)


def test_show_cell(render=False):
    pytest.importorskip("vtk")
    # filename = download_mesh("toy.vtk", "f48abda972822bab224b91a74d695573")
    # mesh = meshplex.read(filename)

    points = (
        np.array(
            [
                [1.0, 0.0, -1.0 / np.sqrt(8)],
                [-0.5, +np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
                [-0.5, -np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
                [0.0, 0.0, np.sqrt(2.0) - 1.0 / np.sqrt(8)],
            ]
        )
        / np.sqrt(3.0)
    )

    # points = np.array(
    #     [
    #         [1.0, 0.0, -1.0 / np.sqrt(8)],
    #         [-0.5, +np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
    #         [-0.5, -np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
    #         [0.0, 0.5, 1.0],
    #     ]
    # ) / np.sqrt(3.0)
    # points = np.array(
    #     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # )

    cells = [[0, 1, 2, 3]]
    mesh = meshplex.MeshTetra(points, cells)
    mesh.show_cell(
        0,
        barycenter_rgba=(1, 0, 0, 1.0),
        circumcenter_rgba=(0.1, 0.1, 0.1, 1.0),
        circumsphere_rgba=(0, 1, 0, 1.0),
        incenter_rgba=(1, 0, 1, 1.0),
        insphere_rgba=(1, 0, 1, 1.0),
        face_circumcenter_rgba=(0, 0, 1, 1.0),
        control_volume_boundaries_rgba=(1.0, 0.0, 0.0, 1.0),
        line_width=3.0,
        close=True,
        render=render,
    )


if __name__ == "__main__":
    test_show_cell(render=True)
    # test_regular_tet0(0.5)
    # test_toy_geometric()
