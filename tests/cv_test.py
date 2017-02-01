# -*- coding: utf-8 -*-
#
import fetch_data
import voropy

import os
from math import fsum
import numpy
import unittest


class TestVolumes(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(
            self,
            mesh,
            volume,
            cv_norms, covol_norms, cellvol_norms,
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
        self.assertAlmostEqual(volume, total_cellvolume, delta=tol * volume)
        norm = numpy.linalg.norm(mesh.cell_volumes, ord=2)
        self.assertAlmostEqual(cellvol_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.cell_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(cellvol_norms[1], norm, delta=tol)

        # If everything is Delaunay and the boundary elements aren't flat, the
        # volume of the domain is given by
        #   1/n * edge_lengths * ce_ratios.
        # Unfortunately, this isn't always the case.
        # ```
        # total_ce_ratio = fsum(mesh.edge_lengths**2 * mesh.ce_ratios / dim)
        # self.assertAlmostEqual(volume, total_ce_ratio, delta=tol * volume)
        # ```
        # Check ce_ratio norms.
        alpha = fsum(mesh.ce_ratios**2)
        self.assertAlmostEqual(covol_norms[0], alpha, delta=tol)
        alpha = max(abs(mesh.ce_ratios))
        self.assertAlmostEqual(covol_norms[1], alpha, delta=tol)

        # Check the volume by summing over the absolute value of the
        # control volumes.
        vol = fsum(mesh.control_volumes)
        self.assertAlmostEqual(volume, vol, delta=tol * volume)
        # Check control volume norms.
        norm = numpy.linalg.norm(mesh.control_volumes, ord=2)
        self.assertAlmostEqual(cv_norms[0], norm, delta=tol)
        norm = numpy.linalg.norm(mesh.control_volumes, ord=numpy.Inf)
        self.assertAlmostEqual(cv_norms[1], norm, delta=tol)

        return

    def test_regular_tri(self):
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
            ])
        cells = numpy.array([[0, 1, 2]])
        mesh = voropy.mesh_tri.MeshTri(points, cells)

        tol = 1.0e-14

        # ce_ratios
        self.assertAlmostEqual(mesh.ce_ratios[0], 0.5, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], 0.5, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], 0, delta=tol)

        # control volumes
        self.assertAlmostEqual(mesh.control_volumes[0], 0.25, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], 0.125, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], 0.125, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], 0.5, delta=tol)

        # centroids
        self.assertAlmostEqual(mesh.centroids[0][0], 0.25, delta=tol)
        self.assertAlmostEqual(mesh.centroids[0][1], 0.25, delta=tol)
        self.assertAlmostEqual(mesh.centroids[0][2], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[1][0], 2.0/3.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[1][1], 1.0/6.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[1][2], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[2][0], 1.0/6.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[2][1], 2.0/3.0, delta=tol)
        self.assertAlmostEqual(mesh.centroids[2][2], 0.0, delta=tol)

        self.assertEqual(mesh.num_delaunay_violations(), 0)

        # edge_cells
        edge_cells = mesh.compute_edge_cells()
        self.assertEqual(edge_cells, [[0], [0], [0]])

        return

    # def test_degenerate_small0(self):
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
    #     self.assertAlmostEqual(mesh.ce_ratios[0], alpha, delta=tol)
    #     self.assertAlmostEqual(mesh.ce_ratios[1], beta, delta=tol)
    #     self.assertAlmostEqual(mesh.ce_ratios[2], beta, delta=tol)

    #     # control volumes
    #     alpha1 = 0.0625 * (3*h - 1.0/(4*h))
    #     alpha2 = 0.125 * (h + 1.0 / (4*h))
    #     self.assertAlmostEqual(mesh.control_volumes[0], alpha1, delta=tol)
    #     self.assertAlmostEqual(mesh.control_volumes[1], alpha1, delta=tol)
    #     self.assertAlmostEqual(mesh.control_volumes[2], alpha2, delta=tol)

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
    #     self.assertAlmostEqual(mesh.centroids[0][0], 0.416668000016, delta=tol)
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

    def test_degenerate_small0b(self):
        h = 1.0e-3
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, h, 0.0],
            ])
        cells = numpy.array([[0, 1, 2]])
        mesh = voropy.mesh_tri.MeshTri(
                points,
                cells
                )

        tol = 1.0e-14

        # edge lengths
        edge_length = numpy.sqrt(0.5**2 + h**2)
        self.assertAlmostEqual(mesh.edge_lengths[0], 1.0, delta=tol)
        self.assertAlmostEqual(mesh.edge_lengths[1], edge_length, delta=tol)
        self.assertAlmostEqual(mesh.edge_lengths[2], edge_length, delta=tol)

        # ce_ratios
        ce = h
        self.assertAlmostEqual(mesh.ce_ratios[0], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], ce, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], ce, delta=tol)

        # control volumes
        cv = ce * edge_length
        alpha = 0.25 * edge_length * cv
        beta = 0.5*h - 2*alpha
        self.assertAlmostEqual(mesh.control_volumes[0], alpha, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], alpha, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], beta, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], 0.5 * h, delta=tol)

        # surface areas
        g = numpy.sqrt((0.5 * edge_length)**2 + (ce * edge_length)**2)
        alpha = 0.5 * edge_length + g
        beta = edge_length + (1.0 - 2*g)
        self.assertAlmostEqual(mesh.surface_areas[0], alpha, delta=tol)
        self.assertAlmostEqual(mesh.surface_areas[1], alpha, delta=tol)
        self.assertAlmostEqual(mesh.surface_areas[2], beta, delta=tol)

        # centroids
        alpha = 1.0 / 6000.0
        self.assertAlmostEqual(mesh.centroids[0][0], 0.166667, delta=tol)
        self.assertAlmostEqual(mesh.centroids[0][1], alpha, delta=tol)
        self.assertAlmostEqual(mesh.centroids[0][2], 0.0, delta=tol)

        self.assertAlmostEqual(mesh.centroids[1][0], 0.833333, delta=tol)
        self.assertAlmostEqual(mesh.centroids[1][1], alpha, delta=tol)
        self.assertAlmostEqual(mesh.centroids[1][2], 0.0, delta=tol)

        gamma = 0.00038888918518558031
        self.assertAlmostEqual(mesh.centroids[2][0], 0.5, delta=tol)
        self.assertAlmostEqual(mesh.centroids[2][1], gamma, delta=tol)
        self.assertAlmostEqual(mesh.centroids[2][2], 0.0, delta=tol)

        self.assertEqual(mesh.num_delaunay_violations(), 0)
        return

    def test_degenerate_small1(self):
        h = 1.0e-3
        a = 0.3
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [a, h, 0.0],
            ])
        cells = numpy.array([[0, 1, 2]])
        mesh = voropy.mesh_tri.MeshTri(
                points,
                cells
                )

        tol = 1.0e-14

        # edge lengths
        el1 = numpy.sqrt(a**2 + h**2)
        el2 = numpy.sqrt((1.0 - a)**2 + h**2)
        self.assertAlmostEqual(mesh.edge_lengths[0], 1.0, delta=tol)
        self.assertAlmostEqual(mesh.edge_lengths[1], el1, delta=tol)
        self.assertAlmostEqual(mesh.edge_lengths[2], el2, delta=tol)

        # ce_ratios
        ce1 = 0.5 * h / a
        ce2 = 0.5 * h / (1.0 - a)
        self.assertAlmostEqual(mesh.ce_ratios[0], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], ce1, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], ce2, delta=tol)

        # control volumes
        cv1 = ce1 * el1
        alpha1 = 0.25 * el1 * cv1
        cv2 = ce2 * el2
        alpha2 = 0.25 * el2 * cv2
        beta = 0.5*h - (alpha1 + alpha2)
        self.assertAlmostEqual(mesh.control_volumes[0], alpha1, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], alpha2, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], beta, delta=tol)
        self.assertAlmostEqual(sum(mesh.control_volumes), 0.5 * h, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], 0.5 * h, delta=tol)

        # surface areas
        b1 = numpy.sqrt((0.5*el1)**2 + cv1**2)
        alpha0 = b1 + 0.5*el1
        b2 = numpy.sqrt((0.5*el2)**2 + cv2**2)
        alpha1 = b2 + 0.5*el2
        total = 1.0 + el1 + el2
        alpha2 = total - alpha0 - alpha1
        self.assertAlmostEqual(mesh.surface_areas[0], alpha0, delta=tol)
        self.assertAlmostEqual(mesh.surface_areas[1], alpha1, delta=tol)
        self.assertAlmostEqual(mesh.surface_areas[2], alpha2, delta=tol)

        self.assertEqual(mesh.num_delaunay_violations(), 0)
        return

    def test_degenerate_small2(self):
        h = 1.0e-2
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

        tol = 1.0e-12

        # ce_ratios
        alpha = h - 1.0 / (4*h)
        beta = 1.0 / (4*h)
        self.assertAlmostEqual(mesh.ce_ratios[0], alpha, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], beta, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], beta, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], beta, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4], beta, delta=tol)

        # control volumes
        alpha1 = 0.125 * (3*h - 1.0/(4*h))
        alpha2 = 0.125 * (h + 1.0 / (4*h))
        self.assertAlmostEqual(mesh.control_volumes[0], alpha1, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], alpha1, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], alpha2, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[3], alpha2, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], 0.5 * h, delta=tol)
        self.assertAlmostEqual(mesh.cell_volumes[1], 0.5 * h, delta=tol)

        self.assertEqual(mesh.num_delaunay_violations(), 1)

        # edge_cells
        edge_cells = mesh.compute_edge_cells()
        self.assertEqual(edge_cells, [[0, 1], [0], [1], [0], [1]])

        return

    def test_regular_tet0(self):
        a = 1.0  # edge length

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

        self.assertAlmostEqual(mesh.cell_circumcenters[0][0], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][1], 0.0, delta=tol)
        z = a / numpy.sqrt(24.0)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][2], z, delta=tol)

        # covolume/edge length ratios
        val = a / 12.0 / numpy.sqrt(2)
        self.assertAlmostEqual(mesh.ce_ratios[0], val, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], val, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], val, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], val, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4], val, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[5], val, delta=tol)

        # cell volumes
        vol = a**3 / 6.0 / numpy.sqrt(2)
        self.assertAlmostEqual(mesh.cell_volumes[0], vol, delta=tol)

        # control volumes
        val = vol / 4.0
        self.assertAlmostEqual(mesh.control_volumes[0], val, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], val, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], val, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[3], val, delta=tol)

        return

    def test_regular_tet1_algebraic(self):
        a = 1.0  # basis edge length

        points = numpy.array([
            [0, 0, 0],
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
            ])
        cells = numpy.array([[0, 1, 2, 3]])
        tol = 1.0e-10

        mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='algebraic')

        self.assertAlmostEqual(mesh.cell_circumcenters[0][0], a/2.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][1], a/2.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][2], a/2.0, delta=tol)

        # covolume/edge length ratios
        self.assertAlmostEqual(mesh.ce_ratios[0], a/6.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], a/6.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], a/6.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[5], 0.0, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], a**3/6.0, delta=tol)

        # control volumes
        self.assertAlmostEqual(mesh.control_volumes[0], a**3/12.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], a**3/36.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], a**3/36.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[3], a**3/36.0, delta=tol)

        return

    def test_regular_tet1_geometric(self):
        a = 1.0  # basis edge length

        points = numpy.array([
            [0, 0, 0],
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
            ])
        cells = numpy.array([[0, 1, 2, 3]])
        tol = 1.0e-10

        mesh = voropy.mesh_tetra.MeshTetra(points, cells, mode='geometric')

        self.assertAlmostEqual(mesh.cell_circumcenters[0][0], a/2.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][1], a/2.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][2], a/2.0, delta=tol)

        # covolume/edge length ratios
        self.assertAlmostEqual(mesh.ce_ratios[0], a/4.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], a/4.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], a/4.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], -a/24.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4], -a/24.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[5], -a/24.0, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], a**3/6.0, delta=tol)

        # control volumes
        self.assertAlmostEqual(mesh.control_volumes[0], a**3/8.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], a**3/72.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], a**3/72.0, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[3], a**3/72.0, delta=tol)

        return

    def test_degenerate_tet0(self):
        h = 1.0e-2
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.5, 0.5, h],
            ])
        cells = numpy.array([[0, 1, 2, 3]])
        mesh = voropy.mesh_tetra.MeshTetra(points, cells)

        tol = 1.0e-7

        self.assertAlmostEqual(mesh.cell_circumcenters[0][0], 0.5, delta=tol)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][1], 0.5, delta=tol)
        z = 0.5 * h - 1.0 / (4*h)
        self.assertAlmostEqual(mesh.cell_circumcenters[0][2], z, delta=tol)

        # covolume/edge length ratios
        self.assertAlmostEqual(mesh.ce_ratios[0], h / 6.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], h / 6.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], -1.0/24.0 / h, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4],  1.0/12.0 / h, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[5],  1.0/12.0 / h, delta=tol)

        # control volumes
        self.assertAlmostEqual(mesh.control_volumes[0], h / 18.0, delta=tol)
        self.assertAlmostEqual(
                mesh.control_volumes[1],
                1.0/72.0 * (3*h - 1.0/(2*h)),
                delta=tol
                )
        self.assertAlmostEqual(
                mesh.control_volumes[2],
                1.0/72.0 * (3*h - 1.0/(2*h)),
                delta=tol
                )
        self.assertAlmostEqual(
                mesh.control_volumes[3],
                1.0/36.0 * (h + 1.0/(2*h)),
                delta=tol
                )

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], h/6.0, delta=tol)

        return

    def test_degenerate_tet1(self):
        h = 1.0e-1
        points = numpy.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0.25, 0.25, h],
            [0.25, 0.25, -h],
            ])
        cells = numpy.array([
            [0, 1, 2, 3],
            [0, 1, 2, 4]
            ])
        mesh = voropy.mesh_tetra.MeshTetra(points, cells)

        total_vol = h / 3.0

        self._run_test(
                mesh,
                total_vol,
                [0.18734818957173291, 77.0/720.0],
                [2.420625, 5.0/6.0],
                [1.0 / numpy.sqrt(2.0) / 30., 1.0/60.0]
                )
        return

    def test_rectanglesmall(self):
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

        # ce_ratios
        self.assertAlmostEqual(mesh.ce_ratios[0], 0.05, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[1], 0.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[2], 5.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[3], 5.0, delta=tol)
        self.assertAlmostEqual(mesh.ce_ratios[4], 0.05, delta=tol)

        # control volumes
        self.assertAlmostEqual(mesh.control_volumes[0], 2.5, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[1], 2.5, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[2], 2.5, delta=tol)
        self.assertAlmostEqual(mesh.control_volumes[3], 2.5, delta=tol)

        # cell volumes
        self.assertAlmostEqual(mesh.cell_volumes[0], 5.0, delta=tol)
        self.assertAlmostEqual(mesh.cell_volumes[1], 5.0, delta=tol)

        self.assertEqual(mesh.num_delaunay_violations(), 0)

        return

    def test_arrow3d(self):
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

        # # pull this to see what a negative ce_ratio looks like
        # mesh.show()
        # mesh.show_edge(12)
        # from matplotlib import pyplot as plt
        # plt.show()

        self._run_test(
                mesh,
                1.2,
                [numpy.sqrt(0.30104), 0.354],
                [14.281989026063275, 2.4],
                [numpy.sqrt(0.45), 0.45]
                )

        self.assertEqual(mesh.num_delaunay_violations(), 2)

        return

    def test_tetrahedron(self):
        filename = fetch_data.download_mesh(
                'tetrahedron.msh',
                '27a5d7e102e6613a1e58629c252cb293'
                )
        mesh, _, _, _ = voropy.reader.read(filename)
        # mesh.show_edge(54)
        self._run_test(
                mesh,
                64.1500299099584,
                [17.07120343309435, 7.5899731568813653],
                [33.87181266432331, 1.6719101545282922],
                [11.571692332290635, 2.9699087921277054]
                )
        return

    def test_pacman(self):
        filename = fetch_data.download_mesh(
                'pacman.msh',
                '2da8ff96537f844a95a83abb48471b6a'
                )
        mesh, _, _, _ = voropy.reader.read(filename)
        # self._run_test(
        #         mesh,
        #         73.64573933105898,
        #         [3.596101914906618, 0.26638548094154707],
        #         [719.8706213234083, 1.8142648825759053],
        #         [2.6213234038171014, 0.13841739494523228]
        #         )

        self._run_test(
                mesh,
                73.64573933105898,
                [3.5908322974649631, 0.26638548094154707],
                [669.3501944927655, 1.8142648825759053],
                [2.6213234038171014, 0.13841739494523228]
                )

        self.assertEqual(mesh.num_delaunay_violations(), 0)

        return

    # def test_pacman_lloyd(self):
    #    filename = os.path.join(
    #            os.path.dirname(os.path.realpath(__file__)), 'pacman.vtu'
    #            )
    #     mesh, _, _, _ = voropy.reader.read(filename)

    #     mesh = voropy.mesh_tri.lloyd_smoothing(
    #         mesh,
    #         1.0e-2,
    #         verbose=False
    #         )

    #     self._run_test(
    #             mesh,
    #             73.64573933105898,
    #             [3.5571110980719789, 0.2277950272885419],
    #             [534.4169730732567, 1.4906838669106004],
    #             [2.5663979979329499, 0.1213739572895572]
    #             )

    #     self.assertEqual(mesh.num_delaunay_violations(), 0)

    #     return

    def test_shell(self):
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
        self._run_test(
                mesh,
                2 * numpy.sqrt(3),
                [2 * numpy.sqrt(2.0/3.0), 2.0/numpy.sqrt(3.0)],
                [5.0 / 3.0, numpy.sqrt(1.0 / 3.0)],
                [numpy.sqrt(3.0), numpy.sqrt(3.0) / 2.0]
                )

        self.assertEqual(mesh.num_delaunay_violations(), 0)

        return

    def test_sphere(self):
        filename = fetch_data.download_mesh(
                'sphere.msh',
                '70a5dbf79c3b259ed993458ff4aa2e93'
                )
        mesh, _, _, _ = voropy.reader.read(filename)
        self._run_test(
                mesh,
                12.273645818711595,
                [1.0177358705967492, 0.10419690304323895],
                [729.9372898474035, 3.2706494490659366],
                [0.72653362732751214, 0.05350373815413411]
                )

        # self.assertEqual(mesh.num_delaunay_violations(), 60)

        return

    def test_cubesmall(self):
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
        self._run_test(
                mesh,
                10.0,
                [numpy.sqrt(5.0) * 5.0/3.0, 5.0/3.0],
                [27.72375, 5.0/3.0],
                [numpy.sqrt(2.0) * 10.0/3.0, 10.0/3.0]
                )
        return

    # def test_toy(self):
    #     filename = fetch_data.download_mesh(
    #         'toy.msh',
    #         '1d125d3fa9f373823edd91ebae5f7a81'
    #         )
    #     mesh, _, _, _ = voropy.reader.read(filename)
    #     self._run_test(
    #             mesh,
    #             9.3875504672601107,
    #             [0.20348466631551548, 0.010271101930468585],
    #             [396.4116393776213, 3.4508458933423918],
    #             [0.091903119589148916, 0.0019959463063558944],
    #             tol=1.0e-6
    #             )
    #     return

if __name__ == '__main__':
    unittest.main()
