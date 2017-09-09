# -*- coding: utf-8 -*-
#
import unittest

import numpy
import voropy

from helpers import download_mesh


class CurlTest(unittest.TestCase):

    def setUp(self):
        return

    def _run_test(self, mesh):
        # Create circular vector field 0.5 * (y, -x, 0)
        # which has curl (0, 0, 1).
        A = numpy.array([
            [-0.5 * coord[1], 0.5 * coord[0], 0.0]
            for coord in mesh.node_coords
            ])
        # Compute the curl numerically.
        B = mesh.compute_curl(A)

        # mesh.write(
        #     'curl.vtu',
        #     point_data={'A': A},
        #     cell_data={'B': B}
        #     )

        tol = 1.0e-14
        for b in B:
            self.assertAlmostEqual(b[0], 0.0, delta=tol)
            self.assertAlmostEqual(b[1], 0.0, delta=tol)
            self.assertAlmostEqual(b[2], 1.0, delta=tol)
        return

    def test_pacman(self):
        filename = download_mesh(
                'pacman.msh',
                '2da8ff96537f844a95a83abb48471b6a'
                )
        mesh, _, _, _ = voropy.read(filename)
        self._run_test(mesh)
        return


if __name__ == '__main__':
    unittest.main()
