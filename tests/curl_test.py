# -*- coding: utf-8 -*-
#
import fetch_data
import voropy

import numpy
import unittest


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
        for k in range(len(B)):
            self.assertAlmostEqual(B[k][0], 0.0, delta=tol)
            self.assertAlmostEqual(B[k][1], 0.0, delta=tol)
            self.assertAlmostEqual(B[k][2], 1.0, delta=tol)
        return

    def test_pacman(self):
        filename = fetch_data.download_mesh(
                'pacman.msh',
                '2da8ff96537f844a95a83abb48471b6a'
                )
        mesh, _, _, _ = voropy.reader.read(filename)
        self._run_test(mesh)
        return


if __name__ == '__main__':
    unittest.main()
