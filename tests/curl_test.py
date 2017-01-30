# -*- coding: utf-8 -*-
#
import voropy

import os
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
        # Compute the gradient numerically.
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
        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'pacman.vtu'
            )
        mesh, _, _, _ = voropy.reader.read(filename)
        self._run_test(mesh)
        return

if __name__ == '__main__':
    unittest.main()
