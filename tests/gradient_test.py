# -*- coding: utf-8 -*-
#
# from helpers import download_mesh
# import voropy
#
# import numpy
# import unittest
#
#
# class GradientTest(unittest.TestCase):
#
#     def setUp(self):
#         return
#
#     def _run_test(self, mesh):
#         num_nodes = len(mesh.node_coords)
#         # Create function  2*x + 3*y.
#         a_x = 7.0
#         a_y = 3.0
#         a0 = 1.0
#         u = a_x * mesh.node_coords[:, 0] + \
#             a_y * mesh.node_coords[:, 1] + \
#             a0 * numpy.ones(num_nodes)
#         # Get the gradient analytically.
#         sol = numpy.empty((num_nodes, 2))
#         sol[:, 0] = a_x
#         sol[:, 1] = a_y
#         # Compute the gradient numerically.
#         grad_u = mesh.compute_gradient(u)
#
#         tol = 1.0e-13
#         for k in range(num_nodes):
#             self.assertAlmostEqual(grad_u[k][0], sol[k][0], delta=tol)
#             self.assertAlmostEqual(grad_u[k][1], sol[k][1], delta=tol)
#         return
#
#     def test_pacman(self):
#         filename = download_mesh(
#                 'pacman.msh',
#                 '2da8ff96537f844a95a83abb48471b6a'
#                 )
#         mesh, _, _, _ = voropy.read(filename)
#         self._run_test(mesh)
#         return
#
#
# if __name__ == '__main__':
#     unittest.main()
