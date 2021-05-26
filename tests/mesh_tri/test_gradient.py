# from helpers import download_mesh
# import meshplex
#
# import numpy as np
# import unittest
#
#
# class GradientTest(unittest.TestCase):
#
#     def setUp(self):
#         return
#
#     def _run_test(self, mesh):
#         num_nodes = len(mesh.points)
#         # Create function  2*x + 3*y.
#         a_x = 7.0
#         a_y = 3.0
#         a0 = 1.0
#         u = a_x * mesh.points[:, 0] + \
#             a_y * mesh.points[:, 1] + \
#             a0 * np.ones(num_nodes)
#         # Get the gradient analytically.
#         sol = np.empty((num_nodes, 2))
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
#                 'pacman.vtk',
#                 '2da8ff96537f844a95a83abb48471b6a'
#                 )
#         mesh, _, _, _ = meshplex.read(filename)
#         self._run_test(mesh)
#         return
#
#
# if __name__ == '__main__':
#     unittest.main()
