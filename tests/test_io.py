import tempfile

import meshzoo
import numpy as np

import meshplex


def test_io_2d():
    vertices, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), 2)
    mesh = meshplex.MeshTri(vertices, cells)
    # mesh = meshplex.read('pacman.vtu')
    assert mesh.num_delaunay_violations == 0

    # mesh.show(show_axes=False, boundary_edge_color="g")
    # mesh.show_vertex(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        mesh.write(tmpdir + "test.vtk")
        mesh2 = meshplex.read(tmpdir + "test.vtk")

    assert np.all(mesh.cells("points") == mesh2.cells("points"))


# def test_io_3d(self):
#     vertices, cells = meshzoo.cube(
#             0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#             2, 2, 2
#             )
#     mesh = meshplex.MeshTetra(vertices, cells)

#     self.assertEqual(mesh.num_delaunay_violations, 0)

#     # mesh.show_control_volume(0)
#     # mesh.show_edge(0)
#     # import matplotlib.pyplot as plt
#     # plt.show()

#     mesh.write('test.vtu')

#     mesh2, _, _, _ = meshplex.read('test.vtu')

#     for k in range(len(mesh.cells['points'])):
#         self.assertEqual(
#                 tuple(mesh.cells['points'][k]),
#                 tuple(mesh2.cells['points'][k])
#                 )
