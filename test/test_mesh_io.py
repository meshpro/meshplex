import tempfile

import meshzoo

import meshplex


def test_io_2d():
    vertices, cells = meshzoo.rectangle(0.0, 1.0, 0.0, 1.0, 2, 2)
    mesh = meshplex.MeshTri(vertices, cells)
    # mesh = meshplex.read('pacman.vtu')

    assert mesh.num_delaunay_violations() == 0

    # mesh.show(show_axes=False, boundary_edge_color="g")
    # mesh.show_vertex(0)

    _, fname = tempfile.mkstemp(suffix=".vtk")
    mesh.write(fname)

    mesh2 = meshplex.read(fname)

    for k in range(len(mesh.cells["nodes"])):
        assert tuple(mesh.cells["nodes"][k]) == tuple(mesh2.cells["nodes"][k])


# def test_io_3d(self):
#     vertices, cells = meshzoo.cube(
#             0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
#             2, 2, 2
#             )
#     mesh = meshplex.MeshTetra(vertices, cells)

#     self.assertEqual(mesh.num_delaunay_violations(), 0)

#     # mesh.show_control_volume(0)
#     # mesh.show_edge(0)
#     # import matplotlib.pyplot as plt
#     # plt.show()

#     mesh.write('test.vtu')

#     mesh2, _, _, _ = meshplex.read('test.vtu')

#     for k in range(len(mesh.cells['nodes'])):
#         self.assertEqual(
#                 tuple(mesh.cells['nodes'][k]),
#                 tuple(mesh2.cells['nodes'][k])
#                 )
