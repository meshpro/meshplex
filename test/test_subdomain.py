import pathlib

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_get_edges():
    mesh = meshplex.read(this_dir / "meshes" / "pacman.vtk")
    mesh.create_facets()
    edge_mask = mesh.get_edge_mask()
    edge_points = mesh.edges["points"][edge_mask]
    assert len(edge_points) == 1276


def test_mark_subdomain2d():
    mesh = meshplex.read(this_dir / "meshes" / "pacman.vtk")

    class Subdomain1:
        is_boundary_only = True

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] < 0.0

    class Subdomain2:
        is_boundary_only = False

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] > 0.0

    sd1 = Subdomain1()
    vertex_mask = mesh.get_vertex_mask(sd1)
    assert vertex_mask.sum() == 27
    face_mask = mesh.get_face_mask(sd1)
    assert face_mask.sum() == 26
    cell_mask = mesh.get_cell_mask(sd1)
    assert cell_mask.sum() == 0

    sd2 = Subdomain2()
    vertex_mask = mesh.get_vertex_mask(sd2)
    assert vertex_mask.sum() == 214
    face_mask = mesh.get_face_mask(sd2)
    assert face_mask.sum() == 1137
    cell_mask = mesh.get_cell_mask(sd2)
    assert cell_mask.sum() == 371


def test_mark_subdomain3d():
    mesh = meshplex.read(this_dir / "meshes" / "tetrahedron.vtk")

    class Subdomain1:
        is_boundary_only = True

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] < 0.5

    class Subdomain2:
        is_boundary_only = False

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] > 0.5

    sd1 = Subdomain1()
    vertex_mask = mesh.get_vertex_mask(sd1)
    assert vertex_mask.sum() == 16
    face_mask = mesh.get_face_mask(sd1)
    assert face_mask.sum() == 20
    cell_mask = mesh.get_cell_mask(sd1)
    assert cell_mask.sum() == 0

    sd2 = Subdomain2()
    vertex_mask = mesh.get_vertex_mask(sd2)
    assert vertex_mask.sum() == 10
    face_mask = mesh.get_face_mask(sd2)
    assert face_mask.sum() == 25
    cell_mask = mesh.get_cell_mask(sd2)
    assert cell_mask.sum() == 5
