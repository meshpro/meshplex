# -*- coding: utf-8 -*-
#
# pylint: disable=too-few-public-methods
from helpers import download_mesh
import meshplex


def test_get_edges():
    filename = download_mesh("pacman.msh", "2da8ff96537f844a95a83abb48471b6a")
    mesh = meshplex.read(filename)
    mesh.create_edges()
    edge_mask = mesh.get_edge_mask()
    edge_nodes = mesh.edges["nodes"][edge_mask]
    assert len(edge_nodes) == 1276
    return


def test_mark_subdomain2d():
    filename = download_mesh("pacman.msh", "2da8ff96537f844a95a83abb48471b6a")
    mesh = meshplex.read(filename)

    class Subdomain1(object):
        is_boundary_only = True

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] < 0.0

    class Subdomain2(object):
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
    return


def test_mark_subdomain3d():
    filename = download_mesh("tetrahedron.msh", "27a5d7e102e6613a1e58629c252cb293")
    mesh = meshplex.read(filename)

    class Subdomain1(object):
        is_boundary_only = True

        # pylint: disable=no-self-use
        def is_inside(self, x):
            return x[0] < 0.5

    class Subdomain2(object):
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
    return
