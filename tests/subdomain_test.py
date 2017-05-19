# -*- coding: utf-8 -*-
#
from helpers import download_mesh
import voropy


def test_get_edges():
    filename = download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    mesh, _, _, _ = voropy.read(filename)
    mesh.create_edges()
    edge_mask = mesh.get_edge_mask()
    edge_nodes = mesh.edges['nodes'][edge_mask]
    assert len(edge_nodes) == 1276
    return


def test_mark_subdomain2d():
    filename = download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    mesh, _, _, _ = voropy.read(filename)

    class Subdomain1(object):
        is_boundary_only = True

        def is_inside(self, x):
            return x[0] < 0.0

    class Subdomain2(object):
        is_boundary_only = False

        def is_inside(self, x):
            return x[0] > 0.0

    sd1 = Subdomain1()
    vertex_mask = mesh.get_vertex_mask(sd1)
    assert sum(vertex_mask) == 27
    cell_mask = mesh.get_cell_mask(sd1)
    assert sum(cell_mask) == 0

    sd2 = Subdomain2()
    vertex_mask = mesh.get_vertex_mask(sd2)
    assert sum(vertex_mask) == 214
    cell_mask = mesh.get_cell_mask(sd2)
    assert sum(cell_mask) == 371
    return


def test_mark_subdomain3d():
    filename = download_mesh(
            'tetrahedron.msh',
            '27a5d7e102e6613a1e58629c252cb293',
            )
    mesh, _, _, _ = voropy.read(filename)

    class Subdomain1(object):
        is_boundary_only = True

        def is_inside(self, x):
            return x[0] < 0.5

    class Subdomain2(object):
        is_boundary_only = False

        def is_inside(self, x):
            return x[0] > 0.5

    sd1 = Subdomain1()
    vertex_mask = mesh.get_vertex_mask(sd1)
    assert sum(vertex_mask) == 16
    cell_mask = mesh.get_cell_mask(sd1)
    assert sum(cell_mask) == 0

    sd2 = Subdomain2()
    vertex_mask = mesh.get_vertex_mask(sd2)
    assert sum(vertex_mask) == 10
    cell_mask = mesh.get_cell_mask(sd2)
    assert sum(cell_mask) == 5
    return
