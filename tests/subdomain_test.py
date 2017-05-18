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
    edge_ids = mesh.get_edges()
    edge_nodes = mesh.edges['nodes'][edge_ids]
    assert len(edge_nodes) == 1276
    return


def test_mark_subdomain():
    filename = download_mesh(
            'pacman.msh',
            '2da8ff96537f844a95a83abb48471b6a'
            )
    mesh, _, _, _ = voropy.read(filename)
    # mesh.mark_default_subdomains()

    class Subdomain1(object):
        is_boundary_only = True

        def is_inside(self, x):
            return x[0] < 0.0

    class Subdomain2(object):
        is_boundary_only = False

        def is_inside(self, x):
            return x[0] > 0.0

    sd1 = Subdomain1()
    mesh.mark_vertices(sd1)
    assert len(mesh.subdomains[sd1]['vertices']) == 27

    sd2 = Subdomain2()
    mesh.mark_vertices(sd2)
    assert len(mesh.subdomains[sd2]['vertices']) == 214
    return
