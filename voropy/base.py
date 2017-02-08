# -*- coding: utf-8 -*-
#
import meshio
import numpy
from numpy.core.umath_tests import inner1d

__all__ = []


def _row_dot(a, b):
    # http://stackoverflow.com/a/26168677/353337
    # return numpy.einsum('ij, ij->i', a, b)
    #
    # http://stackoverflow.com/a/39657905/353337
    return inner1d(a, b)


def compute_tri_areas_and_ce_ratios(e0, e1, e2):
    '''Given triangles (specified by their edges), this routine will return the
    triangle areas and the signed distances of the triangle circumcenters to
    the edge midpoints.
    '''
    # TODO Make sure the edges are sorted such that
    # e0: x0 -> x1
    # e1: x1 -> x2
    # e2: x2 -> x0
    # assert numpy.allclose(e0 + e1, -e2, rtol=0.0, atol=1.0e-14)

    # The covolume-edge ratios for the edges of each cell is the solution of
    # the equation system
    #
    # |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>,
    #
    # where alpha_i are the covolume contributions for the edges. This
    # equation system to hold for all vectors u in the plane spanned by the
    # edges, particularly by the edges themselves.
    #
    # For triangles, the exact solution of the system is
    #
    #  x_1 = <e_2, e_3> / <e1 x e2, e1 x e3> * |simplex|;
    #
    # see <http://math.stackexchange.com/a/1855380/36678>.
    #
    # e0_cross_e1 = numpy.cross(e0, e1)
    # e1_cross_e2 = numpy.cross(e1, e2)
    # e2_cross_e0 = numpy.cross(e2, e0)
    # cell_volumes = 0.5 * numpy.sqrt(
    #         _row_dot(e0_cross_e1, e0_cross_e1)
    #         )
    # a = _row_dot(e1, e2) / _row_dot(e0_cross_e1, -e2_cross_e0)
    # b = _row_dot(e2, e0) / _row_dot(e1_cross_e2, -e0_cross_e1)
    # c = _row_dot(e0, e1) / _row_dot(e2_cross_e0, -e1_cross_e2)
    #
    # Note that the edges are connected by the relationship
    #
    #   e1 + s2*e2 + s3*e3 = 0
    #
    # where the s_ are signs. (We assume s1=1 w.l.o.g.) Since
    #
    #   <e1 x e2, e1 x e3> = <e1 x e2, e1 x (-s1*s3*e1-s2*s3*e2)>
    #                      = -s2*s3 * <e1 x e2, e1 x e2>,
    #
    # we have that
    #
    #   |simplex| = 0.5 * sqrt(abs(<e1 x e2, e1 x e3>))
    #
    # so
    #
    #   x_1 = 0.5 * <e_2, e_3> / sqrt(abs(<e1 x e2, e1 x e3>))
    #       * sign(<e1 x e2, e1 x e3>)
    #
    # Since dot-products are much faster computed than cross-products, we
    # rewrite the former as
    #
    #   <e1 x e2, e1 x e3> = <e1, e1> <e2, e3> - <e1, e2> <e1, e3>.
    #
    e0_dot_e1 = _row_dot(e0, e1)
    e0_dot_e2 = _row_dot(e0, e2)
    e1_dot_e2 = _row_dot(e1, e2)
    #
    e0_dot_e0 = _row_dot(e0, e0)
    e1_dot_e1 = _row_dot(e1, e1)
    e2_dot_e2 = _row_dot(e2, e2)
    #
    aa = e0_dot_e0 * e1_dot_e2 - e0_dot_e1 * e0_dot_e2
    bb = e1_dot_e1 * e0_dot_e2 - e0_dot_e1 * e1_dot_e2
    cc = e2_dot_e2 * e0_dot_e1 - e0_dot_e2 * e1_dot_e2
    # assert abs(aa) == abs(bb)
    # assert abs(bb) == abs(cc)
    a = e1_dot_e2 * 0.5 / numpy.sqrt(abs(aa)) * numpy.sign(aa)
    b = e0_dot_e2 * 0.5 / numpy.sqrt(abs(bb)) * numpy.sign(bb)
    c = e0_dot_e1 * 0.5 / numpy.sqrt(abs(cc)) * numpy.sign(cc)

    # Any of aa, bb, cc are good for this. A more symmetric
    # variant like
    #   s = numpy.sqrt(abs(
    #       e0_dot_e1 * e1_dot_e2 +
    #       e0_dot_e2 * e1_dot_e2 +
    #       e0_dot_e1 * e0_dot_e2
    #       ))
    # might be preferable (if only for the fact that we don't need to compute
    # e0_dot_e0).
    #
    cell_volumes = 0.5 * numpy.sqrt(abs(aa))
    #
    # TODO We could dodge computing <e1, e1> by noting that
    #
    #   <e1, e1> = -s2 * <e1, e2> - s3 * <e1, e3>,
    #
    # so
    #
    #   <e1 x e2, e1 x e3> =\
    #       -s2 * <e1, e2> * <e2, e3> \
    #       -s3 * <e1, e3> * <e2, e3> \
    #       -     <e1, e2> <e1, e3>.
    #
    # This would require finding out about s2, s3 though. Perhaps we can get
    # this from the input data.
    #

    sol = numpy.column_stack((a, b, c))

    return cell_volumes, sol


def compute_triangle_circumcenters(X):
    '''Computes the center of the circumcenter of all given triangles.
    '''
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Higher_dimensions
    a = X[:, 0, :] - X[:, 2, :]
    b = X[:, 1, :] - X[:, 2, :]
    a_dot_a = _row_dot(a, a)
    b_dot_b = _row_dot(b, b)
    a_dot_b = _row_dot(a, b)
    # N = (<a,a> b - <b,b> a) x (a x b)
    #   = <a,a> (b x (a x b)) - <b,b> (a x (a x b))
    #   = <a,a> (a <b,b> - b <b,a>) - <b,b> (a <a,b> - b <a,a>)
    #   = a <b,b> (<a,a> - <a,b>) + b <a,a> (<b,b> - <b,a>)
    alpha = b_dot_b * (a_dot_a - a_dot_b)
    beta = a_dot_a * (b_dot_b - a_dot_b)
    N = a * alpha[..., None] + b * beta[..., None]
    # <a x b, a x b> = <a, a> <b, b> - <a, b>^2
    a_cross_b2 = a_dot_a * b_dot_b - a_dot_b**2
    return 0.5 * N / a_cross_b2[..., None] + X[:, 2, :]


class _base_mesh(object):

    def __init__(self,
                 nodes,
                 cells_nodes
                 ):
        self.node_coords = nodes
        return

    def write(self,
              filename,
              point_data=None,
              cell_data=None,
              field_data=None
              ):
        if self.node_coords.shape[1] == 2:
            n = len(self.node_coords)
            a = numpy.ascontiguousarray(
                numpy.column_stack([self.node_coords, numpy.zeros(n)])
                )
        else:
            a = self.node_coords

        if self.cells['nodes'].shape[1] == 3:
            cell_type = 'triangle'
        elif self.cells['nodes'].shape[1] == 4:
            cell_type = 'tetra'
        else:
            raise RuntimeError('Only triangles/tetrahedra supported')

        meshio.write(
            filename,
            a,
            {cell_type: self.cells['nodes']},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )

    def compute_edge_lengths(self, edge_ids=None):
        if edge_ids is not None:
            edges = self.node_coords[self.edges['nodes'][edge_ids, 1]] \
                - self.node_coords[self.edges['nodes'][edge_ids, 0]]
        else:
            edges = self.node_coords[self.edges['nodes'][:, 1]] \
                - self.node_coords[self.edges['nodes'][:, 0]]
        return numpy.sqrt(_row_dot(edges, edges))

    def get_edges(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['edges']

    def get_vertices(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['vertices']

    def mark_subdomain(self, subdomain):
        # find vertices in subdomain
        if subdomain.is_boundary_only:
            nodes = self.get_vertices('boundary')
        else:
            nodes = self.get_vertices('everywhere')

        subdomain_vertices = []
        for vertex_id in nodes:
            if subdomain.is_inside(self.node_coords[vertex_id]):
                subdomain_vertices.append(vertex_id)
        subdomain_vertices = numpy.unique(subdomain_vertices)

        # extract all edges which are completely or half in the subdomain
        if subdomain.is_boundary_only:
            edges = self.get_edges('boundary')
        else:
            edges = self.get_edges('everywhere')

        subdomain_edges = []
        subdomain_split_edges = []
        for edge_id in edges:
            verts = self.edges['nodes'][edge_id]
            if verts[0] in subdomain_vertices:
                if verts[1] in subdomain_vertices:
                    subdomain_edges.append(edge_id)
                else:
                    subdomain_split_edges.append(edge_id)

        subdomain_edges = numpy.unique(subdomain_edges)
        subdomain_split_edges = numpy.unique(subdomain_split_edges)

        self.subdomains[subdomain] = {
                'vertices': subdomain_vertices,
                'edges': subdomain_edges,
                'split_edges': subdomain_split_edges
                }

        return
