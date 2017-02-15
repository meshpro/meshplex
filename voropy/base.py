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


def compute_tri_areas_and_ce_ratios(ei_dot_ej):
    '''Given triangles (specified by their edges), this routine will return the
    triangle areas and the signed distances of the triangle circumcenters to
    the edge midpoints.
    '''
    # The input argument are the dot products
    #
    #   <e1, e2>
    #   <e2, e0>
    #   <e0, e1>
    #
    # of the edges
    #
    #   e0: x1->x2,
    #   e1: x2->x0,
    #   e2: x0->x1.
    #
    # Note that edge e_i is opposite of node i and the edges add up to 0.
    # (Those quantities can be shared between numerous methods, so share them.)
    #

    # There are multiple ways of deriving a closed form for the
    # covolume-edgelength ratios.
    #
    #
    #   * The covolume-edge ratios for the edges of each cell is the solution
    #     of the equation system
    #
    #       |simplex| ||u||^2 = \sum_i \alpha_i <u,e_i> <e_i,u>,
    #
    #     where alpha_i are the covolume contributions for the edges. This
    #     equation system to hold for all vectors u in the plane spanned by the
    #     edges, particularly by the edges themselves.
    #
    #     For triangles, the exact solution of the system is
    #
    #       x_1 = <e_2, e_3> / <e1 x e2, e1 x e3> * |simplex|;
    #
    #     see <http://math.stackexchange.com/a/1855380/36678>.
    #
    #   * In trilinear coordinates
    #     <https://en.wikipedia.org/wiki/Trilinear_coordinates>, the
    #     circumcenter is
    #
    #         cos(alpha0) : cos(alpha1) : cos(alpha2)
    #
    #     where the alpha_i are the angles opposite of the respective edge.
    #     With
    #
    #       cos(alpha0) = <e1, e2> / ||e1|| / ||e2||
    #
    #     and the conversion formula to Cartesian coordinates, ones gets the
    #     expression
    #
    #         ce0 = e1_dot_e2 * 0.5 / sqrt(alpha)
    #
    #     with
    #
    #         alpha = e1_dot_e2 * e0_dot_e1
    #               + e2_dot_e0 * e1_dot_e2
    #               + e0_dot_e1 * e2_dot_e0.
    #
    # Note that some simplifications are achieved by virtue of
    #
    #   e1 + e2 + e3 = 0.
    #
    cell_volumes = 0.5 * numpy.sqrt(
        + ei_dot_ej[2] * ei_dot_ej[0]
        + ei_dot_ej[0] * ei_dot_ej[1]
        + ei_dot_ej[1] * ei_dot_ej[2]
        )

    sol = -ei_dot_ej * 0.25 / cell_volumes[None]

    return cell_volumes, sol


def compute_triangle_circumcenters(X, ei_dot_ei, ei_dot_ej):
    '''Computes the center of the circumcenter of all given triangles.
    '''
    # The input argument are the dot products
    #
    #   <e1, e2>
    #   <e2, e0>
    #   <e0, e1>
    #
    # of the edges
    #
    #   e0: x1->x2,
    #   e1: x2->x0,
    #   e2: x0->x1.
    #
    # Note that edge e_i is opposite of node i and the edges add up to 0.

    # The trilinear coordinates of the circumcenter are
    #
    #   cos(alpha0) : cos(alpha1) : cos(alpha2)
    #
    # where alpha_k is the angle at point k, opposite of edge k. The Cartesian
    # coordinates are (see
    # <https://en.wikipedia.org/wiki/Trilinear_coordinates#Between_Cartesian_and_trilinear_coordinates>)
    #
    #     C = sum_i ||e_i|| * cos(alpha_i)/beta * P_i
    #
    # with
    #
    #     beta = sum ||e_i||*cos(alpha_i)
    #
    # Incidentally, the cosines are
    #
    #    cos(alpha0) = <e1, e2> / ||e1|| / ||e2||,
    #
    # so in total
    #
    #    C = <e_0, e0> <e1, e2> / sum_i (<e_i, e_i> <e{i+1}, e{i+2}>) P0
    #      + ... P1
    #      + ... P2.
    #
    alpha = ei_dot_ei * ei_dot_ej
    alpha_sum = alpha[0] + alpha[1] + alpha[2]

    beta = alpha / alpha_sum[None]

    a = X * beta[..., None]
    cc = a[0] + a[1] + a[2]

    return cc


class _base_mesh(object):

    def __init__(self,
                 nodes,
                 cells_nodes
                 ):
        self.node_coords = nodes
        self._edge_lengths = None
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

    def get_edge_lengths(self):
        if self._edge_lengths is None:
            edges = self.node_coords[self.cell_edge_nodes[..., 1]] \
                - self.node_coords[self.cell_edge_nodes[..., 0]]
            self._edge_lengths = numpy.sqrt(_row_dot(edges, edges))

        return self._edge_lengths

    def get_edges(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['edges']

    def get_cells(self, subdomain):
        if subdomain not in self.subdomains:
            self.mark_subdomain(subdomain)
        return self.subdomains[subdomain]['cells']

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
