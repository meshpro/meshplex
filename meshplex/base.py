import meshio
import numpy

__all__ = []


class _BaseMesh:
    def __init__(self, points, cells_points):
        self._edge_lengths = None

    def write(self, filename, point_data=None, cell_data=None, field_data=None):
        if self.points.shape[1] == 2:
            n = len(self.points)
            a = numpy.ascontiguousarray(
                numpy.column_stack([self.points, numpy.zeros(n)])
            )
        else:
            a = self.points

        if self.cells["points"].shape[1] == 3:
            cell_type = "triangle"
        else:
            assert (
                self.cells["points"].shape[1] == 4
            ), "Only triangles/tetrahedra supported"
            cell_type = "tetra"

        meshio.write_points_cells(
            filename,
            a,
            {cell_type: self.cells["points"]},
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    @property
    def edge_lengths(self):
        if self._edge_lengths is None:
            self._edge_lengths = numpy.sqrt(self.ei_dot_ei)
        return self._edge_lengths

    def get_vertex_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]
        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)
        return self.subdomains[subdomain]["vertices"]

    def get_edge_mask(self, subdomain=None):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over the first index
        is_inside = numpy.all(is_in, axis=tuple(range(1)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_edge

        return is_inside

    def get_face_mask(self, subdomain):
        """Get faces which are fully in subdomain."""
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        # A face is inside if all its edges are in.
        # An edge is inside if all its points are in.
        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last two (face_ids, cell_ids).
        n = len(is_in.shape)
        is_inside = numpy.all(is_in, axis=tuple(range(n - 2)))

        if subdomain.is_boundary_only:
            # Filter for boundary
            is_inside = is_inside & self.is_boundary_facet

        return is_inside

    def get_cell_mask(self, subdomain=None):
        if subdomain is None:
            # https://stackoverflow.com/a/42392791/353337
            return numpy.s_[:]

        if subdomain.is_boundary_only:
            # There are no boundary cells
            return numpy.array([])

        if subdomain not in self.subdomains:
            self._mark_vertices(subdomain)

        is_in = self.subdomains[subdomain]["vertices"][self.idx_hierarchy]
        # Take `all()` over all axes except the last one (cell_ids).
        n = len(is_in.shape)
        return numpy.all(is_in, axis=tuple(range(n - 1)))

    def _mark_vertices(self, subdomain):
        """Mark faces/edges which are fully in subdomain."""
        if subdomain is None:
            is_inside = numpy.ones(len(self.points), dtype=bool)
        else:
            is_inside = subdomain.is_inside(self.points.T).T

            if subdomain.is_boundary_only:
                # Filter boundary
                self.mark_boundary()
                is_inside = is_inside & self.is_boundary_point

        self.subdomains[subdomain] = {"vertices": is_inside}
        return
