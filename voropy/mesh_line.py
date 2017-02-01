# -*- coding: utf-8 -*-
#
import numpy
from voropy.base import _base_mesh
from matplotlib import pyplot as plt


class MeshLine(_base_mesh):
    '''Class for handling 1D line "meshes".
    '''
    def __init__(self, node_coords, cells):
        '''Initialization.
        '''
        super(MeshLine, self).__init__(node_coords, cells)

        num_cells = len(cells)
        self.cells = numpy.empty(
                num_cells,
                dtype=numpy.dtype([('nodes', (int, 2))])
                )
        self.cells['nodes'] = cells

        self.create_cell_volumes()
        self.create_control_volumes()
        return

    def create_cell_volumes(self):
        '''Computes the volumes of the tetrahedra in the mesh.
        '''
        num_cells = len(self.cells['nodes'])
        self.cell_volumes = numpy.empty(num_cells, dtype=float)
        self.cell_volumes = numpy.array([
            abs(self.node_coords[cell['nodes']][1] -
                self.node_coords[cell['nodes']][0])
            for cell in self.cells
            ])
        return

    def create_control_volumes(self):
        '''Compute the control volumes of all nodes in the mesh.
        '''
        self.control_volumes = numpy.zeros(
                len(self.node_coords),
                dtype=float
                )
        for k, cell in enumerate(self.cells):
            node_ids = cell['nodes']
            self.control_volumes[node_ids] += 0.5 * self.cell_volumes[k]

        # Sanity checks.
        sum_cv = sum(self.control_volumes)
        sum_cells = sum(self.cell_volumes)
        alpha = sum_cv - sum_cells
        assert abs(alpha) < 1.0e-6
        return

    def show_vertex_function(self, u):
        plt.plot(self.node_coords, u)
        return
