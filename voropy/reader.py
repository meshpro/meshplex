# -*- coding: utf-8 -*-
#
'''
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import meshio
import numpy
import voropy

__all__ = ['read']


def _sanitize(points, cells):
    uvertices, uidx = numpy.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = points[uvertices]
    return points, cells


def read(filename, flat_cell_correction=None):
    '''Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str
    :returns mesh{2,3}d: The mesh data.
    :returns point_data: Point data read from file.
    :type point_data: dict
    :returns field_data: Field data read from file.
    :type field_data: dict
    '''
    points, cells, point_data, cell_data, field_data = \
        meshio.read(filename)

    # make sure to include the used nodes only
    if 'tetra' in cells:
        points, cells = _sanitize(points, cells['tetra'])
        return voropy.mesh_tetra.MeshTetra(points, cells), \
            point_data, cell_data, field_data
    elif 'triangle' in cells:
        points, cells = _sanitize(points, cells['triangle'])
        return voropy.mesh_tri.MeshTri(
               points, cells,
               flat_cell_correction=flat_cell_correction
               ), \
            point_data, cell_data, field_data
    else:
        raise RuntimeError('Unknown mesh type.')
