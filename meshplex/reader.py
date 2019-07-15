"""
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
"""
import numpy

import meshio

from .mesh_tetra import MeshTetra
from .mesh_tri import MeshTri

__all__ = ["read"]


def _sanitize(points, cells):
    uvertices, uidx = numpy.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = points[uvertices]
    return points, cells


def read(filename):
    """Reads an unstructured mesh into meshplex format.

    :param filenames: The files to read from.
    :type filenames: str
    :returns mesh{2,3}d: The mesh data.
    """
    mesh = meshio.read(filename)

    # make sure to include the used nodes only
    if "tetra" in mesh.cells:
        points, cells = _sanitize(mesh.points, mesh.cells["tetra"])
        return MeshTetra(points, cells)
    elif "triangle" in mesh.cells:
        points, cells = _sanitize(mesh.points, mesh.cells["triangle"])
        return MeshTri(points, cells)

    raise RuntimeError("Illegal mesh type.")
    return
