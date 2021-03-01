"""
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
"""
import meshio
import numpy as np

from ._mesh_tetra import MeshTetra
from ._mesh_tri import MeshTri

__all__ = ["read"]


def _sanitize(points, cells):
    uvertices, uidx = np.unique(cells, return_inverse=True)
    cells = uidx.reshape(cells.shape)
    points = points[uvertices]
    return points, cells


def from_meshio(mesh):
    """Transform from meshio to meshplex format.

    :param mesh: The meshio mesh object.
    :type mesh: meshio.Mesh
    :returns mesh{2,3}d: The mesh data.
    """
    # make sure to include the used nodes only
    tetra = mesh.get_cells_type("tetra")
    if len(tetra) > 0:
        points, cells = _sanitize(mesh.points, tetra)
        return MeshTetra(points, cells)

    tri = mesh.get_cells_type("triangle")
    assert len(tri) > 0
    points, cells = _sanitize(mesh.points, tri)
    return MeshTri(points, cells)


def read(filename):
    """Reads an unstructured mesh into meshplex format.

    :param filenames: The files to read from.
    :type filenames: str
    :returns mesh{2,3}d: The mesh data.
    """
    return from_meshio(meshio.read(filename))
