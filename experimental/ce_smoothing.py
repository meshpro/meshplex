#
# This is an attempt to use the sum of the c/e ratios for triangular mesh
# smoothing.
# Works fine for very small meshes, but for larger meshes, it turns out that
# that the objective functional has many -- too many -- local minima that have
# the iteration trapped in a point away from the minimum.  Lloyd's smoothing is
# clearly superior.
#
from voropy import mesh_tri
import meshio
import numpy
import voropy


def _objective(mesh):
    double_vol = 2 * mesh.cell_volumes
    num_cells = len(mesh.cell_volumes)
    return (
        -numpy.sum(numpy.sum(mesh.ei_dot_ej, axis=0) / double_vol)
        / num_cells
        )


def _grad(mesh):
    double_vol = 2 * mesh.cell_volumes

    s = numpy.sum(mesh.ei_dot_ej, axis=0)
    e = mesh.half_edge_coords

    ei_dot_ei = numpy.einsum('ijk, ijk->ij', e, e)
    grad_x0 = (
        + (e[1] - e[2]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[0][..., None] * (e[1] - e[2])
            + (mesh.ei_dot_ej[1] - mesh.ei_dot_ej[2])[..., None] * e[0]
            )
        ) / double_vol[..., None]**3
    grad_x1 = (
        + (e[2] - e[0]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[1][..., None] * (e[2] - e[0])
            + (mesh.ei_dot_ej[2] - mesh.ei_dot_ej[0])[..., None] * e[1]
            )
        ) / double_vol[..., None]**3
    grad_x2 = (
        + (e[0] - e[1]) * double_vol[..., None]**2
        + 0.5 * s[..., None] * (
            + ei_dot_ei[2][..., None] * (e[0] - e[1])
            + (mesh.ei_dot_ej[0] - mesh.ei_dot_ej[1])[..., None] * e[2]
            )
        ) / double_vol[..., None]**3

    grad_stack = numpy.array([
        grad_x0, grad_x1, grad_x2
        ])

    # add up all the contributions
    grad = numpy.zeros(mesh.node_coords.shape)
    numpy.add.at(grad, mesh.cells['nodes'].T, grad_stack)

    return grad


def smooth(mesh):
    # x = numpy.array([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [1.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.1, 0.5, 0.0],
    #     ])
    # cells = numpy.array([
    #     [0, 1, 4],
    #     [1, 2, 4],
    #     [2, 3, 4],
    #     [3, 0, 4],
    #     ])
    # mesh = mesh_tri.MeshTri(x, cells)
    boundary_verts = mesh.get_boundary_vertices()

    t = 1.0e-3
    for k in range(100):
        mesh = mesh_tri.flip_until_delaunay(mesh)
        x = mesh.node_coords.copy()
        x -= t * _grad(mesh)
        x[boundary_verts] = mesh.node_coords[boundary_verts]
        mesh = mesh_tri.MeshTri(x, mesh.cells['nodes'])
        mesh.write('smoo%04d.vtu' % k)
        print(_objective(mesh))
    return mesh


if __name__ == '__main__':
    pts, cells, _, _, _ = meshio.read('pacman.vtu')

    # x = mesh.node_coords.copy()
    # x[:, :2] += 1.0e-1 * (
    #     numpy.random.rand(len(mesh.node_coords), 2) - 0.5
    #     )
    # x[boundary_verts] = mesh.node_coords[boundary_verts]

    # only include nodes which are part of a cell
    uvertices, uidx = numpy.unique(cells['triangle'], return_inverse=True)
    cells = uidx.reshape(cells['triangle'].shape)
    pts = pts[uvertices]

    mesh = voropy.mesh_tri.MeshTri(pts, cells)
    smooth(mesh)
