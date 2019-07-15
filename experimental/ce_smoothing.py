#
# This is an attempt to use the sum of the c/e ratios for triangular mesh
# smoothing.
# Works fine for very small meshes, but for larger meshes, it turns out that
# that the objective functional has many -- too many -- local minima that have
# the iteration trapped in a point away from the minimum.  Lloyd's smoothing is
# clearly superior.
#
import numpy

import meshio
import meshplex


def _objective(mesh):
    double_vol = 2 * mesh.cell_volumes
    num_cells = len(mesh.cell_volumes)
    return -numpy.sum(numpy.sum(mesh.ei_dot_ej, axis=0) / double_vol) / num_cells


def _grad(mesh):
    double_vol = 2 * mesh.cell_volumes

    s = numpy.sum(mesh.ei_dot_ej, axis=0)
    e = mesh.half_edge_coords

    ei_dot_ei = numpy.einsum("ijk, ijk->ij", e, e)
    grad_x0 = (
        +(e[1] - e[2]) * double_vol[..., None] ** 2
        + 0.5
        * s[..., None]
        * (
            +ei_dot_ei[0][..., None] * (e[1] - e[2])
            + (mesh.ei_dot_ej[1] - mesh.ei_dot_ej[2])[..., None] * e[0]
        )
    ) / double_vol[..., None] ** 3
    grad_x1 = (
        +(e[2] - e[0]) * double_vol[..., None] ** 2
        + 0.5
        * s[..., None]
        * (
            +ei_dot_ei[1][..., None] * (e[2] - e[0])
            + (mesh.ei_dot_ej[2] - mesh.ei_dot_ej[0])[..., None] * e[1]
        )
    ) / double_vol[..., None] ** 3
    grad_x2 = (
        +(e[0] - e[1]) * double_vol[..., None] ** 2
        + 0.5
        * s[..., None]
        * (
            +ei_dot_ei[2][..., None] * (e[0] - e[1])
            + (mesh.ei_dot_ej[0] - mesh.ei_dot_ej[1])[..., None] * e[2]
        )
    ) / double_vol[..., None] ** 3

    grad_stack = numpy.array([grad_x0, grad_x1, grad_x2])

    # add up all the contributions
    grad = numpy.zeros(mesh.node_coords.shape)
    numpy.add.at(grad, mesh.cells["nodes"].T, grad_stack)

    return grad


def smooth(mesh, t=1.0e-3, num_iters=10):
    boundary_verts = mesh.get_boundary_vertices()

    for k in range(num_iters):
        # mesh = mesh_tri.flip_until_delaunay(mesh)
        x = mesh.node_coords.copy()
        x -= t * _grad(mesh)
        x[boundary_verts] = mesh.node_coords[boundary_verts]
        mesh = MeshTri(x, mesh.cells["nodes"])
        mesh.write("smoo%04d.vtu" % k)
        print(_objective(mesh))
    return mesh


def read(filename):
    pts, cells, _, _, _ = meshio.read(filename)

    # x = mesh.node_coords.copy()
    # x[:, :2] += 1.0e-1 * (
    #     numpy.random.rand(len(mesh.node_coords), 2) - 0.5
    #     )
    # x[boundary_verts] = mesh.node_coords[boundary_verts]

    # only include nodes which are part of a cell
    uvertices, uidx = numpy.unique(cells["triangle"], return_inverse=True)
    cells = uidx.reshape(cells["triangle"].shape)
    pts = pts[uvertices]

    return pts, cells


def circle(num_segments=7):
    angles = numpy.linspace(0.0, 2 * numpy.pi, num_segments, endpoint=False)
    pts = numpy.array(
        [numpy.cos(angles), numpy.sin(angles), numpy.zeros(len(angles))]
    ).transpose()
    pts = numpy.vstack([pts, [[-0.8, 0.0, 0.0]]])

    n = len(pts) - 1
    cells = numpy.array([[k, (k + 1) % n, n] for k in range(n)])

    pts = numpy.ascontiguousarray(pts)
    cells = numpy.ascontiguousarray(cells)
    return pts, cells


if __name__ == "__main__":
    # pts, cells = read('pacman.vtu')
    pts, cells = read("boundary_layers.vtu")
    # pts, cells = circle()
    mesh = meshplex.MeshTri(pts, cells)
    smooth(mesh, t=1.0e-3, num_iters=100)
