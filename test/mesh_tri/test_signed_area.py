import pathlib
import numpy
import meshio

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_signed_area():
    mesh = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")
    assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-15)
    X = mesh.points[:, :2]

    mesh = meshplex.MeshTri(X, mesh.get_cells_type("triangle"))

    vols = mesh.signed_cell_areas
    assert numpy.all(abs(abs(vols) - mesh.cell_volumes) < 1.0e-12 * mesh.cell_volumes)


def test_signed_area2():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = numpy.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)
    ref = 0.5
    assert abs(mesh.signed_cell_areas[0] - ref) < 1.0e-10 * abs(ref)

    mesh.points = numpy.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ref = -0.5
    assert abs(mesh.signed_cell_areas[0] - ref) < 1.0e-10 * abs(ref)
