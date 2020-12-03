import pathlib

import meshio
import numpy

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_signed_area_basic():
    points = numpy.array([[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0]])
    cells = numpy.array([[0, 1, 2], [0, 3, 2]])
    mesh = meshplex.MeshTri(points, cells)
    ref = numpy.array([0.5, -0.55])
    assert numpy.all(numpy.abs(mesh.signed_cell_areas - ref) < 1.0e-10 * numpy.abs(ref))


def test_signed_area_pacman():
    mesh = meshio.read(this_dir / ".." / "meshes" / "pacman.vtk")
    assert numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-15)
    X = mesh.points[:, :2]

    mesh = meshplex.MeshTri(X, mesh.get_cells_type("triangle"))

    vols = mesh.signed_cell_areas
    # all cells are positively oriented in this mesh
    assert numpy.all(mesh.signed_cell_areas > 0.0)
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
