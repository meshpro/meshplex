import pathlib

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_pacman():
    mesh = meshplex.read(this_dir / "meshes" / "pacman.vtk")
    mesh = meshplex.MeshTri(mesh.points[:, :2], mesh.cells("points"))

    mesh._compute_cell_values()
    mesh._compute_cell_values(mask=[0, 12, 53, 54, 55])
