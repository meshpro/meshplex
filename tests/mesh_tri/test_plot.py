import pathlib

import meshplex

this_dir = pathlib.Path(__file__).resolve().parent


def test_show_mesh():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman-optimized.vtk")
    mesh = meshplex.MeshTri(mesh.points[:, :2], mesh.cells("points"))
    print(mesh)  # test __repr__
    # mesh.plot(show_axes=False)
    mesh.show(
        show_axes=False,
        cell_quality_coloring=("viridis", 0.0, 1.0, True),
        show_point_numbers=True,
        show_edge_numbers=True,
        show_cell_numbers=True,
        mark_points=[1],
        mark_edges=[0],
        mark_cells=[0, 3, 7],
        nondelaunay_edge_color="r",
        boundary_edge_color="b",
        control_volume_centroid_color="g",
    )
    # mesh.save("pacman.png", show_axes=False)


def test_show_vertex():
    mesh = meshplex.read(this_dir / ".." / "meshes" / "pacman-optimized.vtk")
    # mesh.plot_vertex(125)
    mesh.show_vertex(125)


if __name__ == "__main__":
    test_show_mesh()
