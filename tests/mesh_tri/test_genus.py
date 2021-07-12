import meshzoo
import meshplex

def test_euler_characteristic():
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cells = [[0, 1, 2]]
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.euler_characteristic == 1
    assert mesh.genus == 0

    points, cells = meshzoo.icosa_sphere(5)
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.euler_characteristic == 2
    assert mesh.genus == 0

    points, cells = meshzoo.moebius(num_twists=1, nl=21, nw=6)
    mesh = meshplex.MeshTri(points, cells)
    assert mesh.euler_characteristic == 0
    assert mesh.genus == 0


    exit(1)
