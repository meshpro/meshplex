import numpy as np
import meshplex


# control volume centroids
def test_cvc_tri():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2]])
    mesh = meshplex.MeshTri(points, cells)

    ref = np.array([[0.25, 0.25], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])

    print(mesh.control_volume_centroids)
    assert mesh.control_volume_centroids.shape == ref.shape
    assert np.all(
        np.abs(ref - mesh.control_volume_centroids) < np.abs(ref) * 1.0e-13 + 1.0e-13
    )
