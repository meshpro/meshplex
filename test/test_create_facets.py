import numpy as np
import pytest

import meshplex


@pytest.mark.skip()
def test_line():
    X = np.array([0.0, 0.1, 0.4, 1.0])
    cells = np.array([[0, 1], [1, 2], [2, 3]])
    mesh = meshplex.Mesh(X, cells)
    mesh.create_facets()

    ref = np.array([0, 1, 2, 3])
    assert np.all(mesh.facets["points"] == ref)

    ref = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )
    assert np.all(mesh.cells("facets") == ref)


def test_tri():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = meshplex.Mesh(X, cells)
    mesh.create_facets()

    ref = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [2, 3],
        ]
    )
    assert np.all(mesh.facets["points"] == ref)

    ref = np.array(
        [
            [3, 1, 0],
            [4, 2, 1],
        ]
    )
    assert np.all(mesh.cells("facets") == ref)


def test_tetra():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    mesh = meshplex.Mesh(X, cells)
    mesh.create_facets()

    ref = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 3],
            [1, 2, 4],
        ]
    )
    assert np.all(mesh.facets["points"] == ref)

    ref = np.array(
        [
            [5, 3, 1, 0],
            [6, 4, 2, 0],
        ]
    )
    assert np.all(mesh.cells("facets") == ref)


def test_duplicate_cells():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    cells = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1]])
    mesh = meshplex.Mesh(X, cells)
    with pytest.raises(meshplex.MeshplexError):
        mesh.create_facets()
