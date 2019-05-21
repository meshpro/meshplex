# meshplex

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/meshplex/master.svg)](https://circleci.com/gh/nschloe/meshplex/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/meshplex.svg)](https://codecov.io/gh/nschloe/meshplex)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Documentation Status](https://readthedocs.org/projects/meshplex/badge/?version=latest)](https://readthedocs.org/projects/meshplex/?badge=latest)
[![PyPi Version](https://img.shields.io/pypi/v/meshplex.svg)](https://pypi.org/project/meshplex)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshplex.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/meshplex)

<p align="center">
  <img src="https://nschloe.github.io/meshplex/meshplex-logo.svg" width="20%">
</p>

Compute all sorts of interesting points, areas, and volumes in triangular and
tetrahedral meshes, with a focus on efficiency. Useful in many contexts, e.g.,
finite-element and finite-volume computations.

meshplex is used in [optimesh](https://github.com/nschloe/optimesh) and
[pyfvm](https://github.com/nschloe/pyfvm).

### Quickstart

```python
import numpy
import meshplex

# create a simple MeshTri instance
points = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
cells = numpy.array([[0, 1, 2]])
mesh = meshplex.MeshTri(points, cells)
# or read it from a file
# mesh = meshplex.read('pacman.msh')

# triangle volumes
print(mesh.cell_volumes)

# circumcenters, centroids, incenters
print(mesh.cell_circumcenters)
print(mesh.cell_centroids)
print(mesh.cell_incenters)

# circumradius, inradius, cell quality, angles
print(mesh.cell_circumradius)
print(mesh.cell_inradius)
print(mesh.cell_quality)  # d * inradius / circumradius (min 0, max 1)
print(mesh.angles)

# control volumes, centroids
print(mesh.control_volumes)
print(mesh.control_volume_centroids)

# covolume/edge length ratios
print(mesh.ce_ratios)

# flip edges until the mesh is Delaunay
mesh.flip_until_delaunay()

# show the mesh
mesh.show()
```

meshplex works much the same way with tetrahedral meshes. For a documentation of all
classes and functions, see [readthedocs](https://meshplex.readthedocs.io/).

(For mesh creation, check out
[this list](https://github.com/nschloe/awesome-scientific-computing#meshing)).

### Installation

meshplex is [available from the Python Package
Index](https://pypi.org/project/meshplex/), so simply type
```
pip3 install --user meshplex
```
to install.

### Testing

To run the meshplex unit tests, check out this repository and type
```
pytest
```

### License

meshplex is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
