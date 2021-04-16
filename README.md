<p align="center">
  <a href="https://github.com/nschloe/meshplex"><img alt="meshplex" src="https://nschloe.github.io/meshplex/meshplex-logo.svg" width="60%"></a>
  <p align="center">Fast tools for simplex meshes.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/meshplex.svg?style=flat-square)](https://pypi.org/project/meshplex)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/meshplex.svg?style=flat-square)](https://pypi.org/pypi/meshplex/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshplex.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/meshplex)
[![PyPi downloads](https://img.shields.io/pypi/dm/meshplex.svg?style=flat-square)](https://pypistats.org/packages/meshplex)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)
[![Documentation Status](https://readthedocs.org/projects/meshplex/badge?style=flat-square&version=latest)](https://readthedocs.org/projects/meshplex/?badge=latest)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/meshplex/ci?style=flat-square)](https://github.com/nschloe/meshplex/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/meshplex.svg?style=flat-square)](https://codecov.io/gh/nschloe/meshplex)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/meshplex.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/meshplex)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Compute all sorts of interesting points, areas, and volumes in simplex (triangle,
tetrahedral, n-simplex) meshes of any dimension, with a focus on efficiency. Useful in
many contexts, e.g., finite-element and finite-volume computations.

meshplex is used in [optimesh](https://github.com/nschloe/optimesh) and
[pyfvm](https://github.com/nschloe/pyfvm).

### Quickstart

meshplex can compute the following data:
```python
import meshplex

# create a simple Mesh instance
points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
cells = [[0, 1, 2]]
mesh = meshplex.Mesh(points, cells)
# or read it from a file
# mesh = meshplex.read("pacman.vtk")

# triangle volumes, heights
print(mesh.cell_volumes)
print(mesh.signed_cell_volumes)
print(mesh.cell_heights)

# circumcenters, centroids, incenters
print(mesh.cell_circumcenters)
print(mesh.cell_centroids)
print(mesh.cell_incenters)

# circumradius, inradius, cell quality
print(mesh.cell_circumradius)
print(mesh.cell_inradius)
print(mesh.q_radius_ratio)  # d * inradius / circumradius (min 0, max 1)

# control volumes, centroids
print(mesh.control_volumes)
print(mesh.control_volume_centroids)

# covolume/edge length ratios
print(mesh.ce_ratios)

# count Delaunay violations
print(mesh.num_delaunay_violations)

# removes some cells
mesh.remove_cells([0])
```
For triangular meshes (`MeshTri`), meshplex also has some mesh manipulation routines:
<!--exdown-skip-->
```python
mesh.show()  # show the mesh
mesh.angles  # compute angles
mesh.flip_until_delaunay()  # flips edges until the mesh is Delaunay
```

For a documentation of all classes and functions, see
[readthedocs](https://meshplex.readthedocs.io/).

(For mesh creation, check out
[this list](https://github.com/nschloe/awesome-scientific-computing#meshing)).

### Plotting

#### Triangles
<img src="https://nschloe.github.io/meshplex/pacman.png" width="30%">

<!--exdown-skip-->
```python
import meshplex

mesh = meshplex.read("pacman-optimized.vtk")
mesh.show(
    # show_coedges=True,
    # control_volume_centroid_color=None,
    # mesh_color="k",
    # nondelaunay_edge_color=None,
    # boundary_edge_color=None,
    # comesh_color=(0.8, 0.8, 0.8),
    show_axes=False,
)
```

#### Tetrahedra
<img src="https://nschloe.github.io/meshplex/tetra.png" width="30%">

<!--exdown-skip-->
```python
import numpy as np
import meshplex

# Generate tetrahedron
points = (
    np.array(
        [
            [1.0, 0.0, -1.0 / np.sqrt(8)],
            [-0.5, +np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
            [-0.5, -np.sqrt(3.0) / 2.0, -1.0 / np.sqrt(8)],
            [0.0, 0.0, np.sqrt(2.0) - 1.0 / np.sqrt(8)],
        ]
    )
    / np.sqrt(3.0)
)
cells = [[0, 1, 2, 3]]

# Create mesh object
mesh = meshplex.MeshTetra(points, cells)

# Plot cell 0 with control volume boundaries
mesh.show_cell(
    0,
    # barycenter_rgba=(1, 0, 0, 1.0),
    # circumcenter_rgba=(0.1, 0.1, 0.1, 1.0),
    # circumsphere_rgba=(0, 1, 0, 1.0),
    # incenter_rgba=(1, 0, 1, 1.0),
    # insphere_rgba=(1, 0, 1, 1.0),
    # face_circumcenter_rgba=(0, 0, 1, 1.0),
    control_volume_boundaries_rgba=(1.0, 0.0, 0.0, 1.0),
    line_width=3.0,
)
```

### Installation

meshplex is [available from the Python Package
Index](https://pypi.org/project/meshplex/), so simply type
```
pip install meshplex
```
to install.

### License
This software is published under the [GPLv3
license](https://www.gnu.org/licenses/gpl-3.0.en.html).
