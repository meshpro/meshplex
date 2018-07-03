# meshplex

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/meshplex/master.svg)](https://circleci.com/gh/nschloe/meshplex/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/meshplex.svg)](https://codecov.io/gh/nschloe/meshplex)
[![Codacy grade](https://img.shields.io/codacy/grade/b524f1e339244cf9a429784681a7f248.svg)](https://app.codacy.com/app/nschloe/meshplex/dashboard)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/meshplex.svg)](https://pypi.org/project/meshplex)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshplex.svg?logo=github&label=Stars)](https://github.com/nschloe/meshplex)

Compute Voronoi tesselations and everything else you need for finite-volume
discretizations. Do it fast, too.

### Usage

Given a triangular or tetrahedral mesh, meshplex computes

 * covolumes,
 * control volumes,
 * cell circumcenters,
 * the surface areas,
 * control volume circumcenters,

and much more.

To use meshplex, simple read a mesh (e.g., [this
pacman](https://sourceforge.net/projects/meshzoo-data/files/pacman.msh/download)):
```python
mesh = meshplex.read('pacman.msh')

print(mesh.node_coords)
print(mesh.control_volumes)

mesh.show()
```
(For mesh creation, check out [pygmsh](https://github.com/nschloe/pygmsh),
[mshr](https://bitbucket.org/fenics-project/mshr),
[pygalmesh](https://github.com/nschloe/pygalmesh),
[meshzoo](https://github.com/nschloe/meshzoo),
[optimesh](//github.com/nschloe/optimesh) or any other tool.)


### Installation

meshplex is [available from the Python Package
Index](https://pypi.org/project/meshplex/), so simply type
```
pip install -U meshplex
```
to install or upgrade.

### Testing

To run the meshplex unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

meshplex is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
