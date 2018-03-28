# voropy

[![Build Status](https://travis-ci.org/nschloe/voropy.svg?branch=master)](https://travis-ci.org/nschloe/voropy)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/voropy.svg)](https://codecov.io/gh/nschloe/voropy)
[![Codacy grade](https://img.shields.io/codacy/grade/b524f1e339244cf9a429784681a7f248.svg)](https://app.codacy.com/app/nschloe/voropy/dashboard)
[![PyPi Version](https://img.shields.io/pypi/v/voropy.svg)](https://pypi.python.org/pypi/voropy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/voropy.svg?logo=github&style=social&label=Stars)](https://github.com/nschloe/voropy)

![](https://nschloe.github.io/voropy/logo-180x180.png)

Compute Voronoi tesselations and everything else you need for finite-volume
discretizations. Do it fast, too.

### Usage

Given a triangular or tetrahedral mesh, voropy computes

 * covolumes,
 * control volumes,
 * cell circumcenters,
 * the surface areas,
 * control volume circumcenters,

and much more.

To use voropy, simple read a mesh (e.g., [this
pacman](https://sourceforge.net/projects/meshzoo-data/files/pacman.msh/download)):
```python
mesh = voropy.read('pacman.msh')

print(mesh.node_coords)
print(mesh.control_volumes)

mesh.show()
```
(For mesh creation, check out [pygmsh](https://github.com/nschloe/pygmsh),
[mshr](https://bitbucket.org/fenics-project/mshr),
[frentos](https://github.com/nschloe/frentos),
[meshzoo](https://github.com/nschloe/meshzoo) or any other tool.)

#### Mesh smoothing

voropy comes with a smoothing tool for triangular meshes after
[Lloyd](https://en.wikipedia.org/wiki/Lloyd's_algorithm) that can dramatically
improve the quality your mesh. To use, simply type
```
mesh_smoothing --verbose -t 1.0e-3 pacman.msh out.msh
```
![](https://nschloe.github.io/voropy/lloyd.gif)

### Installation

#### Python Package Index

voropy is [available from the Python Package
Index](https://pypi.python.org/pypi/voropy/), so simply type
```
pip install -U voropy
```
to install or upgrade.

#### Manual installation

Download voropy from
[the Python Package Index](https://pypi.python.org/pypi/voropy/).
Place voropy in a directory where Python can find it (e.g.,
`$PYTHONPATH`).  You can install it system-wide with
```
python setup.py install
```

### Testing

To run the voropy unit tests, check out this repository and type
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

voropy is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
