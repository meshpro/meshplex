# voropy

[![Build Status](https://travis-ci.org/nschloe/voropy.svg?branch=master)](https://travis-ci.org/nschloe/voropy)
[![Code Health](https://landscape.io/github/nschloe/voropy/master/landscape.png)](https://landscape.io/github/nschloe/voropy/master)
[![codecov](https://codecov.io/gh/nschloe/voropy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/voropy)
[![PyPi Version](https://img.shields.io/pypi/v/voropy.svg)](https://pypi.python.org/pypi/voropy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/voropy.svg?style=social&label=Star&maxAge=2592000)](https://github.com/nschloe/voropy)

![voropy logo](https://nschloe.github.io/voropy/logo-180x180.png)

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
mesh = voropy.reader.read('pacman.msh')

print(mesh.node_coords)
print(mesh.control_volumes)

from matplotlib import pyplot as plt
mesh.show()
plt.show()
```
(For mesh creation, check out [pygmsh](https://github.com/nschloe/pygmsh),
[mshr](https://bitbucket.org/fenics-project/mshr),
[frentos](https://github.com/nschloe/frentos),
[meshzoo](https://github.com/nschloe/meshzoo) or any other tool.)

#### Lloyd smoothing

voropy comes with a smoothing tool for triangular meshes after
[Lloyd](https://en.wikipedia.org/wiki/Lloyd's_algorithm) that can dramatically
improve the quality your mesh. To use, simply type
```
lloyd_smoothing --verbose -t 1.0e-3 pacman.msh out.msh
```
![Lloyd's algorithm](https://nschloe.github.io/voropy/lloyd.gif)

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
