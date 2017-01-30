# voropy

[![Build Status](https://travis-ci.org/nschloe/voropy.svg?branch=master)](https://travis-ci.org/nschloe/voropy)
[![Code Health](https://landscape.io/github/nschloe/voropy/master/landscape.png)](https://landscape.io/github/nschloe/voropy/master)
[![codecov](https://codecov.io/gh/nschloe/voropy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/voropy)
[![PyPi Version](https://img.shields.io/pypi/v/voropy.svg)](https://pypi.python.org/pypi/voropy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/voropy.svg?style=social&label=Star&maxAge=2592000)](https://github.com/nschloe/voropy)

Compute Voronoi tesselations and everything else you need for finite-volume
discretizations. Do it fast, too.

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

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

voropy is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
