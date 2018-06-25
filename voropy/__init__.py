# -*- coding: utf-8 -*-
#
from __future__ import print_function

from voropy.__about__ import __version__, __author__, __author_email__

from . import mesh_line
from . import mesh_tri
from . import mesh_tetra

from .helpers import get_signed_simplex_volumes
from .reader import read

__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "mesh_line",
    "mesh_tri",
    "mesh_tetra",
    "read",
    "get_signed_simplex_volumes",
]

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")
