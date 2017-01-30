# -*- coding: utf-8 -*-
#
from . import mesh_tri
from . import mesh_tetra
from .reader import read

__all__ = [
    'mesh_tri',
    'mesh_tetra',
    'reader'
    ]

from voropy.__about__ import (
    __version__,
    __author__,
    __author_email__
    )
