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

__version__ = '0.1.0'
__author__ = 'Nico Schlömer'
__author_email__ = 'nico.schloemer@gmail.com'
