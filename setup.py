# -*- coding: utf-8 -*-
#
#from distutils.core import setup
from setuptools import setup, find_packages
import os
import codecs

from pyfvm import __version__, __author__, __author_email__


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content

setup(
    name='PyFVM',
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    packages=find_packages(),
    description='Finite Volume Discretizations for Python',
    long_description=read('README.rst'),
    url='https://github.com/nschloe/pyfvm',
    download_url='https://github.com/nschloe/pyfvm/releases',
    license='License :: OSI Approved :: MIT License',
    platforms='any',
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'vtk'
        ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics'
        ]
    )
