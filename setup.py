# -*- coding: utf-8 -*-
#
from setuptools import setup, find_packages
import os
import codecs

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, 'voropy', '__about__.py')) as f:
    exec(f.read(), about)


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
    name='voropy',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=find_packages(),
    description='Finite Volume Discretizations for Python',
    long_description=read('README.rst'),
    url='https://github.com/nschloe/voropy',
    download_url='https://github.com/nschloe/voropy/releases',
    license='License :: OSI Approved :: MIT License',
    platforms='any',
    install_requires=[
        'matplotlib',
        'meshio',
        'meshzoo',  # only required by tests
        'numpy',
        'scipy',
        ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics'
        ]
    scripts=[
        'tools/lloyd_smoothing'
        ]
    )
