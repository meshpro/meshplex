# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Sphinx 1.* compat (for readthedocs)
master_doc = "index"


# -- Project information -----------------------------------------------------

project = "meshplex"
copyright = "2017-2019, Nico Schlömer"
author = "Nico Schlömer"

#
# https://packaging.python.org/single_source_version/
this_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
about_file = os.path.join(this_dir, "..", "meshplex", "__about__.py")
with open(about_file) as f:
    exec(f.read(), about)
# The short X.Y version.
# version = ".".join(about["__version__"].split(".")[:2])
# The full version, including alpha/beta/rc tags.
release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "sphinx.ext.todo",
    # "sphinx.ext.coverage",
    # "sphinx.ext.pngmath",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_favicon = "_static/meshplex-32x32.ico"

html_theme_options = {
    "logo": "meshplex-logo.svg",
    "github_user": "nschloe",
    "github_repo": "meshplex",
    "github_banner": True,
    "github_button": False,
}
