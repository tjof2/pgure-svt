# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------


project = "pgure-svt"
copyright = "2015-2020, Tom Furnival"
author = "Tom Furnival"

# The full version, including alpha/beta/rc tags
release = "0.6.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Generate autosummary even if no references
autosummary_generate = True
autodoc_default_flags = ["members"]
autoclass_content = "both"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# Code pygments style
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "sphinx_rtd_theme"
html_theme = "bootstrap"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "navbar_sidebarrel": False,
    "navbar_links": [
        ("Install", "install"),
        ("Examples", "examples"),
        ("API", "api"),
        ("GitHub", "https://github.com/tjof2/pgure-svt", True),
    ],
    "bootswatch_theme": "sandstone",
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()


# -- Options for gallery ---------------------------------------------------

sphinx_gallery_conf = {
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("pguresvt", "numpy"),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {"pguresvt": None},
}
