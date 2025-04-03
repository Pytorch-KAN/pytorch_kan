"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# Add the project root directory to the path so that Sphinx can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'PyTorch KAN'
copyright = f'{datetime.now().year}, PyTorch KAN Team'
author = 'PyTorch KAN Team'
release = '0.1.0'  # Current version of your package

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.viewcode',     # Add links to the source code
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',      # Support for math equations
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.githubpages',  # Creates .nojekyll file when building for GitHub Pages
    'myst_parser',             # Support for Markdown files
    'sphinx_copybutton',       # Add copy buttons to code blocks
    'nbsphinx',                # Include Jupyter notebooks
]

# Mappings for intersphinx to link to external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Configure autodoc
autodoc_member_order = 'bysource'  # Document members in source code order
autodoc_typehints = 'description'  # Put typehints in the parameter description
autodoc_default_options = {
    'members': True,           # Document all members
    'undoc-members': True,     # Document members without docstrings
    'show-inheritance': True,  # Show base classes
    'special-members': '__init__',  # Include __init__ method
}

# Napoleon settings for parsing docstrings
napoleon_google_docstring = True   # Parse Google-style docstrings
napoleon_numpy_docstring = True    # Parse NumPy-style docstrings
napoleon_include_init_with_doc = False  # Don't include __init__ docstring separately
napoleon_include_private_with_doc = False  # Don't include private members
napoleon_include_special_with_doc = True   # Include special members (like __call__)
napoleon_use_admonition_for_examples = True  # Use admonition for examples
napoleon_use_admonition_for_notes = True     # Use admonition for notes
napoleon_use_admonition_for_references = False  # Don't use admonition for references
napoleon_use_ivar = False     # Don't use :ivar: role
napoleon_use_param = True     # Use :param: role
napoleon_use_rtype = True     # Use :rtype: role
napoleon_use_keyword = True   # Use :keyword: role
napoleon_preprocess_types = False  # Don't preprocess types

# Add support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns to exclude from the documentation
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'preamble': r'''
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{amsthm}
    ''',
}

# -- Options for MathJax output ----------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'