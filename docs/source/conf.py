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
# Also explicitly add the src directory
sys.path.insert(0, os.path.abspath('../../src'))

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
    'sphinx_design',           # Advanced layout components
    'sphinx_togglebutton',     # Toggle content visibility
    'sphinx_tabs.tabs',        # Tabbed content
    'sphinx_autodoc_typehints',# Better type hints
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

# Mock imports for modules that cannot be imported during doc generation
autodoc_mock_imports = ['src', 'src.basis', 'src.nn']

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
html_theme = 'sphinx_book_theme'  # Use the modern Sphinx Book Theme
html_static_path = ['_static']
html_favicon = '_static/images/icon.svg'
html_logo = '_static/images/icon.svg'
html_theme_options = {
    # Repository settings
    "repository_url": "https://github.com/Pytorch-KAN/pytorch_kan",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "path_to_docs": "docs/source",
    
    # Navigation settings
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "toc_title": "On this page",
    
    # Announcement banner - removed as requested
    # "announcement": "This is a modern implementation of Kolmogorov-Arnold Networks (KAN)!",
    
    # Theme settings
    "use_sidenotes": True,
    "use_download_button": True,
    
    # Footer content
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# Custom CSS/JS files
html_css_files = [
    'custom.css',
]

# Add JavaScript files - prevent-flash.js loads early to prevent color flashing
html_js_files = [
    'prevent-flash.js',
]

# Path to custom templates
templates_path = ['_templates']

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

# -- Options for sphinx-autodoc-typehints ------------------------------------
typehints_fully_qualified = False
typehints_document_rtype = True
always_document_param_types = True

# -- Options for sphinx-copybutton -------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True