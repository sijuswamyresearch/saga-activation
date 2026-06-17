# Configuration file for the Sphinx documentation builder.
import os, sys
sys.path.insert(0, os.path.abspath('../src'))

# Mock imports so Sphinx doesn't crash if these aren't installed on the doc server
autodoc_mock_imports = ["torch", "torchvision", "triton"] 

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/_build']
project   = "SAGA"
copyright = "2026, Siju K.S. et al."
author    = "Siju K.S., Vipin Venugopal, Mithun Kumar Kar, Jayakrishnan Anandakrishnan"
release   = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

html_theme   = "sphinx_rtd_theme"
html_title   = "SAGA Documentation"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True