"""Sphinx configuration for SPADES documentation."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
# Enable documentation-only import fallback when optional unit packages are absent.
os.environ.setdefault("SPADES_ALLOW_UNIT_FALLBACK", "1")

project = "SPADES"
author = "SPADES contributors"
copyright = "2026, SPADES contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Compiled and optional dependencies are mocked so docs can build in clean CI jobs.
autodoc_mock_imports = [
    "spades.dhfs_wrapper",
    "spades.radial_wrapper",
    "numpy",
    "periodictable",
    "scipy",
    "numba",
    "tqdm",
    "mpmath",
    "matplotlib",
    "yaml",
]

html_theme = "alabaster"
html_static_path = ["_static"]
