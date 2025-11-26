# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# NOUVEAU : Imports pour trouver le code source (étape cruciale)
import os
import sys

# Indique à Sphinx de chercher le code dans le répertoire parent (la racine du projet)
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "camera_classify"
copyright = "2025, Mathys BAUCHET"
author = "Mathys BAUCHET"
release = "1.0.0"  # Utiliser une version simple, le format de date n'est pas standard

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# MODIFIÉ : Ajout des extensions nécessaires pour lire les docstrings Python
extensions = [
    "sphinx.ext.autodoc",  # Pour lire les docstrings
    "sphinx.ext.napoleon",  # Si vous utilisez le style NumPy/Google Docstrings
    "sphinx_autodoc_typehints",  # Pour afficher les annotations de type Python
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "fr"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# MODIFIÉ : Utiliser un thème plus moderne (recommandé)
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Configuration spécifique à Autodoc
autoclass_content = "both"
autodoc_member_order = "bysource"
