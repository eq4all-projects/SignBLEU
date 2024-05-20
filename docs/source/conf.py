# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SignBLEU'
copyright = '2024, EQ4ALL'
author = 'EQ4ALL'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
#html_theme_options = {
#    'prev_next_buttons_location': None,
#}

#html_theme = 'sphinx_book_theme'
#html_theme_options = {
#    'show_toc_level': 2,
#    'toc_title': 'Contents',
#}

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'show_toc_level': 2,
}

html_static_path = ['_static']
html_css_files = ['css/custom.css']
