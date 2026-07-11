# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# For ideas:

# https://github.com/pradyunsg/furo/blob/main/docs/conf.py
# https://github.com/sphinx-gallery/sphinx-gallery/blob/master/doc/conf.py

# --------------------------------------------------------------------------

import sys
import os
from datetime import date
import warnings

import sigmaepsilon.math as library

from sphinx.config import Config

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = library.__pkg_name__
copyright = "2014-%s, Bence Balogh" % date.today().year
author = "Bence Balogh"


def setup(app: Config):
    app.add_config_value("project_name", project, "html")


# The short X.Y version.
version = library.__version__
# The full version, including alpha/beta/rc tags.
release = "v" + library.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Parses Markdown (.md) files as Sphinx documents, alongside reStructuredText.
    # pip install myst-parser for this.
    "myst_nb",
    # Measures and reports how long each document took to build; prints a
    # "slowest documents" summary at the end of the build.
    "sphinx.ext.duration",
    # Runs doctest-style (>>> ...) code blocks/examples embedded in docstrings
    # and documents as executable tests (via `make doctest`).
    "sphinx.ext.doctest",
    # Pulls docstrings from Python modules/classes/functions into the docs
    # (the basis for the API reference pages).
    "sphinx.ext.autodoc",
    # Enables cross-referencing to the docs of other Sphinx projects (see
    # `intersphinx_mapping` below), e.g. linking to NumPy/SciPy API docs.
    "sphinx.ext.intersphinx",
    # Lets autodoc understand NumPy- and Google-style docstrings (this project
    # uses NumPy style) and renders them as proper reStructuredText.
    "sphinx.ext.napoleon",
    #'sphinx_gallery.gen_gallery',
    #'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
    # Renders Jupyter notebooks (.ipynb) as documentation pages, executing
    # them (or using saved outputs) and embedding the resulting cells/plots.
    # "nbsphinx_link",  # for including notebook files from outside the sphinx source root
    # Adds a "copy to clipboard" button to code blocks; configured below to
    # strip prompts (>>>, $, In [1]:, ...) when copying.
    "sphinx_copybutton",
    # Renders LaTeX math (via MathJax); configured further down in
    # `mathjax3_config`.
    "sphinx.ext.mathjax",
    # Adds support for BibTeX citations/bibliographies (see
    # `bibtex_bibfiles`/`bibtex_default_style` below).
    "sphinxcontrib.bibtex",
    # Converts SVG images to PDF when building LaTeX/PDF output, since LaTeX
    # cannot embed SVGs directly.
    "sphinxcontrib.rsvgconverter",
    # Adds "[source]" links from API docs to highlighted source code pages.
    "sphinx.ext.viewcode",
    # Auto-generates summary tables/stub pages for documented modules,
    # classes and functions (paired with `autosummary_generate` below).
    "sphinx.ext.autosummary",
    # Enables `..todo::` directives and a `.. todolist::` summary of
    # outstanding TODOs in the docs.
    "sphinx.ext.todo",
    # Adds the `make coverage` builder, reporting which objects are missing
    # documentation.
    "sphinx.ext.coverage",
    # Allows defining shorthand link roles (e.g. `:issue:`) that expand to
    # full URLs via a template, instead of writing full links each time.
    "sphinx.ext.extlinks",
    # Provides UI components (cards, grids, badges, dropdowns, tabs, etc.)
    # for richer page layouts.
    "sphinx_design",
    # Adds `.. tab::` directives for inline tabbed content blocks.
    "sphinx_inline_tabs",
    # Executes embedded Matplotlib plotting code and inserts the resulting
    # figures as images in the docs.
    "matplotlib.sphinxext.plot_directive",
]

autosummary_generate = True

templates_path = ["_templates"]

exclude_patterns = ["_build"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

language = "EN"

# See warnings about bad links
nitpicky = True
nitpick_ignore = [
    ("", "Pygments lexer name 'ipython' is not known"),
    ("", "Pygments lexer name 'ipython3' is not known"),
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "github-dark"
highlight_language = "python3"

intersphinx_mapping = {
    "python": (r"https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": (r"https://numpy.org/doc/stable/", None),
    "scipy": (r"http://docs.scipy.org/doc/scipy/", None),
    "sympy": (r"https://docs.sympy.org/latest/", None),
    "matplotlib": (r"https://matplotlib.org/stable", None),
    "sphinx": (r"https://www.sphinx-doc.org/en/master", None),
    "pandas": (r"https://pandas.pydata.org/pandas-docs/stable/", None),
    "sigmaepsilon.core": (r"https://sigmaepsiloncore.readthedocs.io/en/latest/", None),
    "sigmaepsilon.deepdict": (r"https://sigmaepsilondeepdict.readthedocs.io/en/latest/", None),
}

# sphinx_copybutton configuration --------------------------------------------

copybutton_exclude = '.linenos, .gp, .go'
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# napoleon config ---------------------------------------------------------

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = True

# -- bibtex configuration -------------------------------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# If no encoding is specified, utf-8-sig is assumed.
# bibtex_encoding = 'latin'

# -- MathJax Configuration -------------------------------------------------

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# -- Image scapers configuration -------------------------------------------------

image_scrapers = ("matplotlib",)

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": f"https://github.com/sigma-epsilon/{project}",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPi",
            "url": f"https://pypi.org/project/{project}/",
            "icon": "fas fa-box-open",
            "type": "fontawesome",
        },
    ],
    "logo": {
        # Because the logo is also a homepage link, including "home" in the alt text is good practice
        "text": "SigmaEpsilon.Math",
    },
}
html_js_files = [
    "require.min.js",
    "custom.js",
]
html_css_files = ["custom.css"]
html_context = {"default_mode": "light"}
html_static_path = ["_static"]

# -- Options for MyST parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "dollarmath",   # $...$ and $$...$$ math
    "amsmath",      # LaTeX amsmath environments (align, etc.)
    "colon_fence",  # ::: fences for directives (used by sphinx_design)
]

# -- Options for myst_nb -------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/

nb_execution_allow_errors = True
nb_execution_mode = "off"