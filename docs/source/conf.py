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
import re
from datetime import date
from pathlib import Path
import warnings

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.ext import intersphinx
from sphinx_design.shared import PassthroughTextElement
from sphinx_markdown_builder.contexts import SubContextParams, WrappedContext

import sigmaepsilon.math as library

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
version = library.__version__
release = "v" + library.__version__


def _md_visit_classifier(self, node: nodes.classifier) -> None:
    self.add(" : ")


def _md_depart_classifier(self, node: nodes.classifier) -> None:
    pass


def _md_admonition_handlers(title: str) -> tuple:
    def visit(self, node: nodes.Admonition) -> None:
        self._push_box(title)

    def depart(self, node: nodes.Admonition) -> None:
        self._pop_context(node)

    return visit, depart


def _md_visit_generic_admonition(self, node: nodes.admonition) -> None:
    # The title of a generic admonition is its first child, so it becomes the
    # box heading and only the remaining children are rendered as the body.
    title = node[0].astext() if isinstance(node[0], nodes.title) else "NOTE"
    self._push_box(title.upper())
    for child in node.children[1:]:
        child.walkabout(self)
    self._pop_context(node)
    raise nodes.SkipNode


def _md_depart_generic_admonition(self, node: nodes.admonition) -> None:
    pass


def _md_visit_caption(self, node: nodes.caption) -> None:
    self._push_context(WrappedContext("*", params=SubContextParams(2, 2)))


def _md_depart_caption(self, node: nodes.caption) -> None:
    self._pop_context(node)


def _is_card_link(node: nodes.Node) -> bool:
    return any(
        "sd-stretched-link" in ref.get("classes", [])
        for ref in node.findall(nodes.reference)
    )


def _md_visit_passthrough(self, node: PassthroughTextElement) -> None:
    # sphinx-design wraps both the title and the link of a card into these
    # elements. The link of a card has the title of the linked page as its text,
    # so the title of a card is only kept if the card has no link, otherwise it
    # would appear twice.
    if _is_card_link(node):
        self.ensure_eol(2)
        self._push_context(WrappedContext("", params=SubContextParams(2, 2)))
        return
    card = node.parent
    while card is not None and "sd-card" not in card.get("classes", []):
        card = card.parent
    if card is not None and _is_card_link(card):
        raise nodes.SkipNode
    self.ensure_eol(2)
    self._push_context(WrappedContext("**", params=SubContextParams(2, 2)))


def _md_depart_passthrough(self, node: PassthroughTextElement) -> None:
    self._pop_context(node)


def _md_visit_container(self, node: nodes.container) -> None:
    # Separate the sphinx-design cards, which are containers, by blank lines.
    if "sd-card" in node.get("classes", []):
        self.ensure_eol(2)


def _md_depart_container(self, node: nodes.container) -> None:
    if "sd-card" in node.get("classes", []):
        self.ensure_eol(2)


def _md_visit_toctree(self, node: addnodes.toctree) -> None:
    # Hidden toctrees are left in the doctree, but they have no content to show.
    raise nodes.SkipNode


# Reference targets that can't be resolved as they are, and what they stand for. Most
# of them come from the type hints, where the names are imported from third-party
# libraries under a shorter name than the one in their inventory.
_REFERENCE_ALIASES = {
    "ndarray": "numpy.ndarray",
    "Number": "numbers.Number",
    "Symbol": "sympy.core.symbol.Symbol",
    "sympy.Matrix": "sympy.matrices.dense.MutableDenseMatrix",
    "sympy.physics.vector.ReferenceFrame.orientnew": (
        "sympy.physics.vector.frame.ReferenceFrame.orientnew"
    ),
    "awkward.highlevel.Array": "awkward.Array",
    "FrameLike": "sigmaepsilon.math.linalg.meta.FrameLike",
    "FunctionLike": "sigmaepsilon.math.function.function.FunctionLike",
}


def _add_api_page_description(app: Sphinx, doctree: nodes.document) -> None:
    # The pages generated by autosummary have no front matter, so they get the first
    # sentence of the docstring of the documented object as their description, the
    # same one the summary tables show. Otherwise sphinx_llm would fall back to the
    # first line of the page in llms.txt, which is the "Bases: ..." line of a class.
    if "_autosummary/" not in app.env.docname:
        return
    for paragraph in doctree.findall(nodes.paragraph):
        text = " ".join(paragraph.astext().split())
        if not text or text.startswith("Bases:"):
            continue
        # the first sentence ends at a period followed by a capital letter, so
        # that abbreviations like "e.g." don't end it
        sentence = re.split(r"(?<!e\.g)(?<!i\.e)(?<=[\w)\]`])\.\s+(?=[A-Z])", text)
        sentence = sentence[0].rstrip(".") + "."
        doctree.insert(0, nodes.meta(name="description", content=sentence))
        return


def _resolve_aliased_reference(app, env, node, contnode):
    if node.get("refdomain") != "py":
        return None
    target = node.get("reftarget", "")
    if target not in _REFERENCE_ALIASES:
        return None
    node["reftarget"] = _REFERENCE_ALIASES[target]
    resolved = env.get_domain("py").resolve_xref(
        env,
        node["refdoc"],
        app.builder,
        node["reftype"],
        node["reftarget"],
        node,
        contnode,
    )
    return resolved or intersphinx.missing_reference(app, env, node, contnode)


def setup(app: Sphinx):
    app.connect("doctree-read", _add_api_page_description)
    app.connect("missing-reference", _resolve_aliased_reference)

    # Teach the Markdown builder used by sphinx_llm the nodes it does not
    # support out of the box, otherwise their content is dropped.
    md_builder = "llms-markdown"
    app.add_node(
        nodes.classifier,
        override=True,
        **{md_builder: (_md_visit_classifier, _md_depart_classifier)},
    )
    app.add_node(
        nodes.admonition,
        override=True,
        **{
            md_builder: (
                _md_visit_generic_admonition,
                _md_depart_generic_admonition,
            )
        },
    )
    app.add_node(
        addnodes.toctree,
        override=True,
        **{md_builder: (_md_visit_toctree, None)},
    )
    app.add_node(
        nodes.caption,
        override=True,
        **{md_builder: (_md_visit_caption, _md_depart_caption)},
    )
    app.add_node(
        PassthroughTextElement,
        override=True,
        **{md_builder: (_md_visit_passthrough, _md_depart_passthrough)},
    )
    app.add_node(
        nodes.container,
        override=True,
        **{md_builder: (_md_visit_container, _md_depart_container)},
    )
    for node_cls, title in [
        (nodes.tip, "TIP"),
        (nodes.danger, "DANGER"),
        (nodes.caution, "CAUTION"),
        (nodes.error, "ERROR"),
    ]:
        app.add_node(
            node_cls, override=True, **{md_builder: _md_admonition_handlers(title)}
        )


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.rsvgconverter",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    # LLM-friendly documentation (llms.txt, llms-full.txt and per-page Markdown)
    # https://github.com/NVIDIA/sphinx-llm
    "sphinx_llm.txt",
]

_docs_base_url = "https://sigmaepsilonmath.readthedocs.io/en/latest/"

# Without this, sphinx_llm falls back to the package metadata, i.e. the README.
llms_txt_description = (
    "sigmaepsilon.math is a Python library of applied mathematics tools: vector "
    "and tensor algebra in arbitrary (skew or orthogonal) reference frames, "
    "jagged arrays and a Numba-jittable CSR matrix, symbolic and numeric functions "
    "and relations, linear and mixed-integer programming, genetic algorithms, ant "
    "colony optimization, particle swarm optimization, differential evolution, "
    "(moving) least squares approximation and graph algorithms. Performance-critical "
    "parts are JIT-compiled with the optional Numba dependency.\n"
    "\n"
    "How to read this file:\n"
    "\n"
    "- Every link below points to a Markdown version of a documentation page.\n"
    "- Links are relative URLs, resolved against the directory that contains "
    "the llms.txt file you are reading.\n"
    "- Fetch the linked `.html.md` files directly; they contain the full page "
    "content as Markdown.\n"
    "- If you are reading a local copy, resolve the links against its directory "
    "on disk the same way.\n"
    "\n"
    f"Examples for the top-level index at {_docs_base_url}llms.txt:\n"
    "\n"
    "- `[API Reference](api.html.md)` resolves to "
    f"{_docs_base_url}api.html.md\n"
    "- `[User Guide](user_guide/index.html.md)` resolves to "
    f"{_docs_base_url}user_guide/index.html.md\n"
    "- `[Particle Swarm Optimization](user_guide/optimization/pso.html.md)` "
    f"resolves to {_docs_base_url}user_guide/optimization/pso.html.md\n"
    "\n"
    "Each subdirectory also has its own index (for example "
    f"{_docs_base_url}user_guide/llms.txt) whose links are relative to that "
    "subdirectory, so the same page is linked there as "
    "`optimization/pso.html.md`."
)

# Nodes with no meaningful Markdown representation: <meta> tags and the
# abbreviation used for the keyword-only "*" marker in signatures.
llms_txt_suppress_unknown_node_warnings = ["meta", "abbreviation"]

# Also generate llms-full.txt, the whole documentation in a single file.
llms_txt_full_build = True

# To keep the LLM contexts small, llms.txt and llms-full.txt only contain the pages
# that show how to use the library, and the API reference. The pages left out still
# get their per-page Markdown files, so they can be downloaded one by one.
llms_txt_exclude = [
    "index",  # the landing page, the overview is in llms_txt_description
    "theory/*",  # derivations, not needed to use the library
    "development",
    "developer_guide/*",
    "LLMs",
]

# Only the pages of the API reference have summary tables. Scanning every page would
# also pick up the example in developer_guide/documenting.md and generate duplicate pages.
autosummary_generate = sorted(
    f"api/{path.name}" for path in (Path(__file__).parent / "api").glob("*.md")
)

# The summary tables of the modules only list the members in their __all__.
autosummary_ignore_module_all = False

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    ".DS_Store",
    # snippets pulled into other pages with the `include` directive
    "_snippets",
    # scratch notebooks used during development, not meant to be rendered
    "tests",
]

# The master toctree document.
master_doc = "index"

# -- MyST configuration ------------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$ math
    "amsmath",  # LaTeX amsmath environments (align, etc.)
    "colon_fence",  # ::: fences for directives (used by sphinx_design)
    "substitution",  # {{ version }} and the like
    "attrs_inline",
    "deflist",
]
myst_substitutions = {"version": version}
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# -- myst-nb configuration ---------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

# The notebooks are rendered with the outputs stored in them.
nb_execution_mode = "off"
nb_execution_allow_errors = True
nb_render_markdown_format = "myst"

language = "EN"

# See warnings about bad links
nitpicky = True
nitpick_ignore = [
    ("", "Pygments lexer name 'ipython' is not known"),
    ("", "Pygments lexer name 'ipython3' is not known"),
    # Numba has no Sphinx inventory to link to.
    ("py:class", "numba.typed.typeddict.Dict"),
    # type aliases used in the type hints, they have no documentation of their own
    ("py:class", "ArrayLike"),
    ("py:class", "SparseLike"),
    ("py:class", "BoundsLike"),
    # private base classes, shown in the "Bases:" line of their public subclasses
    ("py:class", "sigmaepsilon.math.function.metafunction.MetaFunction"),
    ("py:class", "sigmaepsilon.math.linalg.abstract.AbstractTensor"),
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "github-dark"
highlight_language = "python3"

intersphinx_mapping = {
    "python": (r"https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": (r"https://numpy.org/doc/stable/", None),
    "scipy": (r"https://docs.scipy.org/doc/scipy/", None),
    "sympy": (r"https://docs.sympy.org/latest/", None),
    "awkward": (r"https://awkward-array.org/doc/main/", None),
    "networkx": (r"https://networkx.org/documentation/stable/", None),
    "matplotlib": (r"https://matplotlib.org/stable", None),
    "sphinx": (r"https://www.sphinx-doc.org/en/master", None),
    "pandas": (r"https://pandas.pydata.org/pandas-docs/stable/", None),
    "pydantic": (r"https://docs.pydantic.dev/latest/", None),
    "sigmaepsilon.core": (r"https://sigmaepsiloncore.readthedocs.io/en/latest/", None),
    "sigmaepsilon.deepdict": (
        r"https://sigmaepsilondeepdict.readthedocs.io/en/latest/",
        None,
    ),
}

# sphinx_copybutton configuration --------------------------------------------

copybutton_exclude = ".linenos, .gp, .go"
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

# -- MathJax Configuration -------------------------------------------------

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}

# -- Numbering of figures, tables and equations -----------------------------
numfig = True
math_numfig = True  # needs numfig
# only number the labeled equations
math_number_all = False
numfig_secnum_depth = 1  # 1=section, 2=subsection, etc.

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
    "use_edit_page_button": True,
    "show_toc_level": 2,
}
html_js_files = [
    "require.min.js",
    "custom.js",
    "js/llm_tools.js",
]
html_css_files = ["custom.css"]
html_context = {
    "default_mode": "light",
    # "Edit on GitHub" button, linking each page (and notebook) to its source
    "github_user": "sigma-epsilon",
    "github_repo": project,
    "github_version": "main",
    "doc_path": "docs/source",
}
html_static_path = ["_static"]
