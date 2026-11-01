---
myst:
  html_meta:
    description: "How to document sigmaepsilon.math: NumPy-style docstrings, MyST Markdown and Jupyter Notebook sources, page descriptions for llms.txt, and building the docs with Sphinx."
---

# How to Document Your Code

Documenting your code is an essential part of the development process. It helps
you and other developers understand the codebase and its functionality. In this
section, we will discuss how to document your code using `Sphinx`.

Documenting your code consists of the following steps:

1. **Writing docstrings**: Docstrings are the first step in documenting your
   code. They are used to describe the purpose and functionality of a function,
   class, or module.

2. **Writing source files for Sphinx**: Sphinx is a documentation generator
   that supports various file formats. You need to write
   source files in either of these formats to generate documentation using Sphinx.

3. **Building the documentation**: Once you have written the source files, you
   can build the documentation using Sphinx. Sphinx will generate HTML, PDF, or
   other formats based on the source files.

In the remaining sections, we will discuss each of these steps in detail.

## Writing docstrings

Docstrings are used to describe the purpose and functionality of a function,
class, or module. They are written as a string literal that is the first
statement in a function, class, or module. Docstrings can be accessed using the
`__doc__` attribute of the object.

There are several formats for writing docstrings, such as `Google`, `Numpy`,
and `reStructuredText`. In the projects developed by SigmaEpsilon, we use the
`NumPy` format for writing docstrings. See {doc}`docstrings` for the details and
for how to check your docstrings locally.

A few resources to help you write good docstrings:

- [PEP 257](https://www.python.org/dev/peps/pep-0257/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

## Writing source files for Sphinx

In projects belonging to SigmaEpsilon, we write the documentation in
[MyST Markdown](https://myst-parser.readthedocs.io/) and `Jupyter Notebooks`.
MyST is a flavour of Markdown that supports everything Sphinx needs, like
directives (admonitions, `toctree`, `autoclass`, etc.), roles (cross-references,
citations, math) and numbered equations. Notebooks are rendered by
[MyST-NB](https://myst-nb.readthedocs.io/), and their markdown cells use the
very same MyST syntax, so there is a single markup language for everything.
`Jupyter Notebooks` are used to write tutorials and examples for the library.

A few examples of the syntax:

````markdown
(my-label)=

## A section with a label

A reference to {ref}`the section above <my-label>`, a link to another
{doc}`page <../user_guide/index>` and to a class, like
{class}`~sigmaepsilon.math.linalg.ReferenceFrame`.

```{note}
This is a note.
```
````

You can edit the source files using any text editor, such as `VS Code` or
`Jupyter Lab`.

```{note}
Notebooks are rendered with the outputs stored in them, they are not executed
during the build. Make sure to run a notebook and save it with its outputs
before committing it.
```

## The API reference

The pages of the API reference live in `docs/source/api/`, one page per topic, and
document the objects with the autodoc directives (`autoclass`, `automodule`,
`autofunction`), wrapped in `{eval-rst}` blocks:

````markdown
```{eval-rst}
.. autoclass:: sigmaepsilon.math.optimize.pso.ParticleSwarmOptimization
    :members:
```
````

To document a new object, add it to the page of the topic it belongs to.

## Page descriptions and LLM-friendly documentation

Every page starts with a front matter that holds a one-sentence description of the page:

```markdown
---
myst:
  html_meta:
    description: "How to install sigmaepsilon.math from PyPI with pip and check the installation."
---
```

For notebooks, the same goes into the notebook-level metadata (in Jupyter Lab,
*Property Inspector* → *Advanced Tools* → *Notebook metadata*):

```json
"myst": {
  "html_meta": {
    "description": "How to define and solve linear programming problems."
  }
}
```

The description ends up in the `<meta name="description">` tag of the HTML page, and
it is also the description of the page in `llms.txt`. Without it, a description is
extracted from the content of the page, which is usually much worse, so remember to
add one to every new page.

The [sphinx-llm](https://github.com/NVIDIA/sphinx-llm) extension (`"sphinx_llm.txt"`
in `docs/source/conf.py`) generates the LLM-friendly versions of the documentation
alongside the HTML output:

| File | Description |
| --- | --- |
| `llms.txt` | Index listing the usage pages with their titles and descriptions |
| `llms-full.txt` | The usage pages concatenated into a single Markdown file |
| `{pagename}.html.md` | Per-page Markdown, one file per HTML page |

To keep the context of LLMs small, only the pages that show how to use the library
(the user guide, the examples and the API reference) make it into
`llms.txt` and `llms-full.txt`. The rest is listed in `llms_txt_exclude` in `conf.py`;
those pages still get their per-page Markdown file. When adding a page that doesn't
show the usage of the library, add it to `llms_txt_exclude` as well.

The Markdown is produced by a secondary build with a Markdown builder. Nodes the
builder doesn't know are dropped with an `unknown node type` warning; `setup()` in
`conf.py` teaches it the ones used in these docs (admonitions, captions and
sphinx-design cards), so if a new kind of content triggers the warning, add a
handler there. The {doc}`Using the Library with LLMs <../LLMs>` page explains to the
readers how to use these files, and `docs/source/_static/js/llm_tools.js` adds the
*Download Markdown* link to the sidebar of every page.

## Building the documentation

Once you have written the source files, you can build the documentation using
Sphinx. For this, navigate to the `docs` directory and run the following command:

```shell
uv run make html
```

Or run this command from the root directory of the project:

```shell
uv run sphinx-build -b html docs/source docs/build
```

These commands will generate the HTML documentation in the `docs/build` directory. To open
the documentation in a browser, navigate to the `docs/build/html` directory and open
`index.html` in a browser.
