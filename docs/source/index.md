---
myst:
  html_meta:
    description: "sigmaepsilon.math is a Python library for applied mathematics: vector and tensor algebra in arbitrary frames, sparse arrays, linear and nonlinear optimization, approximation and graph theory."
---

# **SigmaEpsilon.Math** - A Python Library for Applied Mathematics in Natural Sciences

```{toctree}
:maxdepth: 1
:hidden:

User Guide <user_guide/index>
Theory Guide <theory/theory_linalg>
API Reference <api>
Development <development>
```

**Version**: {{ version }}

**Useful links**:
{doc}`Installation <user_guide/installation>` |
{ref}`Getting Started <getting_started>` |
[Issue Tracker](https://github.com/sigma-epsilon/sigmaepsilon.math/issues) |
[Source Repository](https://github.com/sigma-epsilon/sigmaepsilon.math) |
{doc}`Using with LLMs <LLMs>`

The [sigmaepsilon.math](https://sigmaepsilonmath.readthedocs.io/en/latest/) library is the mathematical department of the `SigmaEpsilon` project, a collection of
Python libraries for computational mechanics and other disciplines. The library includes tools that emerged during
work on other parts of the `SigmaEpsilon` ecosystem, but are general enough to be used in other projects.

Implementations are fast as they rely on the vector math capabilities of [NumPy](https://numpy.org/doc/stable/index.html), while other computationally
sensitive calculations are JIT-compiled using [Numba](https://numba.pydata.org/). Here and there we also use [NetworkX](https://networkx.org/documentation/stable/index.html), [SciPy](https://scipy.org/), [SymPy](https://www.sympy.org/en/index.html) and [scikit-learn](https://scikit-learn.org/stable/).

(highlights)=

## Highlights

```{include} _snippets/highlights.md
```

## Contents

::::::{grid} 2

:::::{grid-item-card}
:img-top: _static/index-images/getting_started.svg

Getting Started
^^^

The getting started guide is your entry point. It helps you to set up
a development environment and make the first steps with the library.

+++

```{button-ref} user_guide/introduction
:ref-type: doc
:expand:
:color: secondary
:click-parent:

Get me started
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/user_guide.svg

User Guide
^^^

The user guide provides a detailed walkthrough of the library, touching
the key features with useful background information and explanation.

+++

```{button-ref} user_guide/index
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the user guide
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/api.svg

API Reference
^^^

The reference guide contains a detailed description of the functions,
modules, and objects included in the library. It describes how the
methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

+++

```{button-ref} api
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the reference guide
```
:::::

:::::{grid-item-card}
:img-top: _static/index-images/contributor.svg

Contributor's Guide
^^^

Want to add to the codebase? The contributing guidelines will guide you through
the process of improving the library.

+++

```{button-ref} development
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the contributor's guide
```
:::::

:::::{grid-item-card}
:columns: 12
:img-top: _static/index-images/llm.svg

Using the Library with LLMs
^^^

Working with an AI assistant? Feed it the LLM-friendly version of this documentation: download `llms.txt` or copy its URL for quick access.

+++

```{button-ref} LLMs
:ref-type: doc
:expand:
:color: secondary
:click-parent:

To the LLM guide
```
:::::

::::::

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
