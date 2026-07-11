===================
Docstring Style
===================

Purpose
=======

The project uses `NumPy-style <https://numpydoc.readthedocs.io/en/latest/format.html>`_
docstrings (``Parameters``, ``Returns``, ``Attributes``, ``Raises``, ``Examples``, ``See Also``,
etc.) throughout the codebase. This is checked automatically on every pull request by
`Codacy <https://www.codacy.com>`_ using `pydocstyle <https://www.pydocstyle.org>`_, and issues
show up there as minor "Documentation" findings (e.g. ``D401: First line should be in imperative
mood``). Fixing these locally before pushing saves a round trip through CI.

Checking your code locally
===========================

Install `pydocstyle`, if you don't already have it:

.. code-block:: shell

   pip install pydocstyle

Then run it against the package, using the ``numpy`` convention (this matches the docstring style
already used in this codebase):

.. code-block:: shell

   pydocstyle --convention=numpy src/sigmaepsilon/math

To check a single file or a single directory instead of the whole package:

.. code-block:: shell

   pydocstyle --convention=numpy src/sigmaepsilon/math/optimize/bga.py
   pydocstyle --convention=numpy src/sigmaepsilon/math/linalg

No output and a zero exit code means the checked files are clean. Each finding is reported as a
file, line number, and a `pydocstyle error code <https://www.pydocstyle.org/en/stable/error_codes.html>`_
(``D1xx`` for missing docstrings, ``D2xx`` for whitespace/formatting, ``D4xx`` for content rules)
with a short explanation, for example:

.. code-block:: text

   src/sigmaepsilon/math/optimize/state.py:11 in public class `OptimizerState`:
           D401: First line should be in imperative mood (perhaps 'Return', not 'Returns')

Most common findings and how to fix them
==========================================

- **D100-D107 (missing docstring)** — every public module, class, method, function, and magic
  method needs a docstring. Add a concise, accurate one-liner describing what the code actually
  does.
- **D200 (one-liner should fit on one line)** — a docstring with a single sentence and no
  additional sections should be written as ``"""Do X."""``, not spread across multiple lines.
- **D205 (blank line required between summary and description)** — if a docstring has a longer
  description or a ``Parameters``/``Returns``/etc. section below the summary sentence, leave one
  blank line between the summary and everything that follows.
- **D400/D401 (first line: period, imperative mood)** — the summary line must end with a period
  and use the imperative mood, e.g. ``"""Return the dual frame."""`` rather than
  ``"""Returns the dual frame."""``.
- **D405 (section name capitalization)** — NumPy-style section headers must be capitalized exactly
  as ``Parameters``, ``Returns``, ``Attributes``, ``Raises``, ``Examples``, ``See Also``, etc.
- **D301 (use a raw string for docstrings containing backslashes)** — docstrings with LaTeX-style
  math (e.g. ``\mathbf``, ``\left``) must be written as raw strings, ``r"""..."""``, to avoid
  `SyntaxWarning: invalid escape sequence` and to have the backslashes render correctly.

Things to keep in mind while fixing docstrings
================================================

- These are docstring-only fixes: never change code behavior while fixing a docstring finding.
- Preserve the existing NumPy-style section structure and content — only fix the flagged
  formatting/wording issue, don't rewrite documentation that's already accurate.
- Doctest-style examples (lines starting with ``>>>``) are executed as tests by
  ``tests/test_docstrings.py``. Reformatting a docstring must not change the behavior or output of
  any ``>>>`` example it contains.
- Run Black on any file you touch, since docstring edits can shift line wrapping:

.. code-block:: shell

   poetry run black src/

.. note::

   pydocstyle's built-in conventions have a couple of pairs of mutually exclusive checks (for
   example ``D203``/``D211`` and ``D212``/``D213``) that cannot both be satisfied at once. The
   ``numpy`` convention used here already resolves these in the codebase's favor, so running with
   ``--convention=numpy`` locally should match what Codacy reports. If Codacy ever flags one side
   of such a pair that isn't covered by the ``numpy`` convention, that's a Codacy tool
   configuration issue rather than something to fix in the code — raise it with whoever
   administers the project's Codacy settings.
