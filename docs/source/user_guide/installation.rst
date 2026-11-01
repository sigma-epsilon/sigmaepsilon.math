============
Installation
============

.. include:: ../global_refs.rst

.. note::

   If you are a developer, please refer to the :ref:`installation for developers <development>` section.

.. important::

   For the best performance, install the library together with Numba_:

   .. code-block:: shell

      pip install "sigmaepsilon.math[numba]"

   Numba is optional, but without it the performance-critical parts of the library run as plain
   Python, which can be orders of magnitude slower. See :ref:`performance` for the details.

You can install the library from ``PyPI`` via ``pip``. The recommended way is to install it
together with Numba_, using the ``numba`` extra:

.. code-block:: shell

   pip install "sigmaepsilon.math[numba]"

The quotes prevent shells like ``zsh`` from interpreting the square brackets. If you don't need
the performance-critical parts of the library (see :ref:`performance`), you can also install it
without Numba:

.. code-block:: shell

   pip install sigmaepsilon.math

Numba can be added later at any time with ``pip install numba``, there is no need to reinstall
the library.

To test your installation, run the following command in your Python environment:

.. code-block:: shell

   $ python
   Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import sigmaepsilon.math
   >>> sigmaepsilon.math.__version__
   'x.y.z'

Where ``x.y.z`` is the version number of ``sigmaepsilon.math``. To check whether the library
uses Numba, do this:

.. code-block:: python

   >>> from sigmaepsilon.core.jit import NUMBA_AVAILABLE
   >>> NUMBA_AVAILABLE
   True

.. _performance:

Performance and the role of Numba
---------------------------------

Numba_ is a just-in-time (JIT) compiler that translates numerical Python functions to optimized
machine code at runtime. The performance-critical parts of the library are written as such
functions, many of them running their loops in parallel on all available CPU cores. The machine
code is cached on disk, so the compilation only slows down the very first call of a function.

If Numba is not installed, the very same functions run as plain Python. The results are the
same, but the computations can be much slower, especially for large inputs. Some examples,
measured on a laptop, excluding the one-time compilation:

.. list-table::
   :header-rows: 1

   * - Operation
     - With Numba
     - Without Numba
   * - Inverting 100 000 3x3 matrices (``inv3x3_bulk``)
     - 1.6 ms
     - 750 ms
   * - Transforming 100 000 vectors to another frame (``show_vector``)
     - 2.3 ms
     - 95 ms

The parts of the library that benefit from Numba are

* vector and tensor algebra, and the related utilities in ``sigmaepsilon.math.linalg``
* sparse and jagged arrays (``JaggedArray``, ``csr_matrix``) and the solvers in ``linalg``
* array utilities, like ``unique2d`` and the histogram helpers
* least-squares and moving least-squares approximation in ``sigmaepsilon.math.approx``
* graph algorithms in ``sigmaepsilon.math.graph``

The following parts don't use Numba at all, so there is no reason to install it if you only need
these:

* the optimizers in ``sigmaepsilon.math.optimize`` (genetic algorithms, ant colony optimization,
  linear programming)
* symbolic and numeric functions in ``sigmaepsilon.math.function``
* numerical integration and k-nearest neighbour search

.. note::

   Instances of ``csr_matrix`` can be passed to your own Numba-jitted functions, which of course
   requires Numba to be installed.
