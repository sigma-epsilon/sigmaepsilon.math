====================================================================================
**SigmaEpsilon.Math** - A Python Library for Applied Mathematics in Natural Sciences
====================================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   User Guide <user_guide>
   API Reference <api>
   Development <development>

**Version**: |version|

**Useful links**:
:doc:`Installation <user_guide/installation>` |
:doc:`Getting Started <user_guide/getting_started>` |
`Issue Tracker <https://github.com/sigma-epsilon/sigmaepsilon.math/issues>`_ | 
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.math>`_

.. _sigmaepsilon.math: https://sigmaepsilon.math.readthedocs.io/en/latest/
.. _Awkward: https://awkward-array.org/doc/main/
.. _PyArrow: https://arrow.apache.org/docs/python/index.html
.. _NumPy: https://numpy.org/doc/stable/index.html
.. _Numba: https://numba.pydata.org/
.. _NetworkX: https://networkx.org/documentation/stable/index.html
.. _SciPy: https://scipy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _SymPy: https://www.sympy.org/en/index.html


The `sigmaepsilon.math`_ library is the mathematical department of the `SigmaEpsilon` project, a collection of 
Python libraries for computational mechanics and related disciplines.

The library includes tools that emerged during work on other parts of the `SigmaEpsilon` ecosystem.
Implementations are fast as they rely on the vector math capabilities of `NumPy`_, while other computationally 
sensitive calculations are JIT-compiled using `Numba`_.

Here and there we also use `NetworkX`_, `SciPy`_, `SymPy`_ and `scikit-learn`_.


Highlights
==========

* **Linear Algebra**

  * A mechanism that guarantees to maintain the property of objectivity of tensorial quantities.
  * A `ReferenceFrame` class for all kinds of frames, and dedicated `RectangularFrame` and `CartesianFrame` 
    classes as special cases, all NumPy compliant.
  * NumPy compliant classes like `Tensor` and `Vector` to handle various kinds of tensorial quantities efficiently.
  * A `JaggedArray` and a Numba-jittable `csr_matrix` to handle sparse data.

* **Operations Research**

  * Classes to define and solve linear and nonlinear optimization problems.
    
    * A `LinearProgrammingProblem` class to define and solve any kind of linear optimization problem.
    * A `BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.

* **Graph Theory**

  * Algorithms to calculate rooted level structures and pseudo peripheral nodes of a `networkx` graph, which are 
    useful if you want to minimize the bandwidth of sparse symmetrix matrices.

Contents
========

.. grid:: 2
    
    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        The getting started guide contains a basic introduction to the main concepts 
        and links to additional tutorials.

        +++

        .. button-ref:: getting_started
            :expand:
            :color: secondary
            :click-parent:

            Get me started

    .. grid-item-card::
        :img-top: ../source/_static/index-images/user_guide.svg

        User Guide
        ^^^^^^^^^^

        The user guide provides a more detailed walkthrough of the library, touching 
        the key features with useful background information and explanation.

        +++

        .. button-ref:: user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/api.svg

        API Reference
        ^^^^^^^^^^^^^

        The reference guide contains a detailed description of the functions,
        modules, and objects included in the library. The reference describes how the
        methods work and which parameters can be used. It assumes that you have an
        understanding of the key concepts.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: ../source/_static/index-images/contributor.svg

        Contributor's Guide
        ^^^^^^^^^^^^^^^^^^^

        Want to add to the codebase? The contributing guidelines will guide you through
        the process of improving the library.

        +++

        .. button-ref:: examples_gallery
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`