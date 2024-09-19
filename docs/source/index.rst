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
:doc:`Getting Started <getting_started>` |
`Issue Tracker <https://github.com/sigma-epsilon/sigmaepsilon.math/issues>`_ | 
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.math>`_

.. include:: global_refs.rst

The `sigmaepsilon.math`_ library is the mathematical department of the `SigmaEpsilon` project, a collection of 
Python libraries for computational mechanics and related disciplines. The library includes tools that emerged during work on other parts of the `SigmaEpsilon` ecosystem.
Implementations are fast as they rely on the vector math capabilities of `NumPy`_, while other computationally 
sensitive calculations are JIT-compiled using `Numba`_. Here and there we also use `NetworkX`_, `SciPy`_, `SymPy`_ and `scikit-learn`_.

Highlights
==========

* :ref:`Linear Algebra <user_guide_linalg>`

  * A :class:`~sigmaepsilon.math.linalg.ReferenceFrame` class for all kinds of frames, and dedicated :class:`~sigmaepsilon.math.linalg.RectangularFrame` and :class:`~sigmaepsilon.math.linalg.CartesianFrame` 
    classes as special cases, all NumPy compliant.
  * NumPy compliant classes like :class:`~sigmaepsilon.math.linalg.Tensor` and :class:`~sigmaepsilon.math.linalg.Vector` to handle various kinds of tensorial 
    quantities efficiently with a built-in mechanism that guarantees to maintain the property of objectivity.
  * A :class:`~sigmaepsilon.math.linalg.sparse.JaggedArray` and a Numba-jittable :class:`~sigmaepsilon.math.linalg.sparse.csr_matrix` to handle sparse data.

* :ref:`Optimization <user_guide_optimization>`

  * Classes to define and solve linear and nonlinear optimization problems.
    
    * A :class:`~sigmaepsilon.math.optimize.LinearProgrammingProblem` class to define and solve continuous, integer or mixed-integer linear optimization problems.
    * A :class:`~sigmaepsilon.math.optimize.bga.BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.

* :ref:`Appriximation <user_guide_approximation>`

  * Several methods and classes to approximate functions and data, including a :class:`~sigmaepsilon.math.approx.mls.MLSApproximator` for multilinear regression
    using the moving least squares method.

* :ref:`Graph Theory <user_guide_graph>`

  * Algorithms to calculate rooted level structures and pseudo peripheral nodes of a graph, and a :class:`~sigmaepsilon.math.graph.graph.Graph` class
    that extends ``networkx.Graph``.

Contents
========

.. grid:: 2
    
    .. grid-item-card::
        :img-top: ../source/_static/index-images/getting_started.svg

        Getting Started
        ^^^^^^^^^^^^^^^

        The getting started guide is your entry point. It helps you to set up
        a development environment and make the first steps with the library.

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

        The user guide provides a detailed walkthrough of the library, touching 
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
        modules, and objects included in the library. It describes how the
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

        .. button-ref:: development
            :expand:
            :color: secondary
            :click-parent:

            To the contributor's guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`