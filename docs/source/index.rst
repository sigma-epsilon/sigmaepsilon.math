====================================================================================
**SigmaEpsilon.Math** - A Python Library for Applied Mathematics in Natural Sciences
====================================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   User Guide <user_guide>
   API Reference <api>
   Development <development>

**Version**: |version|

**Useful links**:
:doc:`Installation <user_guide/installation>` |
:ref:`Getting Started <getting_started>` |
`Issue Tracker <https://github.com/sigma-epsilon/sigmaepsilon.math/issues>`_ | 
`Source Repository <https://github.com/sigma-epsilon/sigmaepsilon.math>`_

.. include:: global_refs.rst

The `sigmaepsilon.math`_ library is the mathematical department of the `SigmaEpsilon` project, a collection of 
Python libraries for computational mechanics and other disciplines. The library includes tools that emerged during 
work on other parts of the `SigmaEpsilon` ecosystem, but are general enough to be used in other projects.

Implementations are fast as they rely on the vector math capabilities of `NumPy`_, while other computationally 
sensitive calculations are JIT-compiled using `Numba`_. Here and there we also use `NetworkX`_, `SciPy`_, `SymPy`_ and `scikit-learn`_.

.. _highlights:

Highlights
==========

.. include:: highlights.rst

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

        .. button-ref:: user_guide/index
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


Bibliography, indices and tables
================================

* :doc:`Bibliography <bibliography>`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`