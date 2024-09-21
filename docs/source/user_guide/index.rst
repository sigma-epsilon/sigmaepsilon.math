.. _getting_started:

============
Introduction
============

What is `sigmaepsilon.math`?
============================

.. include:: ..\global_refs.rst

The `sigmaepsilon.math`_ library is the mathematical department of the `SigmaEpsilon` project, 
a collection of Python libraries for computational mechanics and related disciplines.
It includes the tools that emerged during work on other parts of the `SigmaEpsilon` ecosystem.

Since most of these tools are general enough to be useful in other contexts, we decided to
extract them into a separate library.

What can I use it for?
======================

The main areas of focus are linear algebra, optimization, approimation and graph theory. The most important
features are highlighted below and each topic is covered in detail in the :doc:`User Guide <../user_guide>`.

The library also provides some classes which are good base classes or are accessible inside Numba-jitted code,
paving the way for further development in various topics.

Highlights
----------

.. include:: ../highlights.rst

Is it fast?
===========

Yes, it is. The library is designed to be fast, as it relies on the vector math capabilities of `NumPy`_ and `SciPy`_, 
while other computationally sensitive calculations are JIT-compiled using `Numba`_. Thanks to `Numba`_, the 
implemented algorithms are able to bypass the limitations of Python's GIL and are parallelized on multiple cores, 
utilizing the full potential of what the hardware has to offer.

How do I install ``sigmaepsilon.math``?
=======================================

To install the library, follow the instructions in the :doc:`Installation Guide <installation>`.
