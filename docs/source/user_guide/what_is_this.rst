============================
What is `sigmaepsilon.math`?
============================

.. include:: ..\global_refs.rst

The `sigmaepsilon.math`_ library is the mathematical department of the `SigmaEpsilon` project, 
a collection of Python libraries for computational mechanics and related disciplines.
It includes the tools that emerged during work on other parts of the `SigmaEpsilon` ecosystem.

Since most of these tools are general enough to be useful in other contexts, we decided to
extract them into a separate library.

The library is designed to be fast, as it relies on the vector math capabilities of `NumPy`_, while other computationally
sensitive calculations are JIT-compiled using `Numba`_. That being said, the primary goal of the library is not performance,
or to produce industry-grade solutions, but to provide a set of tools that are easy to use by students,
engineers and scientists. Sometimes a solution makes it into the library because it is a good teaching example, or because
it is a good starting point for further development.

The main areas of focus are linear algebra, operations research, and graph theory, which
we will discuss in more detail below.

Linear Algebra
==============

Linear algebra is a branch of mathematics that deals with vectors, vector spaces (also known as linear spaces), 
linear transformations, and systems of linear equations. It is fundamental to various areas of mathematics and is 
widely used in fields like physics, computer science, engineering, economics, and statistics.

* **Vectors and Tensors:** We provide NumPy-compliant classes like `Tensor` and `Vector` to handle various kinds of 
  tensorial quantities efficiently. These classes guarantee to maintain the property of objectivity of tensorial 
  quantities. These are not the usual Vectors and Tensors introduced in most Data Science and Machine Learning courses,
  but proper Vectors and Tensors in the sense of physics and engineering.

* **Reference Frames for Vectors and Tensors:** We provide a `ReferenceFrame` class for all kinds of frames, and 
  dedicated `RectangularFrame` and `CartesianFrame` classes as special cases, all NumPy compliant.

* **Sparse Data:** We provide a `JaggedArray` and a Numba-jittable `csr_matrix` to handle sparse and irregular data. 

Optimization
============

* **Linear Programming:** We provide a `LinearProgrammingProblem` class to define and solve almost  any kind of continuous 
  linear optimization problem. The implementation is suitable to handle small and medium-sized problems.
  The strenght of this class is not performance, but ease of use and flexibility. The solution is tightly integrated
  with `SymPy`_.

* **Nonlinear Programming:** We provide a `BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.
  The implementation is extendible which allows for further customization.
  
Graph Theory
============

Graph theory is a branch of mathematics that focuses on the study of graphs, which are structures made up of vertices 
(also called nodes) connected by edges. These structures are used to model relationships between pairs of objects, making 
graph theory an essential tool in a wide range of fields, from computer science and biology to social sciences and engineering.

The library includes a Graph class built on top of `NetworkX`_ that provides algorithms to calculate rooted level structures
and pseudo peripheral nodes of a graph. These algorithms are useful if you want to minimize the bandwidth of sparse matrices,
which is a common task in computational mechanics.

Miscellaneous
=============

.. note::

    This section is under construction.
