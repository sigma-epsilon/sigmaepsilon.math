==============
Linear Algebra
==============

Linear algebra is a branch of mathematics that deals with vector spaces and linear mappings 
between these spaces. It is a fundamental tool in many areas of science and engineering, including 
computer graphics, machine learning, and physics. Linear algebra provides a framework for solving 
systems of linear equations, analyzing geometric transformations, and understanding the properties 
of vectors and matrices. It is such an important subject, that we decided to cover it in more depth
than what would be required to use the library effectively.

Another motivation behind this is the frequent oversimplification of core mathematical 
concepts. For example, you might see statements like

- "A vector is an array of numbers that encapsulates magnitude and direction." 
  (from an article -sadly- written by someone with a PhD in machine learning)
- A vector is a one dimensional matrix :cite:p:`Ananthaswamy2024whymachineslearn`
- Functions are vectors in an infinite dimensional space :cite:p:`Ananthaswamy2024whymachineslearn`
- "A Tensor is a N-dimensional Matrix." (from a blog post on a popular educational website)
- "A vector is a 1-dimensional tensor." (from a blog post on a popular educational website)
- "A matrix is a two-dimensional list of vectors." (from an article on Medium)
- "Tensors are multi-dimensional arrays with a uniform type." (from the learning materials of a popular deep-learning library)

When misconceptions like this become widespread, they can cause confusion in at least two 
ways:

1) Communication Breakdown

   If one person uses “vector” to mean “abstract element of a vector space,” while another believes a vector 
   is always “an array of numbers,” they will talk past each other. This often leads to misunderstandings 
   about what operations are valid or how concepts generalize (for example, to infinite‐dimensional spaces or
   function spaces).

2) Limiting Perspective and Application

   Reducing “vector” to “array” can make people miss the breadth and power of the underlying mathematical 
   structure. Abstract vector spaces enable us to reason about things like polynomial functions, continuous 
   function spaces, or even more exotic constructions—all under the same formalism. When people conflate the 
   representation (arrays of coordinates) with the concept (abstract vectors), they miss out on these deeper 
   connections and may draw incorrect conclusions.

In short, false information or oversimplifications hamper effective teaching, learning, and collaboration. 
They introduce inconsistencies and misunderstandings that have to be untangled later, often wasting time and 
stunting the very progress that clear mathematical language is designed to foster.

By the end of this guide, you will understand what a vector, a matrix, an array or a tensor truly is.

**It's important to clarify that this guide is not intended to criticize others. Linear algebra is a challenging subject, 
and its terminology has evolved over time, making it understandable that people may struggle with it. The goal here is 
simply to provide clarity and ensure that we share a common understanding.**

Learning objectives
===================

By the end of this setion of the theory guide, you should be able to:

- Understand the basic concepts of linear algebra.
- Understand vectors, vector spaces, linear maps, matrices, norms, the essential building blocks of 
  linear algebra.
- To be able to point out the incorrectness of the statements above.

Introduction
============

Why Linear Algebra?
-------------------

The power of modern (or abstract) algebra lies in its ability to extend the concept of **distance** beyond geometry.
Why is that a big deal? Because it lets us reason about things that aren’t inherently geometric. We can define the 
distance between two words, two songs, or two movies. We can measure the distance between people—not in the physical 
sense, but in terms of their preferences, behaviors, or opinions. And once we have a notion of distance, we can 
**compare** things. If we can compare things, we can **order** them. If we can order things, we can **make decisions**. 
If we can make decisions, we can **optimize**. And if we can optimize, we can **make things better**.
That’s why linear algebra is so powerful—it’s not just about numbers and equations. It’s about structure, reasoning, 
and ultimately, making better choices.

**Linear algebra enables us to make things better.**

A little bit of history
-----------------------

Before we delve into the fundamental concepts of linear algebra, let’s briefly explore its historical 
development. This background is important because terms like “vector” and “tensor” can carry slightly 
different meanings in physics, mathematics, machine learning or other subject domains. Understanding where these ideas 
originated can help clarify why such differences exist.

Modern mathematics and science began with Isaac Newton and Gottfried Wilhelm Leibniz, who independently
invented calculus in the late 17th century. Calculus provided a powerful framework for understanding
change and motion, but it was limited in scope. In the 18th century, mathematicians like Leonhard Euler
and Joseph-Louis Lagrange began to develop a more general theory of functions and equations. This work
culminated in the 19th century with the development of linear algebra, which provided a unified framework
for studying systems of linear equations and geometric transformations. The term “linear algebra” was first
used by the mathematician James Joseph Sylvester in the 1850s. Since then, linear algebra has become a
fundamental tool in many areas of mathematics, science, and engineering.

It was probably Newton who first introduced the concept of a vector in the 17th century. He used vectors to describe
forces and velocities in his laws of motion, but he didn't use the term "vector". The term "vector" was first
used in the 19th century by the Irish mathematician William Rowan Hamilton. Hamilton introduced the concept
of a quaternion, which is a more general form of a vector that can represent rotations in three-dimensional
space. The term “vector” comes from the Latin word for “carrier” or “bearer,” reflecting the idea that a vector 
carries both magnitude and direction. Note that when we describe Newton and Hamilton as mathematicians, we are not
entirely accurate. They were natural phylosophers, which is what we call scientists today and the term "mathematician"
was not as well defined as it is today. **It is safe to say that vectors were first used to explain motion and forces
in physics, and were only later generalized to represent abstract entities.**

Vectors and Vector Spaces
=========================

.. admonition:: **Definition** (list, length, coordinate)
   :class: definition-box

   Suppose $n$ is a nonnegative integer. A list of length n is an ordered collection of n elements,
   (which may be real numbers, other lists, or more abstract entities) separated by commas and 
   surrounded by parentheses. A list of length $n$ looks like this:
   
   .. math::
   
      (x_1, \ldots, x_n).

   Two lists are equal if and only if they have the same length and the same elements in the same order. In the list
   above, $x_j$ is the $j^{th}$ **coordinate** (or component) of the list.

This definition aligns well with the definition of a list in Python. A list differs from a set in terms of order 
and repetition of elements. A set is an unordered collection of unique elements, while a list is an ordered collection 
of elements that may be repeated. In other words, order and repetition are irrelevant in sets, but they are relevant in lists.

Another thing to note is that many mathematicians call a list of length $n$ an $n$-tuple. The terms "list" and "tuple" are
often used interchangeably in mathematics, but in Python, a tuple is a specific data structure that is immutable 
(i.e., it cannot be changed). Tuples and lists in Python are both lists in the mathematical sense, but they are not vectors
as we will see shortly.

.. admonition:: **Definition** (field, :math:`\mathbf{F}^{n}`)
   :class: definition-box

   A **field** is a set of numbers that can be added, subtracted, multiplied, and divided. The most common fields 
   are the real numbers :math:`\mathbf{R}` and the complex numbers :math:`\mathbf{C}`. 
   
   The set of all lists of length $n$ with elements from a field :math:`\mathbf{F}`
   is denoted :math:`\mathbf{F}^{n}`.

.. admonition:: **Definition** (vector space, vector, point)
   :class: definition-box

   A vector space is a set :math:`V` along with an addition on :math:`V` and
   a scalar multiplication on :math:`V` such that the following properties hold:

   **commutativity**  
   :math:`u + v = v + u` for all :math:`u, v \in V;`

   **associativity**  
   :math:`(u + v) + w = u + (v + w) \quad\text{and}\quad (ab)v = a(bv)`
   for all :math:`u, v, w \in V` and all :math:`a, b \in \mathbf{F};`

   **additive identity**  
   there exists an element :math:`0 \in V` such that :math:`v + 0 = v`
   for all :math:`v \in V;`

   **additive inverse**  
   for every :math:`v \in V`, there exists :math:`w \in V`
   such that :math:`v + w = 0;`

   **multiplicative identity**  
   :math:`1v = v` for all :math:`v \in V;`

   **distributive properties**  
   :math:`a(u + v) = au + av \quad\text{and}\quad (a + b)v = av + bv`
   for all :math:`a, b \in \mathbf{F}` and all :math:`u, v \in V.`

   Elements of a vector space are called **vectors** or **points**.

The important thing here is that in general, a vector space is an abstract entity whose elements might be lists,
functions, or weird objects, whatever that satisfies the properties above. In strict mathematical terms, **only vectors in a 
finite-dimensional vector space can be unambiguously represented by a finite 1d array of numbers**, and only after choosing a 
basis for the space. In some cases, we can assume that the so-called standard basis is choosen and kind of lurking in the background 
without explicitly mentioning it. Nevertheless, it is important to keep in mind that vectors are abstract entities that can be 
represented in many ways an while a 1d array is a perfectly valid (and common) *coordinate representation* for finite-dimensional 
vectors over a field, saying "vectors are 1d arrays" is too narrow.

.. admonition:: **Notation** :math:`\mathbf{F}^S`
   :class: definition-box

   If :math:`S` is a set, then :math:`\mathbf{F}^S` denotes the set of all functions
   from :math:`S` to :math:`\mathbf{F}`.

You can verify that :math:`\mathbf{F}^S` is a vector space.
As an example, if :math:`S` is the interval :math:`[0, 1]` and :math:`F = R`, then :math:`\mathbf{R}^{[0, 1]}`
is the vector space of real-valued functions in the interval :math:`[0, 1]`.
Moreover, the vector spaces :math:`\mathbf{F}^n` and :math:`\mathbf{F}^{\{1,2,\ldots\}}` are special cases of this
definition, where :math:`\mathbf{F}^n` is the vector space of all functions from the set :math:`\{1, \ldots, n\}` to
:math:`\mathbf{F}` and :math:`\mathbf{F}^{\{1,2,\ldots\}}` is the vector space of all functions from the set of natural numbers
:math:`\mathbb{N}` to :math:`\mathbf{F}`.

.. admonition:: **Definition** (span, spans)
   :class: definition-box

   The set of all linear combinations of a list of vectors $v_1, \\ldots, v_m$ in $V$ is called the **span** of $v_1, 
   \\ldots, v_m$, denoted $span(v_1, \\ldots, v_m)$. In other words,

   :math:`span(v_1, \ldots, v_m) = \{ a_1 v_1 + \ldots + a_m v_m : a_1, \ldots, a_m \in \mathbf{F} \}`.

   The span of the empty list $()$ is defined to be $\{0\}$. If $span(v_1, \\ldots, v_m)$ equals $V$, we say that $v_1, 
   \\ldots, v_m$ **spans** $V$.

Now we can make one of the key definitions in linear algebra. Remember, that by definition, every list has finite length.

.. admonition:: **Definition** (finite-dimensional vector space)
   :class: definition-box

   A vector space $V$ is said to be **finite-dimensional** if some list of vectors in it spans the space. If no finite 
   list of vectors spans $V$, then $V$ is said to be **infinite-dimensional**.

Another fundamental concept in linear algebra is the notion of a basis, but before we define it, we need to introduce the 
concept of linear independence.

.. admonition:: **Definition** (linear independence)
   :class: definition-box

   A list of vectors $v_1, \\ldots, v_m$ in a vector space $V$ is said to be **linearly independent** if the only way to 
   write the zero vector as a linear combination of $v_1, \\ldots, v_m$ is to take all coefficients to be zero. In other words, 
   the list $v_1, \\ldots, v_m$ is linearly independent if the equation

   :math:`a_1 v_1 + \ldots + a_m v_m = 0`

   implies that $a_1 = \\ldots = a_m = 0$.

Now we bring together the concepts of linear independence and spanning to define a basis.

.. admonition:: **Definition** (basis)
   :class: definition-box

   A **basis** for a vector space $V$ is a linearly independent list of vectors in $V$ that spans $V$. 
   
In other words, a list of vectors $v_1, \\ldots, v_m$ is a basis for $V$ if every vector in $V$ can be written 
uniquely as a linear combination of $v_1, \\ldots, v_m$.

.. admonition:: **Definition** (standard basis)
   :class: definition-box

   The **standard basis** for a vector space $V$ is the list of vectors :math:`e_1, \ldots, e_n` 
   where the :math:`j^{\text{th}}` vector has a 1 in the :math:`j^{\text{th}}` slot and 0's elsewhere.

A vector space can have many different bases, but the standard basis is unique and a particularly useful one.

We have defined finite dimensional vector spaces, but we haven't defined what the dimension means.

.. admonition:: **Definition** (dimension)
   :class: definition-box

   The **dimension** of a finite-dimensional vector space $V$ is the number of vectors in a basis for $V$.
   The dimension of the vecctor space $V$ is denoted by $dim V$ or $dim(V)$.

Any two bases for a finite-dimensional vector space have the same number of vectors. Because the vectors of the basis 
must be linearly independent, the dimension of a vector space is well-defined.

.. admonition:: **Definition** (supspace)
   :class: definition-box

   A subset :math:`U` of :math:`V` is a supspace of :math:`V` if :math:`U` is also a vector space.

The next result helps to quickly tell if a subset of a vector space is a subspace.

.. admonition:: Conditions for a subspace
   :class: definition-box

   The easiest way to check if a subset :math:`U` of :math:`V` is a subspace of :math:`V` is to check the following 
   three conditions:

   **additive identity**

   .. math::

      0 \in U;

   **closure under addition**

   .. math::

      u, w \in U \quad \text{implies} \quad u + w \in U;

   **closure under scalar multiplication**

   .. math::

      u \in U, a \in \mathbf{F} \quad \text{implies} \quad a\,u \in U.
   

Polynomials
===========

You might wonder why we are talking about polynomials in a linear algebra guide. The reason is that polynomials are
vectors. This means that we can use the tools of linear algebra to study polynomials.

.. admonition:: **Definition** (polynomial, :math:`\mathcal{P}(\mathbf{F})`)
   :class: definition-box

   A function :math:`p:\mathbf{F} \to \mathbf{F}` is called a **polynomial** with
   coefficients in :math:`\mathbf{F}` if there exist :math:`a_0, \dots, a_m \in \mathbf{F}`
   such that

   .. math::

      p(x) = a_0 + a_1 x + \cdots + a_m x^m

   for all :math:`x \in \mathbf{F}`.

   :math:`\mathcal{P}(\mathbf{F})` is the set of all polynomials with coefficients in :math:`\mathbf{F}`.

With the usual operations of addition and scalar multiplication, the set :math:`\mathcal{P}(\mathbf{F})` 
forms a vector space over :math:`\mathbf{F}`. Note that :math:`\mathcal{P}(\mathbf{F})` is an infinite
dimensional vector space, because it has no finite basis. Note that the fact that :math:`\mathcal{P}(\mathbf{F})` is
a vector space makes polynomials vectors. This is important because it allows us to use the tools of linear algebra 
to study polynomials.

.. admonition:: **Definition** (degree of a polynomial, :math:`\text{deg} \, p`)
   :class: definition-box

   The **degree** of a polynomial :math:`p \in \mathcal{P}(\mathbf{F})` is the largest integer :math:`m` such that
   :math:`a_m \neq 0`. If :math:`p` has degree :math:`m`, we write :math:`\text{deg} \, p = m`.

   The degree of the zero polynomial is defined to be :math:`-\infty`.

For practical applications, we are often interested in polynomials of a finite degree.

.. admonition:: **Definition** (polynomial, :math:`\mathcal{P}_{m}(\mathbf{F})`)
   :class: definition-box

   For :math:`m` a nonnegative integer, :math:`\mathcal{P}_{m}(\mathbf{F})` denotes the set of all polynomials
   with coefficients in :math:`\mathbf{F}` and degree at most :math:`m`.


Don't get mislead by all this. Just because polynomials can be described as vectors, doesn't mean
that all functions can be. For instance, let's consider the set of all functions from real numbers
to integers:

.. math::

   \mathcal{S} = \{ f: \mathbf{R} \to \mathbf{R} : f(x) \in \mathbf{Z} \text{ for all } x \in \mathbf{R} \}.

It is up to the reader to verify, that this set is not closed under scalar multiplication (among other things), therefore
it is not a vector space.


Linear Maps
===========

The interesting and most important part of linear algebra is the study of linear maps between vector spaces.

.. admonition:: **Notation**
   :class: definition-box

   - $\\mathbf{F}$ denotes $\\mathbf{R}$ or $\\mathbf{C}$.
   - $V$ and $W$ denote vector spaces over $\\mathbf{F}$.

Now we are ready for one of the key definitions in linear algebra.

.. admonition:: **Definition** (linear map)
   :class: definition-box

   A linear map from :math:`V` to :math:`W` is a function :math:`T: V \to W`
   with the following properties:

   **additivity**  
   :math:`T(u + v) = T(u) + T(v)` for all :math:`u, v \in V;`

   **homogeneity**  
   :math:`T(\lambda v) = \lambda\,T(v)` for all :math:`\lambda \in \mathbf{F}` 
   and all :math:`v \in V.`

Note that for linear maps, we often use the notation $Tv$ as well as the more standard functional notation $T(v)$.
Also, be aware that the term "linear map" is synonymous with "linear transformation" and "linear operator".

.. admonition:: **Notation** :math:`\mathcal{L}(V, W)`
   :class: definition-box

   The set of all linear maps from :math:`V` to :math:`W` is denoted
   :math:`\mathcal{L}(V, W)`.

An example for a linear map is :math:`T \in \mathcal{L}(\mathcal{P}(\mathbf{R}), \mathbf{R})` defined by
:math:`T(p) = \int_0^1 p(x) dx`. 

We continue by defining some important properties of linear maps.

.. admonition:: **Definition** addition and scalar multiplication of linear maps
   :class: definition-box

   Suppose :math:`S, T \in \mathcal{L}(V, W)` and :math:`\lambda \in \mathbf{F}`.
   The **sum** :math:`S + T` and the **product** :math:`\lambda T` are the linear
   maps from :math:`V` to :math:`W` defined by

   .. math::

      (S + T)(v) = S v + T v 
      \quad\text{and}\quad
      (\lambda T)(v) = \lambda (T v)

   for all :math:`v \in V`.

.. admonition:: :math:`\mathcal{L}(V, W)` is a vector space
   :class: definition-box

   With the operations of addition and scalar multiplication defined above, the set :math:`\mathcal{L}(V, W)` of linear 
   maps from :math:`V` to :math:`W` forms a vector space.

This of course implies that linear maps are vectors themselves too. From the section about matrices of linear maps,
it will be obvious why the statement "a vector is a 1d array" is incorrect.

Null space and range
--------------------

.. admonition:: **Definition** (null space, null :math:`T`)
   :class: definition-box

   The **null space** of a linear map :math:`T \in \mathcal{L}(V, W)` is the set of all vectors :math:`v \in V` such that
   :math:`T v = 0`. The null space of :math:`T` is denoted :math:`\text{null} \, T`.

   Some mathematicians use the term **kernel** instead of null space.

.. admonition:: **Definition** (range, range :math:`T`)
   :class: definition-box

   The **range** of a linear map :math:`T \in \mathcal{L}(V, W)` is the set of all vectors :math:`w \in W` such that
   :math:`w = T v` for some :math:`v \in V`. The range of :math:`T` is denoted :math:`\text{range} \, T`.

   Some mathematicians use the term **image** instead of range.

.. admonition:: The null space and the range are subspaces
   :class: definition-box

   The null space and the range of a linear map are subspaces of the vector spaces :math:`V` and :math:`W`, respectively.

Injectivity and surjectivity
----------------------------

Injectivity and surjectivity are important properties of linear maps, because they tell us about the
structure of the map and the relationship between the vector spaces involved. They are heaviliy used in the study of
linear maps, and they are also important in the study of linear equations and systems of equations.

.. admonition:: **Definition** injective
   :class: definition-box

   A linear map :math:`T \in \mathcal{L}(V, W)` is said to be **injective** if :math:`T u = T v` implies :math:`u = v` 
   for all :math:`u, v \in V`. 

   In other words, :math:`T` is injective if it maps distinct vectors in :math:`V` to distinct vectors in :math:`W`.

   Also, it can be shown that a linear map being injective is equivalent to :math:`\text{null} \, T = \{0\}`. 
   In other words, :math:`T` is injective if and only if the only vector in the null space of :math:`T` is the zero vector.

   Many mathematicians use the term **one-to-one** instead of injective.

.. admonition:: **Definition** surjective
   :class: definition-box

   A linear map :math:`T \in \mathcal{L}(V, W)` is said to be **surjective** if :math:`T v = w` for all :math:`w \in W`
   and some :math:`v \in V`.

   In other words, :math:`T` is surjective if it maps all vectors in :math:`V` to all vectors in :math:`W`.

   It follows that a linear map being surjective is equivalent to :math:`\text{range} \, T = W`.

Fundamental theorem of linear maps
----------------------------------

The following result is so important, that it gets a dramatic name.

.. admonition:: Fundamental theorem of linear maps
   :class: definition-box

   Suppose :math:`V` is finite dimensional and :math:`T \in \mathcal{L}(V, W)`. Then :math:`\text{range} T` is finite
   dimensional and the following equation holds:
   
   .. math::

      \text{dim} \, V = \text{dim} \, \text{null} \, T + \text{dim} \, \text{range} \, T.

It might be challenging to see the importance of this theorem at first, but it is a powerful result that
was used to derive some very important statements. The following two results can both be proven very easily 
using the fundamental theorem of linear maps.

.. admonition:: A map to a smaller dimensional space is not injective
   :class: definition-box

   Suppose :math:`T \in \mathcal{L}(V, W)` and :math:`\text{dim} \, V > \text{dim} \, W`. Then :math:`T` is not injective.

   In other words, if the dimension of the domain of a linear map is greater than the dimension of the codomain, 
   then the map cannot be injective.

One can use the fundamental theorem of linear maps to show that the null space of a linear map is not :math:`\{0\}` if the 
dimension of the domain is greater than the dimension of the codomain. It follows directly, that the map cannot be injective.

This has an important implication for the study of systems of linear equations.

.. admonition:: Homogeneous systems of linear equations
   :class: definition-box

   A homogeneous system of linear equations with more variables than equations has nonzero solutions.

Yet another important result follows from the fundamental theorem of linear maps.

.. admonition:: A map to a larger dimensional space is not surjective
   :class: definition-box

   Suppose :math:`T \in \mathcal{L}(V, W)` and :math:`\text{dim} \, V < \text{dim} \, W`. Then :math:`T` is not surjective.

   In other words, if the dimension of the domain of a linear map is less than the dimension of the codomain, 
   then the map cannot be surjective.

Again, we can use this result to make a statement about systems of linear equations.

.. admonition:: Inhomogeneous systems of linear equations
   :class: definition-box

   An inhomogeneous system of linear equations with more equations than variables has no solution for some choice
   of the constant terms.

These results truly show the significance of the fundamental theorem of linear maps. These results are merely consequences of the
fundamental theorem applied to specific cases of linear maps. Needless to say, the implications of these results are far-reaching
and have a profound impact on the study of many engineering and scientific disciplines.

Matrices
--------

Matrices are **a convenient way to represent** linear maps between finite-dimensional vector spaces.
In fact, we know that if :math:`v_1, \ldots, v_m` is a basis of :math:`V` and :math:`T: V \to W` is a 
linear map, then the values :math:`T(v_1), \ldots, T(v_m)` determine the values of :math:`T` on
arbitrary vectors in :math:`V`. As we will soon see, **matrices are used as an efficient method of recording
the values** of the :math:`T(v_j)`'s in terms of a basis of :math:`W`.

.. admonition:: **Definition** (matrix, :math:`A_{j,k}`)
   :class: definition-box

   Let :math:`m` and :math:`n` be positive integers. An :math:`m`-by-:math:`n`
   **matrix** :math:`A` is a rectangular array of elements of :math:`\mathbf{F}`
   with :math:`m` rows and :math:`n` columns:

   .. math::

      A =
      \begin{pmatrix}
      A_{1,1} & \cdots & A_{1,n} \\
      \vdots & \ddots & \vdots \\
      A_{m,1} & \cdots & A_{m,n}
      \end{pmatrix}.

   The notation :math:`A_{j,k}` denotes the entry in row :math:`j`, column :math:`k`
   of :math:`A`. In other words, the first index refers to the row number and the
   second index refers to the column number.

We've come to another key definition.

.. admonition:: **Definition** (matrix of a linear map, :math:`\mathcal{M}(T)`)
   :class: definition-box

   Suppose :math:`T \in \mathcal{L}(V, W)` and :math:`v_1, \dots, v_n` is a basis of
   :math:`V` and :math:`w_1, \dots, w_m` is a basis of :math:`W`. The **matrix** of
   :math:`T` with respect to these bases is the :math:`m`-by-:math:`n` matrix
   :math:`\mathcal{M}(T)` whose entries :math:`A_{j,k}` are defined by

   .. math::

      T v_k = A_{1,k} w_1 + \cdots + A_{m,k} w_m.

   If the bases are not clear from the context, then the notation

   .. math::

      \mathcal{M} \big( T, (v_1, \dots, v_n), (w_1, \dots, w_m) \big)

   is used.

In other words, the matrix of a linear map tells how the basis vectors of :math:`V` can be composed as linear combinations
of the basis vectors of :math:`W`.

The important thing to note here that **the matrix of a linear map depends on the linear map, and the bases of the 
vector spaces involved**. The matrix of a linear map is not an intrinsic property of the linear map itself, but rather
a representation of the linear map with respect to a particular choice of bases. **This is the primary reason why the
statement "a tensor is a multi-dimensional array" is incorrect.** A tensor is a multi-linear map (as it will be defined
later), and a multi-linear map is not a multi-dimensional array. A multi-linear map can be represented as a multi-dimensional
array, but the array depends on the choice of bases.

It might me easier to remember the structure of :math:`\mathcal{M}(T)` by noting that the :math:`k^{\text{th}}` column 
of :math:`\mathcal{M}(T)` consists of the scalars needed to write  :math:`T v_k` as a linear combination of 
:math:`(w_1, \dots, w_m)`:

.. math::

   T v_k = \sum_{j=1}^{m} A_{j,k} w_j.

Another way to think about the matrix of a linear map is by using the definition of a standard basis.
If you think of elements of :math:`\mathbf{F}^m` as columns of $m$ numbers, then you can think of the 
:math:`k^{\text{th}}` column of :math:`\mathcal{M}(T)` as the column of numbers that you get when you apply
:math:`T` to the :math:`k^{\text{th}}` standard basis vector of :math:`\mathbf{F}^n`.

.. admonition:: **Notation** :math:`\mathbf{F}^{m,n}`
   :class: definition-box

   For $m$ and $n$ positive integers, the set of all $m$-by-$n$ matrices with entries from :math:`\mathbf{F}` 
   is denoted :math:`\mathbf{F}^{m,n}`.

Now comes the fun part.

.. admonition:: :math:`\mathbf{F}^{m,n}` is a vector space
   :class: definition-box

   With the usual definitions of matrix addition and scalar multiplication, the set :math:`\mathbf{F}^{m,n}` of all
   :math:`m`-by-:math:`n` matrices forms a vector space over :math:`\mathbf{F}`.

In other words, matrices are vectors. It only follows that rows and columns of matrices are also vectors, but this doesn't
mean that matrices are vectors of vectors. If we consider the definition of a list, we can say that a particular $m$-by-$n$ 
matrix is an m-tuple of vectors of length $n$, whose first element is the first row of the matrix, and so on. Threfore, to
say that a matrix is a list of vectors, is a valid interpretation. Note however, that the implication in the other way is not
generally true, because not all lists of vectors satisfy the structural properties of a matrix. For instance, the following list
(of length 2) of vectors is not a matrix because it can't be represented as a rectangular array of numbers:

.. math::

   \left( \left( 1, 2 \right), \left( 4, 5, 6 \right) \right).


.. admonition:: **Definition** (matrix of a vector, :math:`\mathcal{M}(v)`)
   :class: definition-box

   Suppose :math:`v \in V` and :math:`v_1, \dots, v_n` is a basis of
   :math:`V`. The **matrix** of :math:`v` with respect to this basis is the
   :math:`n`-by-:math:`1` matrix

   .. math::

      \mathcal{M}(v) =
      \begin{pmatrix}
      c_1 \\
      \vdots \\
      c_n
      \end{pmatrix},

   where :math:`c_1, \dots, c_n` are scalars such that

   .. math::

      v = c_1 v_1 + \cdots + c_n v_n.

Matrix multiplication
^^^^^^^^^^^^^^^^^^^^^

Upper-triangular and diagonal matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A central goal of linear algenra is to show that given an operator :math:`T \in \mathcal{L}(V)`,
there exists a basis of :math:`V` such that the matrix of :math:`T` with respect to this basis 
has a reasonably simple matrix. The term 'reasonably simple' is a bit vague, but it usually means 
that the matrix is upper-triangular or diagonal. This is big because in practice, when solving linear
systems of equations, the coefficient matrix can be enormously large, at it would be very useful if
we could somehow transform it using a suitable base, so that the matrix is upper-triangular or diagonal.
Then, we would solve the system in that basis, and transform the solution back to the original basis.
For this reason, it is quite important to know when this simplification is possible. For this, we
need to cover eigenvalues and eigenvectors first, which are the key to this simplification. Here, we
only introduce the concepts of upper-triangular and diagonal matrices, and we will cover eigenvalues
and eigenvectors in a later section.

Invertibility
-------------

.. admonition:: **Definition** (invertible, inverse)
   :class: definition-box

   - A linear map :math:`T \in \mathcal{L}(V, W)` is said to be **invertible** if there exists a linear map
     :math:`S \in \mathcal{L}(W, V)` such that :math:`ST` equals the identity map on :math:`V` and :math:`TS` 
     equals the identity map on :math:`W`.

   - A linear map :math:`S \in \mathcal{L}(W, V)` satisfying :math:`ST = I` and :math:`TS = I` is called
     an inverse of :math:`T` (note that the first :math:`I` is the identity map on :math:`V` and the second
     :math:`I` is the identity map on :math:`W`). 
   
It can be proven fairly easily that the inverse of a linear map is unique, if it exists, and so the following
notation makes sense.

.. admonition:: **Notation** :math:`T^{-1}`
   :class: definition-box

   If :math:`T` is invertible, then its inverse is denoted by :math:`T^{-1}`.

If you are not familiar with how this works, note that introducing the notation for the inverse consisted of two steps. First we
defined the meaning, and then we showed that the meaning is unique. Only after that we introduced the notation.

The following result characterizes invertible linear maps.

.. admonition:: Invertibility is equivalent to injectivity and surjectivity
   :class: definition-box

   A linear map is invertible if and only if it is both injective and surjective.
   
We only mention here, that there is such a thing as a pseudo-inverse of a linear map, which is not
injective or surjective. The pseudo-inverse is a generalization of the inverse of a linear map, and it is used in
many applications. However, pseudo-inverses and generalized inverses are beyond the scope of this guide.

Operators and functionals
-------------------------

There are some linear maps that are so important, that they get a special name and notation.

.. admonition:: **Definition** (operator, :math:`\mathcal{L}(V)`)
   :class: definition-box

   * A linear map from a vector space to itself is called an **operator**.

   * The set of all operators on a vector space :math:`V` is denoted
     :math:`\mathcal{L}(V)`. In other words, :math:`\mathcal{L}(V) = \mathcal{L}(V, V)`.

One of the main reasons that a richer theory exists for operators than for general linear maps is that
operators can be raised to powers, which is key for applying polynomials to operators.

Operators are important, because a physical system can very often be described as a vector space, and
the evolution of the system can be described as an operator acting on the vector space. Moreover, in
computational mechanics, we are interested in discrete solutions, which means that we are working with
finite-dimensional vector spaces. Because of this, the following result is significant.

.. admonition:: Injectivity is equivalent to surjectivity in finite-dimensional operators
   :class: definition-box

   Suppose :math:`V` is finite-dimensional and :math:`T \in \mathcal{L}(V)`. Then the following are
   equivalent:

   - :math:`T` is injective;
   - :math:`T` is surjective;
   - :math:`T` is invertible.

Another special type of linear map is a linear functional.

.. admonition:: **Definition** linear functional
   :class: definition-box

   A linear map from a vector space to its field is called a **linear functional**.
   In other words, a linear functional is an element of :math:`\mathcal{L}(V, \mathbf{F})`.
   
Here are a few examples:

- Define :math:`\varphi : \mathbf{R}^3 \to \mathbf{R}` by :math:`\varphi(x, y, z) = 4x - 5y + 2z`. 
  Then :math:`\varphi` is a linear functional on :math:`\mathbf{R}^3`.

- Define :math:`\varphi : \mathcal{P}(\mathbf{R}) \to \mathbf{R}` by :math:`\varphi(p) = \int_0^1 p(x)\,dx`.
  Then :math:`\varphi` is a linear functional on :math:`\mathcal{P}(\mathbf{R})`.

Duality
-------

Duality is an advanced topic, but it doesn't hurt if you know a little about it.

.. admonition:: **Definition** (dual space, :math:`V'`)
   :class: definition-box

   The dual space of  :math:`V`, denoted :math:`V'`, is the vector space of all linear functionals
   on :math:`V`. In other words, :math:`V' = \mathcal{L}(V, \mathbf{F})`.

.. admonition:: **Definition** dual basis
   :class: definition-box

   If :math:`v_1, \ldots, v_n` is a basis of :math:`V`, then the **dual basis** of
   of :math:`v_1, \ldots, v_n` is the list of elements :math:`\varphi_1, \ldots, \varphi_n`
   of elements of :math:`V'`, where each :math:`\varphi_j` is the linear functional on :math:`V`
   such that

   .. math::
      
      \varphi_j(v_k) = \begin{cases}
      1 & \text{if } j = k, \\
      0 & \text{if } j \neq k.
      \end{cases}

.. admonition:: Dual basis is a basis of the dual space
   :class: definition-box

   Suppose :math:`V` is finite-dimensional. Then the dual basis of a basis of :math:`V` is a basis
   of the dual space :math:`V'`.

.. admonition:: **Definition** (dual map, :math:`T'`)
   :class: definition-box

   If :math:`T \in \mathcal{L}(V, W)`, then the **dual map** of :math:`T` is the linear map
   :math:`T' \in \mathcal{L}(W', V')` defined by
   
   .. math::

      T' (\varphi) = \varphi \circ T \quad \text{for} \varphi \in W'.

It might not be obvious at this point why, but duality is huge in both the theory
and applications of linear algebra. To give you a few examples:

- In machine learning we exploit duality theory to find
  separating hyperplanes by transforming the primal problem into a dual problem, which is easier to solve
  computationally and gives access to kernel methods. 
- In physics, we use duality to express conservation laws in a coordinate invariant way. 
- In computational mechanics, we use duality to express the weak form of the
  governing equations in a coordinate invariant way. 

In summary, duality is a powerful tool that often allows us to express complex problems in a more manageable form. 
It is also a powerful tool for proving theorems and deriving results in linear algebra.

The matrix of the dual of a linear map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. admonition:: **Definition** (transpose, :math:`A^t`)
   :class: definition-box

   The **transpose** of a matrix :math:`A` is the matrix :math:`A^t` obtained from
   :math:`A` by interchanging its rows and columns. More specifically, if :math:`A` is an
   :math:`m\text{-by-}n` matrix, then :math:`A^t` is an :math:`n\text{-by-}m` matrix
   whose entries are given by the equation

   .. math::

      A^t_{k,j} = A_{j,k}.

.. admonition:: The matrix of :math:`T'` is the transpose of the matrix of :math:`T`
   :class: definition-box

   Suppose :math:`T \in \mathcal{L}(V, W)`. Then :math:`\mathcal{M}(T') = \mathcal{M}(T)^t`.

Eigenvalues and eigenvectors
============================


Diagonalization
---------------


Gram-Schmidt process
--------------------


Inner product spaces
====================

Riesz representation theorem
----------------------------


Orthogonal complement and minimization problems
-----------------------------------------------


Minimization problems
^^^^^^^^^^^^^^^^^^^^^


Operators on inner product spaces
=================================

The deepest results related to inner product spaces deal with operators on inner product spaces.


The spectral theorem
--------------------

The complex spectral theorem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The real spectral theorem
^^^^^^^^^^^^^^^^^^^^^^^^^

Polar decomposition and singular value decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Operators on complex vector spaces
==================================


Operators on real vector spaces
===============================


Trace and determinant
=====================



Tensors
=======



Arrays of tensors
-----------------




Suggested Readings
==================

- `Wikipedia: Linear Algebra <https://en.wikipedia.org/wiki/Linear_algebra>`_
- `Khan Academy: Linear Algebra <https://www.khanacademy.org/math/linear-algebra>`_
- `MIT OpenCourseWare: Linear Algebra <https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/>`_
- `Linear Algebra Done Right by Sheldon Axler <https://www.springer.com/gp/book/9783319110790>`_

