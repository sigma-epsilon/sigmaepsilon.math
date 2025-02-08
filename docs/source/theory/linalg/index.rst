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

**Linear algebra makes things better.**

Isn't there something you want to make better?

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

The important thing to note here that the matrix of a linear map depends on the linear map, and the bases of the 
vector spaces involved. The matrix of a linear map is not an intrinsic property of the linear map itself, but rather
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


Polynomials
===========


Tensors
=======



Arrays
------




Suggested Readings
==================

- `Wikipedia: Linear Algebra <https://en.wikipedia.org/wiki/Linear_algebra>`_
- `Khan Academy: Linear Algebra <https://www.khanacademy.org/math/linear-algebra>`_
- `MIT OpenCourseWare: Linear Algebra <https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/>`_
- `Linear Algebra Done Right by Sheldon Axler <https://www.springer.com/gp/book/9783319110790>`_

