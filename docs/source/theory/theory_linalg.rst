============
Theory Guide
============

---------------
Linear Algebra
---------------

The Direction Cosine Matrix
===========================

The notion of the *Direction Cosine Matrix* (DCM) is meant to unify the direction of 
relative transformation between two frames.

.. note::
   Click :download:`here <../_static/linalgR3.pdf>` to read the extended version of this
   brief intro as a pdf document.

If a vector :math:`\mathbf{v}` is given in frames :math:`\mathbf{A}` and :math:`\mathbf{B}` as

.. math::
   :nowrap:
   
   \begin{equation}
   \mathbf{v} = \alpha_1 \mathbf{a}_{1} + \alpha_2 \mathbf{a}_{2} = \beta_1 \mathbf{b}_{1} + \beta_2 \mathbf{b}_{2},
   \end{equation}
   
then the matrix :math:`^{A}\mathbf{R}^{B}` is called *the DCM from A to B*. It transforms the components as

.. math::
   :nowrap:

   \begin{equation}
   \left[
   \boldsymbol{\beta}
   \right]
   = 
   \left[
   ^{A}\mathbf{R}^{B}
   \right]
   \left[ \boldsymbol{\alpha}\right],
   \quad
   \left[
   \boldsymbol{\alpha}
   \right]
   = 
   \left[
   ^{A}\mathbf{R}^{B}
   \right]^{-1}
   \left[ \boldsymbol{\beta}\right]
   =
   \left[
   ^{B}\mathbf{R}^{A}
   \right]
   \left[ \boldsymbol{\beta}\right]
   \end{equation}

and the base vectors as

.. math::
   :nowrap:

   \begin{equation}
   \left[
   \mathbf{b}_i
   \right]
   = 
   \left[
   ^{B}\mathbf{R}^{A}
   \right]
   \left[ \mathbf{a}_i \right], \quad (i=1,2,3,...).
   \end{equation}