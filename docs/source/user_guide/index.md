---
myst:
  html_meta:
    description: "User guide of sigmaepsilon.math: linear algebra, functions and relations, approximation, optimization and graph theory, with worked examples."
---

(user_guide)=

# User Guide

The user guide provides a detailed walkthrough of the library, touching
the key features with useful background information and explanation.
If you want to know about the details of specific classes and functions,
please refer to the {doc}`API Reference <../api>`.

Every page of the user guide is a Jupyter Notebook that you can download from the
source repository and run on its own.

```{toctree}
:maxdepth: 1
:caption: Getting Started

introduction
installation
```

```{toctree}
:maxdepth: 3
:caption: Contents

linalg
function
approximation
optimization/index
graph
```

```{toctree}
:maxdepth: 1
:caption: Examples

/examples/lagrange_interpolation_1d
/examples/mls_approx_1d
/examples/mls_approx_2d
/examples/mls_approx_knn_1d
/examples/mls_approx_knn_nd
/examples/mls_class_knn_nd
```
