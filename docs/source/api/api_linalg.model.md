---
myst:
  html_meta:
    description: "The data model of the vector and tensor classes of sigmaepsilon.math.linalg: NumPy-compliant custom array containers wrapping a subclass of numpy.ndarray."
---

# Data Model

The data model is a combination of the two approaches suggested by [NumPy](https://numpy.org/doc/stable/index.html) for creating
NumPy-compliant classes. All Vector and Tensor classes are custom array containers that
wrap a direct subclass of NumPy's ndarray class (see [NumPy custom array containers](https://numpy.org/doc/stable/user/basics.dispatch.html) and
[subclassing ndarray](https://numpy.org/doc/stable/user/basics.subclassing.html) for the details). The double structure allows the container class
to manage an arbitrary array object in the background while maintaining a unified interface
to work with directly.
