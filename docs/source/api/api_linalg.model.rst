==========
Data Model
==========

.. include:: ..\global_refs.rst

The data model is a combination of the two approaches suggested by `NumPy`_ for creating
NumPy-compliant classes. All Vector and Tensor classes are custom array containers that 
wrap a direct subclass of NumPy's ndarray class (see `NumPy custom array containers`_ and
`subclassing ndarray`_ for the details). The double structure allows the container class
to manage and arbitrary array object in the baclground while maintaing a unified interface
to work with directly.