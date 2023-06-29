===========================================================================
**sigmaepsilon.math** - A Python Library for Applied Mathematics in Physical Sciences
===========================================================================

.. admonition:: Important
   :class: caution

   sigmaepsilon.math is in the early stages of it's lifetime, and some concepts may change in 
   the future. If you want long-term stability, wait until version 1.0, which is 
   planned to be released if the core concepts all seem to sit. Nonetheless, the library 
   is well tested with a coverage value above 90%, so if something catches your eye use 
   it with confidence, just don't forget to pin down the version of sigmaepsilon.math in your 
   'requirements.txt'. 

.. include:: features.md
    :parser: myst_parser.sphinx_

.. admonition:: Important
   :class: important

   Be aware, that the library uses JIT-compilation through Numba, and as a result,
   first calls to these functions may take longer, but pay off in the long run. 

   .. admonition:: Tip
      :class: tip

      Create a minimal example that best describes a typical usage scenario of yours and
      name it like 'compile.py' or so. You can integrate this file into your CI/CD workflow.


.. include:: user_guide.md
    :parser: myst_parser.sphinx_

.. toctree::
    :maxdepth: 3
    :glob:
    
    notebooks

.. toctree::
    :maxdepth: 6
    :glob:
    :hidden:
    
    api

.. toctree::
    :glob:
    :hidden:
    
    license

   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
