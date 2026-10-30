====================
Testing and Coverage
====================

The following command runs all tests and creates a html report in a folder named `htmlcov` 
(the settings are governed by the `.coveragerc` file):

.. code-block:: shell

   python -m pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.math

Alternatively, you can use `uv` to test the package (requires the `test` dependency group):

.. code-block:: shell

   uv run pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.math
