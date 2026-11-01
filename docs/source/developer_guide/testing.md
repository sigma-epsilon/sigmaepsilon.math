---
myst:
  html_meta:
    description: "How to run the test suite of sigmaepsilon.math with pytest and generate an HTML coverage report."
---

# Testing and Coverage

The following command runs all tests and creates a html report in a folder named `htmlcov`
(the settings are governed by the `.coveragerc` file):

```shell
python -m pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.math
```

Alternatively, you can use `uv` to test the package (requires the `test` dependency group):

```shell
uv run pytest --cov-report=html --cov-config=.coveragerc --cov=sigmaepsilon.math
```
