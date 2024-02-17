#!/bin/bash
export NUMBA_DISABLE_JIT=1
poetry run pytest --cov-report=html:htmlcov --cov-config=.coveragerc_nojit --cov=sigmaepsilon.math
export NUMBA_DISABLE_JIT=0