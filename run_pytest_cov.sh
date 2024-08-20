#!/bin/bash
poetry run pytest --cov-report=html:htmlcov --cov-config=.coveragerc --cov=sigmaepsilon.math
