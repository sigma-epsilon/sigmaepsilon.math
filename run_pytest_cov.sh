#!/bin/bash
poetry run pytest --cov --cov-report=html:htmlcov --cov-config=.coveragerc --cov=sigmaepsilon.math
