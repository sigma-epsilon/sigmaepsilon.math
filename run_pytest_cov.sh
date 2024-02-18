#!/bin/bash
poetry run pytest --cov-report=xml --cov-config=.coveragerc --cov=sigmaepsilon.math
