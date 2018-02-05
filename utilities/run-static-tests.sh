#!/usr/bin/env bash

# echo commands; run tests; exit on first non-zero status
set -x \
    && mypy --strict pybotics \
    && flake8 \
    && vulture --exclude=docs,conftest.py,__init__.py . \
    && xenon --max-absolute B --max-modules B --max-average A pybotics \
    && pipdeptree -w fail -p pybotics \
    && bandit -r -v pybotics \
    && pipenv check pybotics \
    && safety check --full-report \
        -r requirements/main.txt \
        -r requirements/example-testing.txt \
        -r requirements/unit-testing.txt \
        -r requirements/deployment.txt \
        -r requirements/versioning.txt \
        -r requirements/static-testing.txt
