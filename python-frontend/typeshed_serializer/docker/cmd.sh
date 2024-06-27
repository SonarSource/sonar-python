#!/bin/sh

cd /sonar-python/python-frontend/typeshed_serializer
# Recreate the env to make sure that latest dependencies will be downloaded
python -m tox --recreate --notest
python runners/tox_runner.py --skip_tests=true
