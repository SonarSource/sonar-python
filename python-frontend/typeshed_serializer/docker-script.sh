#!/bin/sh

python runners/tox_runner.py --skip_tests=true
chown -R $UID:$GID /sonar-python/python-frontend/src