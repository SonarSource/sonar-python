#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

rm -f flake8-report.txt
uvx flake8 flake8.py --output flake8-report.txt --format=pylint
