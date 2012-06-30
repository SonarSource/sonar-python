#!/bin/sh
pylint --list-msgs | python convert.py > rules_generated.xml
