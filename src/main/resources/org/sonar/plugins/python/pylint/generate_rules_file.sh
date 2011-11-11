#!/bin/sh
pylint --list-msgs | python convert.py > rules.xml
