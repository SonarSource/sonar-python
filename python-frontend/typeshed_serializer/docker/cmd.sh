#!/bin/sh

mkdir -p /sonar-python/python-frontend/typeshed_serializer/serializer/proto_out
cd /sonar-python/python-frontend/src/main/protobuf

# SONARPY-3216 - to investigate how to deal with it properly
protoc -I=. --python_out=../../../typeshed_serializer/serializer/proto_out ./symbols.proto

cd /sonar-python/python-frontend/typeshed_serializer
# Sync dependencies from uv.lock
uv sync
uv run python runners/serializer_runner.py --skip_tests=false
