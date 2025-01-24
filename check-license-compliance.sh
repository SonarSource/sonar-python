#!/bin/bash
set -euo pipefail

mvn org.codehaus.mojo:license-maven-plugin:aggregate-add-third-party \
  -Dlicense.overrideUrl=file://$(pwd)/override-dep-licenses.properties \
  -P-private \
  "$@"
