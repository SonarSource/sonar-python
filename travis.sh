#!/bin/bash

set -euo pipefail

function installTravisTools {
  mkdir -p ~/.local
  curl -sSL https://github.com/SonarSource/travis-utils/tarball/v44 | tar zx --strip-components 1 -C ~/.local
  source ~/.local/bin/install
}

installTravisTools
source ~/.local/bin/installMaven35
source ~/.local/bin/installJDK8

export DEPLOY_PULL_REQUEST=true

regular_mvn_build_deploy_analyze
