#!/bin/bash

set -euo pipefail

./tools/download-pycharm.sh

function installTravisTools {
  mkdir -p ~/.local
  curl -sSL https://github.com/SonarSource/travis-utils/tarball/v56 | tar zx --strip-components 1 -C ~/.local
  source ~/.local/bin/install
}

installTravisTools

export DEPLOY_PULL_REQUEST=true

regular_mvn_build_deploy_analyze

# FIXME(mpaladin) re-enable license check
#  ./check-license-compliance.sh
