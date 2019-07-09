#!/bin/bash

set -euo pipefail

pushd $HOME
curl -L -O https://download-cf.jetbrains.com/python/pycharm-community-2019.1.3.tar.gz
tar xzf pycharm-community-2019.1.3.tar.gz
popd

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
