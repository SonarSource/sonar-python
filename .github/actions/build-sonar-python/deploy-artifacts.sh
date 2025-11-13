#!/bin/bash
# Deploy to public and private Artifactory repositories using JFrog CLI
#
# Required environment variables:
# - ARTIFACTORY_URL: URL to Artifactory repository. Set by 'build-maven'.
# - ARTIFACTORY_DEPLOY_REPO: Repository to deploy public artifacts.
# - ARTIFACTORY_DEPLOY_ACCESS_TOKEN: Access token to deploy to public repository.
# - ARTIFACTORY_PRIVATE_DEPLOY_REPO: Repository to deploy private artifacts
# - ARTIFACTORY_PRIVATE_DEPLOY_ACCESS_TOKEN: Access token to deploy to private repository
# - INSTALLED_ARTIFACTS: Artifacts produced by Maven and installed in the local repository.
# - MAVEN_CONFIG: Path to the Maven configuration directory (typically $HOME/.m2). Set by 'build-maven'.

set -euo pipefail

: "${ARTIFACTORY_URL:?}" "${INSTALLED_ARTIFACTS:?}" "${MAVEN_CONFIG:?}"
: "${ARTIFACTORY_DEPLOY_REPO:?}" "${ARTIFACTORY_DEPLOY_ACCESS_TOKEN:?}"
: "${ARTIFACTORY_PRIVATE_DEPLOY_REPO:?}" "${ARTIFACTORY_PRIVATE_DEPLOY_ACCESS_TOKEN:?}"

public_artifacts=()
private_artifacts=()
for artifact in $INSTALLED_ARTIFACTS; do
  if [[ $artifact == "org/"* ]]; then
    public_artifacts+=("$artifact")
  elif [[ $artifact == "com/"* ]]; then
    private_artifacts+=("$artifact")
  else
    echo "WARN: Unrecognized artifact path: $artifact" >&2
  fi
done

# TODO BUILD-9723 review this function
extract_module_names() {
  artifact=$1
  module=$(echo "$artifact" | sed -E "s,^([^/]+/[^/]+/([^/]+))/([^/]+)/(([0-9].)+[0-9]+)/.*$,\1:\3:\4," | sed "s,/,.,g")
  echo "$module"
}

build_name="${GITHUB_REPOSITORY#*/}"
pushd "$MAVEN_CONFIG/repository"
jfrog config add deploy --artifactory-url "$ARTIFACTORY_URL" --access-token "$ARTIFACTORY_DEPLOY_ACCESS_TOKEN"
jfrog config use deploy
echo "Deploying public artifacts..."
for artifact in "${public_artifacts[@]}"; do
  module=$(extract_module_names "$artifact")
  jfrog rt u --module "$module" --build-name "$build_name" --build-number "$BUILD_NUMBER" "$artifact" "${ARTIFACTORY_DEPLOY_REPO}"
done
echo "Deploying private artifacts..."
jfrog config edit deploy --artifactory-url "$ARTIFACTORY_URL" --access-token "$ARTIFACTORY_PRIVATE_DEPLOY_ACCESS_TOKEN"
for artifact in "${private_artifacts[@]}"; do
  module=$(extract_module_names "$artifact")
  jfrog rt u --module "$module" --build-name "$build_name" --build-number "$BUILD_NUMBER" "$artifact" "${ARTIFACTORY_PRIVATE_DEPLOY_REPO}"
done
popd
