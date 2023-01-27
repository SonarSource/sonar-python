#! /usr/bin/env bash

set -euox pipefail

get_project_version() {
  local version_property
  version_property=$(./gradlew properties | grep --extended-regexp "^version: (.*)")
  if [[ -z "${version_property}" ]]; then
    echo "Could not find property version in project" >&2
    exit 2
  fi
  local version
  version=$(echo "${version_property}" | tr --delete "[:space:]" | cut --delimiter=":" --fields=2)
  version="${version/-SNAPSHOT/}"
  # Because the ws scan script expects a semver-like version (aa.bb.cc.XX), we append the build number to the project version.
  if [[ "${version}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    version="${version}.0"
  fi
  version="${version}.${BUILD_NUMBER:-0}"
  echo "${version}"
}

export PROJECT_VERSION="$(get_project_version)"
