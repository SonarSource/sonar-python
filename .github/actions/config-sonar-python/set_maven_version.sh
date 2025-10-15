#!/bin/bash
# Config script for SonarSource Maven projects.
#
# Required environment variables (must be explicitly provided):
# - BUILD_NUMBER: Build number for versioning

set -euo pipefail

: "${BUILD_NUMBER:?}"

get_current_version() {
  local expression="project.version"
  mvn -q -Dexec.executable="echo" -Dexec.args="\${$expression}" --non-recursive org.codehaus.mojo:exec-maven-plugin:1.3.1:exec -Dexec.outputFile=/tmp/current_version.txt --show-version >/dev/null 2>&1
  local maven_exit_code=$?

  if [ ! -s /tmp/current_version.txt ] || [ $maven_exit_code -ne 0 ]; then
    echo "Failed to evaluate Maven expression '$expression'" >&2
    mvn -X -Dexec.executable="echo" -Dexec.args="\${$expression}" --non-recursive org.codehaus.mojo:exec-maven-plugin:1.3.1:exec
    return 1
  fi

  cat /tmp/current_version.txt
}

# Set the project version as <MAJOR>.<MINOR>.<PATCH>.<BUILD_NUMBER>
# Update current_version variable with the current project version.
# Then remove the -SNAPSHOT suffix if present, complete with '.0' if needed, and append the build number at the end.
set_project_version() {
  local current_version
  if ! current_version=$(get_current_version 2>&1); then
    echo -e "::error file=pom.xml,title=Maven project version::Could not get 'project.version' from Maven project\nERROR: $current_version"
    return 1
  fi

  local release_version="${current_version%"-SNAPSHOT"}"
  local dots="${release_version//[^.]/}"
  local dots_count="${#dots}"

  if [[ "$dots_count" -eq 0 ]]; then
    release_version="${release_version}.0.0"
  elif [[ "$dots_count" -eq 1 ]]; then
    release_version="${release_version}.0"
  elif [[ "$dots_count" -ne 2 ]]; then
    echo "::error file=pom.xml,title=Maven project version::Unsupported version '$current_version' with $((dots_count + 1)) digits."
    return 1
  fi
  release_version="${release_version}.${BUILD_NUMBER}"
  echo "Replacing version ${current_version} with ${release_version}"
  mvn org.codehaus.mojo:versions-maven-plugin:2.7:set -DnewVersion="$release_version" -DgenerateBackupPoms=false --batch-mode --no-transfer-progress --errors
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set_project_version
fi
