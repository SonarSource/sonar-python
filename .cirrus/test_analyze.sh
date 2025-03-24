#!/bin/bash
function run_maven {
  # No need for Maven phase "install" as the generated JAR files do not need to be installed
  # in Maven local repository. Phase "verify" is enough.
  mvn verify org.sonarsource.scanner.maven:sonar-maven-plugin:sonar \
    -Pcoverage \
    -Dmaven.test.redirectTestOutputToFile=false \
    -Dsonar.host.url="$SONAR_HOST_URL" \
    -Dsonar.token="$SONAR_TOKEN" \
    -Dsonar.analysis.buildNumber="$BUILD_NUMBER" \
    -Dsonar.analysis.pipeline="$PIPELINE_ID" \
    -Dsonar.analysis.sha1="$GIT_SHA1" \
    -Dsonar.analysis.repository="$GITHUB_REPO" \
    -Dsonar.analysisCache.enabled=true \
    -Dsonar.sca.recursiveManifestSearch=true \
    -Dsonar.sca.excludedManifests=python-frontend/typeshed_serializer/**,its/plugin/it-python-plugin-test/projects/**,private/its-enterprise/sources_ruling/**,private/its-enterprise/it-python-enterprise-plugin/projects/**,**/test/resources/** \
    -DfailStubGenerationFast=true \
    -Dskip.its=true \
    --batch-mode --errors --show-version \
    "$@"
}


# Fetch all commit history so that SonarQube has exact blame information
# for issue auto-assignment
# This command can fail with "fatal: --unshallow on a complete repository does not make sense"
# if there are not enough commits in the Git repository
# For this reason errors are ignored with "|| true"
git fetch --unshallow || true

# fetch references from github for PR analysis
if [ -n "${GITHUB_BASE_BRANCH}" ]; then
	git fetch origin "${GITHUB_BASE_BRANCH}"
fi

if [ -z "$PIPELINE_ID" ]; then
  PIPELINE_ID=$BUILD_NUMBER
fi

export MAVEN_OPTS="${MAVEN_OPTS:--Xmx1G -Xms128m}"

if [ "${GITHUB_BRANCH}" == "master" ] && [ "$PULL_REQUEST" == "false" ]; then
  echo '======= Build and analyze master'
  git fetch origin "${GITHUB_BRANCH}"
  # Analyze with SNAPSHOT version as long as SQ does not correctly handle
  # purge of release data
  run_maven "$@"
elif [[ "${GITHUB_BRANCH}" == "branch-"* ]] && [ "$PULL_REQUEST" == "false" ]; then
  echo '======= Build and analyze maintenance branches as long-living branches'

  git fetch origin "${GITHUB_BRANCH}"
  run_maven -Dsonar.branch.name="$GITHUB_BRANCH" "$@"

elif [ "$PULL_REQUEST" != "false" ]; then
  echo '======= Build and analyze pull request'
  run_maven -Dsonar.analysis.prNumber="$PULL_REQUEST" "$@"

elif [[ "$GITHUB_BRANCH" == "dogfood-on-"* ]] && [ "$PULL_REQUEST" == "false" ]; then
  echo '======= Build dogfood branch'
  mvn verify \
   -DfailStubGenerationFast=true \
   -Dskip.its=true \
   --batch-mode \
   --errors \
   --show-version \
   "$@"

else
  echo '======= Build, no analysis'

  # No need for Maven phase "install" as the generated JAR files do not need to be installed
  # in Maven local repository. Phase "verify" is enough.

  mvn verify \
      -Dmaven.test.redirectTestOutputToFile=false \
      --batch-mode \
       --errors \
       --show-version \
       "$@"
fi
