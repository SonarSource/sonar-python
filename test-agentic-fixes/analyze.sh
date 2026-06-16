#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
"/Users/romain.birling/Documents/sonar-bin/sonar-scanner-8.0.1.6346-macosx-x64/bin/sonar-scanner" \
  -Dsonar.host.url="${SONAR_HOST_URL:-http://localhost:9000}" \
  -Dsonar.projectKey="quick-fixes-agent-integration" \
  -Dsonar.projectName="quick-fixes-agent-integration" \
  -Dsonar.sources="." \
  -Dsonar.inclusions="file_to_analyze.py" \
  -Dsonar.token=$SONAR_TOKEN
