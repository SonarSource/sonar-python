#!/usr/bin/env bash
#
# verify-fix-versions.sh
#
# This script verifies that all Jira tickets referenced in commits since the last release
# have a fix version set. It outputs results to GITHUB_STEP_SUMMARY.
#
# Required environment variables:
#   JIRA_USER  - Jira username for API authentication
#   JIRA_TOKEN - Jira API token for authentication
#
# Optional environment variables:
#   JIRA_PROJECT_KEY - Jira project key (default: SONARPY)
#   JIRA_BASE_URL    - Jira base URL (default: https://sonarsource.atlassian.net)
#

set -euo pipefail

# Configuration
JIRA_PROJECT_KEY="${JIRA_PROJECT_KEY:-SONARPY}"
JIRA_BASE_URL="${JIRA_BASE_URL:-https://sonarsource.atlassian.net}"
JIRA_API_URL="${JIRA_BASE_URL}/rest/api/3/search/jql"

echo "Using Jira instance: ${JIRA_BASE_URL}"

if [[ -z "${JIRA_USER:-}" ]] || [[ -z "${JIRA_TOKEN:-}" ]]; then
  echo "Error: JIRA_USER and JIRA_TOKEN environment variables are required"
  exit 1
fi

for tool in git curl jq python3; do
  if ! command -v "$tool" &> /dev/null; then
    echo "Error: Required tool '$tool' is not installed"
    exit 1
  fi
done

get_last_release_tag() {
  git tag --sort=-version:refname | head -1
}

get_commits_since_tag() {
  local tag="$1"
  git log "${tag}..HEAD" --oneline --no-merges
}

extract_jira_tickets() {
  local commits="$1"
  # Extract SONARPY-XXXX patterns, remove duplicates, and sort
  echo "$commits" | grep -oE "${JIRA_PROJECT_KEY}-[0-9]+" | sort -u
}

build_jql_query() {
  local tickets="$1"
  local ticket_list
  ticket_list=$(echo "$tickets" | tr '\n' ',' | sed 's/,$//')
  echo "issue in (${ticket_list}) AND fixVersion IS EMPTY"
}

execute_jql_query() {
  local jql="$1"
  local response
  local curl_exit_code=0
  
  local payload
  payload=$(jq -n --arg jql "$jql" '{"jql": $jql, "fields": ["key", "summary", "status"]}')
  
  set +e
  response=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -u "${JIRA_USER}:${JIRA_TOKEN}" \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "${JIRA_API_URL}" 2>&1)
  curl_exit_code=$?
  set -e
  
  if [[ "$curl_exit_code" -ne 0 ]]; then
    echo "Error: curl request failed with exit code ${curl_exit_code}" >&2
    echo "Response: $response" >&2
    exit 1
  fi
  
  local http_code
  http_code=$(echo "$response" | tail -1)
  local body
  body=$(echo "$response" | sed '$d')
  
  if [[ "$http_code" != "200" ]]; then
    echo "Error: Jira API returned HTTP $http_code" >&2
    echo "Response: $body" >&2
    exit 1
  fi
  
  echo "$body"
}

generate_summary() {
  local last_tag="$1"
  local commit_count="$2"
  local ticket_count="$3"
  local jql_query="$4"
  local jira_response="$5"
  
  local issues_without_fixversion
  issues_without_fixversion=$(echo "$jira_response" | jq -r '.issues | length')
  
  # Debug: if still getting unexpected values, show the response structure
  if [[ ! "$issues_without_fixversion" =~ ^[0-9]+$ ]]; then
    echo "Error: Unexpected response from Jira API" >&2
    echo "Response: $jira_response" >&2
    exit 1
  fi
  
  {
    echo "# Jira Fix Version Verification Report"
    echo ""
    echo "## Summary"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Last Release Tag | \`${last_tag}\` |"
    echo "| Commits Since Release | ${commit_count} |"
    echo "| Unique Jira Tickets Found | ${ticket_count} |"
    echo "| Tickets Missing Fix Version | **${issues_without_fixversion}** |"
    echo ""
    echo "## JQL Query"
    echo ""
    echo "\`\`\`"
    echo "${jql_query}"
    echo "\`\`\`"
    echo ""
    echo "[Run this query in Jira](${JIRA_BASE_URL}/issues/?jql=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$jql_query'''))"))"
    echo ""
    
    if [[ "$issues_without_fixversion" -gt 0 ]]; then
      echo "## ⚠️ Issues Without Fix Version"
      echo ""
      echo "The following tickets need a fix version to be set before release:"
      echo ""
      echo "| Ticket | Summary | Status |"
      echo "|--------|---------|--------|"
      
      # Parse issues from response and format as table rows
      echo "$jira_response" | jq -r '.issues[] | "| [\(.key)]('"${JIRA_BASE_URL}"'/browse/\(.key)) | \(.fields.summary | gsub("\\|"; "\\\\|") | .[0:60]) | \(.fields.status.name) |"'
      
      echo ""
      echo "---"
      echo "**Action Required:** Please set the fix version on the above tickets before proceeding with the release."
      echo ""
      echo "If the missing fix version is intentional:"
      echo "- Run \`VerifyJiraFixVersions\` manually with \`fail-on-missing: false\`"
      echo "- Or run \`DoRelease\` with \`fail-on-missing-fix-version: false\`"
    else
      echo "## All Tickets Have Fix Version Set"
      echo ""
      echo "All Jira tickets referenced in commits since \`${last_tag}\` have a fix version assigned."
    fi
  } >> "$GITHUB_STEP_SUMMARY"
  
  if [[ "$issues_without_fixversion" -gt 0 ]]; then
    echo ""
    echo "⚠️  Found ${issues_without_fixversion} ticket(s) without fix version. See job summary for details."
    echo "   To bypass: run VerifyJiraFixVersions with fail-on-missing: false"
    echo "              or DoRelease with fail-on-missing-fix-version: false"
    return 1
  else
    echo "All tickets have fix version set. See job summary for details."
    return 0
  fi
}

handle_no_tags() {
  {
    echo "# Jira Fix Version Verification Report"
    echo ""
    echo "## ⚠️ No Release Tags Found"
    echo ""
    echo "No release tags were found in the repository."
    echo ""
    echo "If this is expected:"
    echo "- Run \`VerifyJiraFixVersions\` manually with \`fail-on-missing: false\`"
    echo "- Or run \`DoRelease\` with \`fail-on-missing-fix-version: false\`"
  } >> "$GITHUB_STEP_SUMMARY"
  
  echo "⚠️  No release tags found in the repository. See job summary for details."
  echo "   To bypass: run VerifyJiraFixVersions with fail-on-missing: false"
  echo "              or DoRelease with fail-on-missing-fix-version: false"
  return 1
}

handle_no_commits() {
  local last_tag="$1"
  
  {
    echo "# Jira Fix Version Verification Report"
    echo ""
    echo "## ⚠️ No Commits Found"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Last Release Tag | \`${last_tag}\` |"
    echo "| Commits Since Release | 0 |"
    echo ""
    echo "No commits were found since the last release tag \`${last_tag}\`."
    echo ""
    echo "This is unusual and might indicate a problem (e.g., wrong branch, missing tags)."
    echo ""
    echo "If this is expected:"
    echo "- Run \`VerifyJiraFixVersions\` manually with \`fail-on-missing: false\`"
    echo "- Or run \`DoRelease\` with \`fail-on-missing-fix-version: false\`"
  } >> "$GITHUB_STEP_SUMMARY"
  
  echo "⚠️  No commits found since ${last_tag}. See job summary for details."
  echo "   To bypass: run VerifyJiraFixVersions with fail-on-missing: false"
  echo "              or DoRelease with fail-on-missing-fix-version: false"
  return 1
}

handle_no_tickets() {
  local last_tag="$1"
  local commit_count="$2"
  
  {
    echo "# Jira Fix Version Verification Report"
    echo ""
    echo "## Summary"
    echo ""
    echo "| Metric | Value |"
    echo "|--------|-------|"
    echo "| Last Release Tag | \`${last_tag}\` |"
    echo "| Commits Since Release | ${commit_count} |"
    echo "| Unique Jira Tickets Found | 0 |"
    echo ""
    echo "## No Jira Tickets Found"
    echo ""
    echo "No commits with Jira ticket references (${JIRA_PROJECT_KEY}-XXXX) were found since the last release."
  } >> "$GITHUB_STEP_SUMMARY"
  
  echo "No Jira tickets found in commits since ${last_tag}. See job summary for details."
}

main() {
  echo "Fetching last release tag..."
  local last_tag
  last_tag=$(get_last_release_tag)
  
  if [[ -z "$last_tag" ]]; then
    handle_no_tags
  fi
  echo "   Last release: ${last_tag}"
  
  echo "Getting commits since ${last_tag}..."
  local commits
  commits=$(get_commits_since_tag "$last_tag")
  
  if [[ -z "$commits" ]]; then
    echo "   No commits found since last release"
    handle_no_commits "$last_tag"
  fi
  
  local commit_count
  commit_count=$(echo "$commits" | wc -l | tr -d ' ')
  echo "   Found ${commit_count} commit(s)"
  
  echo "Extracting Jira tickets..."
  local tickets
  tickets=$(extract_jira_tickets "$commits")
  
  if [[ -z "$tickets" ]]; then
    echo "   No Jira tickets found in commits"
    handle_no_tickets "$last_tag" "$commit_count"
    exit 0
  fi
  
  local ticket_count
  ticket_count=$(echo "$tickets" | wc -l | tr -d ' ')
  echo "   Found ${ticket_count} unique ticket(s)"
  
  echo "Building JQL query..."
  local jql_query
  jql_query=$(build_jql_query "$tickets")
  echo "   Query: ${jql_query}"
  
  echo "Querying Jira API..."
  local jira_response
  jira_response=$(execute_jql_query "$jql_query")
  
  echo "Generating summary..."
  generate_summary "$last_tag" "$commit_count" "$ticket_count" "$jql_query" "$jira_response"
}

main "$@"


