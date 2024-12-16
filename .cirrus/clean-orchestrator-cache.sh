#!/bin/bash
cd "$ORCHESTRATOR_HOME" || exit 1

# Find all sonar-application-* JAR files, sort them by version, and list them
files=$(find . -name "sonar-application-*" | sort --version-sort --field-separator=- --key=3 --reverse)

# Print the files that will be kept (the latest one)
echo "Files that won't be deleted:"
echo "$files" | head -n 1

# Get the files that will be deleted (all except the latest one)
files_to_delete=$(echo "$files" | tail -n +2)

echo ""
# Check if there are files to delete
if [ -z "$files_to_delete" ]; then
  echo "No files will be deleted."
else
  # Print the files that will be deleted
  echo "Files that will be deleted:"
  echo "$files_to_delete"

  # Delete the files that will be deleted
  echo "$files_to_delete" | xargs -I {} sh -c 'rm -f "{}" && rmdir "$(dirname "{}")" 2>/dev/null || true'
fi
