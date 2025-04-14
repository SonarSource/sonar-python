#!/bin/bash

# Function to clean up old versions of files based on a prefix
# Parameters:
#   $1: The file prefix (e.g., "sonarqube-enterprise", "sonar-application")
cleanup_old_versions() {
  local file_prefix="$1"
  # Ensure a prefix was provided
  if [ -z "$file_prefix" ]; then
    echo "Error: No file prefix provided to cleanup_old_versions function."
    return 1 # Indicate error
  fi

  local search_pattern="${file_prefix}-*"

  echo "--- Processing files starting with '${file_prefix}' ---"

  # Find all matching files, sort them by version (descending), and list them
  # Assuming version is the 3rd field when splitting by '-'
  # Using process substitution and mapfile for safer handling of filenames
  local -a all_files
  mapfile -t all_files < <(find . -name "$search_pattern" | sort --version-sort --field-separator=- --key=3 --reverse)

  # Check if any files were found
  if [ ${#all_files[@]} -eq 0 ]; then
    echo "No files found matching '${search_pattern}'."
    echo "--- Finished processing '${file_prefix}' ---"
    echo ""
    return 0 # Nothing to do, success
  fi

  # The first file in the sorted list is the latest one to keep
  local latest_file="${all_files[0]}"
  echo "Latest version (will be kept):"
  echo "$latest_file"

  # Get the files to delete (all except the latest one)
  local -a files_to_delete=("${all_files[@]:1}") # Slice the array starting from the second element

  echo ""
  echo "Files to delete:"
  for file in "${files_to_delete[@]}"; do
    echo "$file"
    rm -f "$file" && rmdir "$(dirname "$file")" 2>/dev/null || true
  done

  echo "--- Finished processing '${file_prefix}' ---"
  echo ""
  return 0 # Indicate success
}

# --- Main Script Logic ---

# Check if ORCHESTRATOR_HOME is set and is a directory
if [ -z "$ORCHESTRATOR_HOME" ]; then
  echo "Error: ORCHESTRATOR_HOME environment variable is not set."
  exit 1
elif [ ! -d "$ORCHESTRATOR_HOME" ]; then
  echo "Error: ORCHESTRATOR_HOME ('$ORCHESTRATOR_HOME') is not a valid directory."
  exit 1
fi

# Change to the target directory. Exit if failed.
cd "$ORCHESTRATOR_HOME" || {
  echo "Error: Could not change directory to '$ORCHESTRATOR_HOME'"
  exit 1
}
echo "Changed directory to $ORCHESTRATOR_HOME"
echo ""

# Call the cleanup function for each desired prefix
cleanup_old_versions "sonarqube-enterprise"
cleanup_old_versions "sonar-application"

echo "All cleanup tasks finished."
exit 0
