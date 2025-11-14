#!/usr/bin/env python3

#
# SonarQube Python Plugin
# Copyright (C) 2011-2025 SonarSource Sàrl
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import argparse
import filecmp
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Validate license files against committed versions')
    parser.add_argument('--temp_licenses', help='Temporary licenses directory')
    parser.add_argument('--committed_licenses', help='Committed licenses directory')
    args = parser.parse_args()

    temp_path = Path(args.temp_licenses)
    committed_path = Path(args.committed_licenses)

    if not temp_path.exists():
        print(f"Error: Temporary licenses directory not found: {args.temp_licenses}")
        sys.exit(1)

    if not committed_path.exists():
        print(f"Error: Committed licenses directory not found: {args.committed_licenses}")
        print("This might be the first time generating licenses.")
        print("To create the committed license files, run:")
        print("  mvn clean package -PupdateLicenses")
        sys.exit(1)

    print("Validating generated license files against committed files...")

    # Use filecmp.dircmp for cross-platform directory comparison (equivalent to diff -r)
    dcmp = filecmp.dircmp(args.temp_licenses, args.committed_licenses)
    differences = collect_differences(dcmp)

    if differences:
        print("[FAILURE] License validation failed!")
        print("Generated license files differ from committed files.")
        print()
        print("Differences found:")
        for diff in differences:
            print(f"  {diff}")
        print()
        print("To update the committed license files, run:")
        print("  mvn clean package -PupdateLicenses")
        print()
        print("Note: This will completely regenerate all license files and remove any stale ones.")
        sys.exit(1)

    print("[SUCCESS] License validation passed - generated files match committed files")


def collect_differences(dcmp_obj, path_prefix=""):
    """Recursively collect all differences between directories"""
    differences = []
    for file in dcmp_obj.left_only:
        differences.append(f"New file generated: {os.path.join(path_prefix, file)}")
    for file in dcmp_obj.right_only:
        differences.append(f"Missing generated file: {os.path.join(path_prefix, file)}")
    for file in dcmp_obj.diff_files:
        differences.append(f"Content differs: {os.path.join(path_prefix, file)}")
    for subdir_name, subdir_dcmp in dcmp_obj.subdirs.items():
        differences.extend(collect_differences(subdir_dcmp, os.path.join(path_prefix, subdir_name)))
    return differences


if __name__ == '__main__':
    main()
