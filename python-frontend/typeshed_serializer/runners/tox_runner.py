#
# SonarQube Python Plugin
# Copyright (C) 2011-2025 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import contextlib
import os
import sys
from os.path import isfile, join
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union, List
from collections.abc import Callable
import logging
import argparse

CURRENT_PATH = os.path.dirname(__file__)
SERIALIZER_SOURCE_CHECKSUM_FILE = os.path.join(CURRENT_PATH, '../serializer_sources.checksum')
CHECKSUMS_DIR = os.path.join(CURRENT_PATH, '../checksums')
SERIALIZER_PATH = os.path.join(CURRENT_PATH, '../serializer')
RESOURCES_FOLDER_PATH = os.path.join(CURRENT_PATH, '../resources')
BINARY_FOLDER_PATH = os.path.join(CURRENT_PATH, '../../src/main/resources/org/sonar/python/types')
PROTOBUF_EXTENSION = '.protobuf'
PYTHON_STUB_EXTENSION = '.pyi'

# Resources subfolders and their corresponding serializer and binary folders
RESOURCES_FOLDERS = {
    'custom': {
        'serializer': 'custom',
        'binary_folder': 'custom_protobuf',
        'source_path': 'custom',
    },
    'importer': {
        'serializer': 'importer',
        'binary_folder': 'third_party_protobuf_mypy',
        'source_path': 'importer',
    },
    'microsoft': {
        'serializer': 'microsoft',
        'binary_folder': 'third_party_protobuf_microsoft',
        'source_path': os.path.join('python-type-stubs','stubs','sklearn'),
    },
    'typeshed_stdlib': {
        'serializer': 'stdlib',
        'binary_folder': 'stdlib_protobuf',
        'source_path': os.path.join('typeshed','stdlib'),
    },
    'typeshed_stubs': {
        'serializer': 'third_party',
        'binary_folder': 'third_party_protobuf',
        'source_path': os.path.join('typeshed','stubs'),
    },
}

logger = logging.getLogger('tox_runner')
handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter(fmt='%(name)s [%(levelname)s] --- %(message)s ---')
logger.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.addHandler(handler)


def fetch_python_file_names(folder_path: str) -> list[str]:
    result = list()
    for file in os.listdir(folder_path):
        if isfile(join(folder_path, file)) and file.endswith('.py'):
            result.append(join(folder_path, file))
    return result


def fetch_resource_file_names(folder_name: str, file_extension: str) -> list[str]:
    result = list()
    for root, _, files in os.walk(folder_name):
        for file in files:
            if file.endswith(file_extension):
                result.append(join(root, file))
    return result


def fetch_config_file_names() -> list[str]:
    return [os.path.join(CURRENT_PATH, '../requirements.txt'), os.path.join(CURRENT_PATH, '../tox.ini')]

def fetch_source_file_names(folder_path: str) -> list[str]:
    filenames = fetch_python_file_names(folder_path)
    config_files = fetch_config_file_names()
    return sorted([*filenames, *config_files])


def fetch_resources_subfolder_files(subfolder_name: str) -> list[str]:
    """Fetch all files from a specific resources subfolder."""
    subfolder_path = os.path.join(RESOURCES_FOLDER_PATH, subfolder_name)
    if not os.path.exists(subfolder_path):
        return []

    all_files = []
    for root, dirs, files in os.walk(subfolder_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            # Skip hidden files
            if not file.startswith('.'):
                all_files.append(os.path.join(root, file))

    return sorted(all_files)


def fetch_binary_files_for_folder(folder_name: str) -> List[str]:
    """Fetch all .protobuf files for a specific resources folder."""
    if folder_name not in RESOURCES_FOLDERS:
        return []

    folder_config = RESOURCES_FOLDERS[folder_name]
    binary_folder = folder_config['binary_folder']
    binary_path = os.path.join(BINARY_FOLDER_PATH, binary_folder)

    if not os.path.exists(binary_path):
        return []

    return sorted(fetch_resource_file_names(binary_path, PROTOBUF_EXTENSION))


def normalize_text_files(file_name: str) -> bytes:
    normalized_file = Path(file_name).read_text(encoding='utf-8').strip().replace('\r\n', '\n').replace('\r', '\n')
    return bytes(normalized_file, 'utf-8')


def read_file(file_name: str) -> bytes:
    return Path(file_name).read_bytes()


def compute_checksum(file_names: list[str], get_file_bytes: Callable[[str], bytes]) -> str:
    _hash = hashlib.sha256()
    for fn in file_names:
        with contextlib.suppress(IsADirectoryError):
            _hash.update(get_file_bytes(fn))
    return _hash.hexdigest()


def read_previous_checksum(checksum_file: str) -> Optional[str]:
    def empty_str_to_none(s: str) -> Optional[str]:
        if not s:
            return None
        return s

    if not Path(checksum_file).is_file():
        return None
    with open(checksum_file, 'r') as file:
        source_checksum = empty_str_to_none(file.readline().strip())
        return source_checksum


def read_previous_folder_checksum(folder_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Read the previous source and binary checksums for a specific resources folder."""
    checksum_file = os.path.join(CHECKSUMS_DIR, f'{folder_name.replace("/", "_")}.checksum')
    if not Path(checksum_file).is_file():
        return None, None
    with open(checksum_file, 'r') as file:
        source_checksum = file.readline().strip() or None
        binary_checksum = file.readline().strip() or None
        return source_checksum, binary_checksum


def write_folder_checksum(folder_name: str, source_checksum: str, binary_checksum: str):
    """Write the source and binary checksums for a specific resources folder."""
    os.makedirs(CHECKSUMS_DIR, exist_ok=True)
    checksum_file = os.path.join(CHECKSUMS_DIR, f'{folder_name.replace("/", "_")}.checksum')
    with open(checksum_file, 'w') as file:
        file.writelines([f'{source_checksum}\n', binary_checksum])


def update_all_checksums():
    with open(SERIALIZER_SOURCE_CHECKSUM_FILE, 'w') as file:
        source_file_names = fetch_source_file_names(SERIALIZER_PATH)
        serializer_source_checksum = compute_checksum(source_file_names, normalize_text_files)
        file.write(serializer_source_checksum)

    # Update ressource & binary checksums for each resources subfolder
    for folder_name, folder_config in RESOURCES_FOLDERS.items():
        folder_files = fetch_resources_subfolder_files(folder_config['source_path'])
        if folder_files:  # Only update if folder has files
            ressource_checksum = compute_checksum(folder_files, normalize_text_files)
            # For binary checksum, we need to find corresponding binary files for this specific folder
            binary_files = fetch_binary_files_for_folder(folder_name)
            binary_checksum = compute_checksum(binary_files, read_file) if binary_files else ''
            write_folder_checksum(folder_name, ressource_checksum, binary_checksum)


def detect_required_serializations(fail_fast=False) -> List[str]:
    """Check which resources folders have changed and return the list of serializers to run."""
    ressources_that_needs_serialization = []

    for folder_name, folder_config in RESOURCES_FOLDERS.items():
        folder_files = fetch_resources_subfolder_files(folder_config['source_path'])
        if not folder_files:  # Skip empty folders
            continue

        binary_files = fetch_binary_files_for_folder(folder_name)

        previous_source_checksum, previous_binary_checksum = read_previous_folder_checksum(folder_name)

        ressources_changed = compute_and_compare_checksums("SOURCE", folder_name, folder_files, normalize_text_files, previous_source_checksum)
        binary_changed = compute_and_compare_checksums("BINARY", folder_name, binary_files, read_file, previous_binary_checksum)

        if ressources_changed or binary_changed:
            # Check for binary inconsistency (binary changed but source didn't)
            if binary_changed and not ressources_changed and fail_fast:
                raise RuntimeError('INCONSISTENT RESOURCE FOLDER BINARY CHECKSUMS')
            ressources_that_needs_serialization.append(folder_config['serializer'])

    return ressources_that_needs_serialization


def compute_and_compare_checksums(check_type: str, folder_name: str, folder_files: list[str], get_file_bytes: Callable[[str], bytes], previous_checksum: str) -> bool:
    current_checksum = compute_checksum(folder_files, get_file_bytes)
    checksum_changed = current_checksum != previous_checksum
    if checksum_changed:
        logger.info(f'{check_type} FILES CHANGED in folder: {folder_name}')
        logger.info(f'Previous checksum: {previous_checksum}')
        logger.info(f'Current checksum: {current_checksum}')
        logger.info(f'Checksum is computed over {len(folder_files)} files')
    else:
        logger.info(f'{check_type} FILES UNCHANGED in folder: {folder_name} - Computed over {len(folder_files)} files - Checksum: {current_checksum}')
    return checksum_changed



def get_serialize_command_to_run(previous_source_checksum: Optional[str], current_sources_checksum: str, changed_serializers: List[str]) -> Optional[List[str]]:
    """Determine the serialization command to run based on checksums and changed serializers.

    Args:
        previous_source_checksum: Previous checksum of source files
        current_sources_checksum: Current checksum of source files
        changed_serializers: List of serializers that have changed

    Returns:
        Command to run as list of strings, or None if no serialization needed
    """
    if previous_source_checksum != current_sources_checksum:
        # Serializer code has changed - run full serialization
        logger.info('SERIALIZER CODE HAS CHANGED - STARTING FULL TYPESHED SERIALIZATION')
        return ['tox', '-e', 'serialize']
    elif changed_serializers:
        logger.info(f"STARTING SELECTIVE TYPESHED SERIALIZATION FOR: {','.join(changed_serializers)}")
        # Run selective serialization through tox environment
        serializers_arg = ','.join(changed_serializers)
        return ['tox', '-e', 'selective-serialize', '--', serializers_arg]
    else:
        logger.info('SKIPPING TYPESHED SERIALIZATION')
        return None


def update_folder_checksums_for_changed_serializers(changed_serializers: List[str]) -> None:
    """Update checksums for resource folders that correspond to changed serializers."""
    for folder_name, folder_config in RESOURCES_FOLDERS.items():
        if folder_config['serializer'] in changed_serializers:
            folder_files = fetch_resources_subfolder_files(folder_config['source_path'])
            if folder_files:
                source_checksum = compute_checksum(folder_files, normalize_text_files)
                # Compute binary checksum from corresponding binary folder
                binary_files = fetch_binary_files_for_folder(folder_name)
                binary_checksum = compute_checksum(binary_files, read_file)
                write_folder_checksum(folder_name, source_checksum, binary_checksum)


def main(skip_tests=False, fail_fast=False, dry_run=False):
    # Check if serializer source code has changed
    source_files = fetch_source_file_names(SERIALIZER_PATH)
    current_sources_checksum = compute_checksum(source_files, normalize_text_files)
    previous_sources_checksum = read_previous_checksum(SERIALIZER_SOURCE_CHECKSUM_FILE)
    serializer_sources_changed = compute_and_compare_checksums("SERIALIZER_SOURCE", SERIALIZER_PATH, source_files, normalize_text_files, previous_sources_checksum)

    if serializer_sources_changed and fail_fast :
        raise RuntimeError('INCONSISTENT SOURCES CHECKSUMS')

    changed_serializers = detect_required_serializations(fail_fast)
    serialize_command_to_run = get_serialize_command_to_run(previous_sources_checksum, current_sources_checksum, changed_serializers)

    # Execute or display the serialize command
    if serialize_command_to_run:
        if dry_run:
            logger.info(f'DRY RUN: Would execute: {" ".join(serialize_command_to_run)}')
        else:
            _ = subprocess.run(serialize_command_to_run, check=True)
            update_folder_checksums_for_changed_serializers(changed_serializers)

    # Run tests after serialization (unless skip_tests=True)
    if not skip_tests:
        if dry_run:
            logger.info('DRY RUN: Would run test')
        else:
            _ = subprocess.run(['tox', '-e', 'py39'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_tests')
    parser.add_argument('--fail_fast')
    parser.add_argument('--dry_run')
    args = parser.parse_args()
    logger.info(f'skip_tests: {args.skip_tests}')
    logger.info(f'fail_fast: {args.fail_fast}')
    logger.info(f'dry_run: {args.dry_run}')
    skip_tests = args.skip_tests == "true"
    fail_fast = args.fail_fast == "true"
    dry_run = args.dry_run == "true"
    main(skip_tests, fail_fast, dry_run)
