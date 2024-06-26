#
# SonarQube Python Plugin
# Copyright (C) 2011-2024 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import contextlib
import os
import sys
from os.path import isfile, join
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from collections.abc import Callable
import logging
import argparse

CURRENT_PATH = os.path.dirname(__file__)
CHECKSUM_FILE = os.path.join(CURRENT_PATH, '../checksum')
SERIALIZER_PATH = os.path.join(CURRENT_PATH, '../serializer')
RESOURCES_FOLDER_PATH = os.path.join(CURRENT_PATH, '../resources')
BINARY_FOLDER_PATH = os.path.join(CURRENT_PATH, '../../src/main/resources/org/sonar/python/types')
PROTOBUF_EXTENSION = '.protobuf'
PYTHON_STUB_EXTENSION = '.pyi'

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
    return ['requirements.txt', 'tox.ini']


def fetch_binary_file_names() -> list[str]:
    return sorted(fetch_resource_file_names(BINARY_FOLDER_PATH, PROTOBUF_EXTENSION))


def fetch_source_file_names(folder_path: str) -> list[str]:
    filenames = fetch_python_file_names(folder_path)
    resources = fetch_resource_file_names(RESOURCES_FOLDER_PATH, PYTHON_STUB_EXTENSION)
    config_files = fetch_config_file_names()
    return sorted([*filenames, *resources, *config_files])


def normalize_text_files(file_name: str) -> bytes:
    normalized_file = Path(file_name).read_text().strip().replace('\r\n', '\n').replace('\r', '\n')
    return bytes(normalized_file, 'utf-8')


def read_file(file_name: str) -> bytes:
    return Path(file_name).read_bytes()


def compute_checksum(file_names: list[str], get_file_bytes: Callable[[str], bytes]) -> str:
    _hash = hashlib.sha256()
    for fn in file_names:
        with contextlib.suppress(IsADirectoryError):
            _hash.update(get_file_bytes(fn))
    return _hash.hexdigest()


def read_previous_checksum(checksum_file: str) -> Tuple[Optional[str], Optional[str]]:
    def empty_str_to_none(s: str) -> Optional[str]:
        if not s:
            return None
        return s

    if not Path(checksum_file).is_file():
        return None, None
    with open(checksum_file, 'r') as file:
        source_checksum = empty_str_to_none(file.readline().strip())
        binaries_checksum = empty_str_to_none(file.readline().strip())
        return source_checksum, binaries_checksum


def update_checksum():
    with open(CHECKSUM_FILE, 'w') as file:
        source_file_names = fetch_source_file_names(SERIALIZER_PATH)
        source_checksum = compute_checksum(source_file_names, normalize_text_files)
        binary_file_names = fetch_binary_file_names()
        binary_checksum = compute_checksum(binary_file_names, read_file)
        file.writelines([f"{source_checksum}\n", binary_checksum])


def main(skip_tests=False, fail_fast=False):
    source_files = fetch_source_file_names(SERIALIZER_PATH)
    (previous_sources_checksum, previous_binaries_checksum) = read_previous_checksum(CHECKSUM_FILE)
    current_sources_checksum = compute_checksum(source_files, normalize_text_files)
    logger.info("STARTING TYPESHED SOURCE FILE CHECKSUM COMPUTATION")
    logger.info(f"Previous checksum {previous_sources_checksum}")
    logger.info(f"Current checksum {current_sources_checksum}")
    logger.info(f"Checksum is computed over {len(source_files)} files")
    if previous_sources_checksum != current_sources_checksum:
        if fail_fast:
            raise RuntimeError('INCONSISTENT BINARY CHECKSUMS')
        else:
            logger.info("STARTING TYPESHED SERIALIZATION")
            subprocess.run(["tox"], check=True)
    else:
        binary_file_names = fetch_binary_file_names()
        current_binaries_checksum = compute_checksum(binary_file_names, read_file)
        logger.info("STARTING TYPESHED BINARY FILES CHECKSUM COMPUTATION")
        logger.info(f"Previous binaries checksum {previous_binaries_checksum}")
        logger.info(f"Current binaries checksum {current_binaries_checksum}")
        logger.info(f"Checksum is computed over {len(binary_file_names)} files")
        if previous_binaries_checksum != current_binaries_checksum:
            raise RuntimeError('INCONSISTENT BINARY CHECKSUMS')
        logger.info("SKIPPING TYPESHED SERIALIZATION")
        # At the moment we need to run the tests in order to not break the quality gate.
        # If the tests are skipped this could potentially result in missing coverage.
        if skip_tests:
            logger.info("SKIPPING TYPESHED SERIALIZER TESTS")
            return
        subprocess.run(['tox', '-e', 'py39'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_tests')
    parser.add_argument('--fail_fast')
    args = parser.parse_args()
    skip_tests = args.skip_tests == "true"
    fail_fast = args.fail_fast == "true"
    main(skip_tests, fail_fast)
