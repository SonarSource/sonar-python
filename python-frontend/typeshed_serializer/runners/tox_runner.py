import contextlib
import os
import sys
from os.path import isfile, join
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
from collections.abc import Callable
import logging

CHECKSUM_FILE = 'checksum'
CHECKSUM_BINARIES_FILE = 'checksum_binaries'
SERIALIZER_PATH = 'serializer'
BINARY_FOLDER_PATH = '../src/main/resources/org/sonar/python/types'

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


def fetch_source_file_names(folder_path: str) -> list[str]:
    filenames = fetch_python_file_names(folder_path)
    resources = fetch_resource_file_names(folder_name='resources', file_extension='.pyi')
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


def read_previous_checksum(checksum_file: str) -> Optional[str]:
    if not Path(checksum_file).is_file():
        return None
    with open(checksum_file, 'r') as file:
        return file.readline()


def update_checksum():
    with open(CHECKSUM_FILE, 'w') as file:
        all_files = fetch_source_file_names(SERIALIZER_PATH)
        file.write(compute_checksum(all_files, normalize_text_files))

    with open(CHECKSUM_BINARIES_FILE, 'w') as file:
        binary_file_names = fetch_resource_file_names(BINARY_FOLDER_PATH, file_extension='.protobuf')
        file.write(compute_checksum(binary_file_names, read_file))


def main():
    previous_sources_checksum = read_previous_checksum(CHECKSUM_FILE)
    all_files = fetch_source_file_names(SERIALIZER_PATH)
    current_sources_checksum = compute_checksum(all_files, normalize_text_files)
    logger.info("STARTING TYPESHED SOURCE FILE CHECKSUM COMPUTATION")
    logger.info(f"Previous checksum {previous_sources_checksum}")
    logger.info(f"Current checksum {current_sources_checksum}")
    logger.info(f"Checksum is computed over {len(all_files)} files")
    if previous_sources_checksum != current_sources_checksum:
        logger.info("STARTING TYPESHED SERIALIZATION")
        subprocess.run(["tox"])
    else:
        binary_file_names = fetch_resource_file_names(BINARY_FOLDER_PATH, file_extension='.protobuf')
        previous_binaries_checksum = read_previous_checksum(CHECKSUM_BINARIES_FILE)
        current_binaries_checksum = compute_checksum(binary_file_names, read_file)
        logger.info("STARTING TYPESHED BINARY FILES CHECKSUM COMPUTATION")
        logger.info(f"Previous binaries checksum {previous_binaries_checksum}")
        logger.info(f"Current binaries checksum {current_binaries_checksum}")
        logger.info(f"Checksum is computed over {len(binary_file_names)} files")
        if previous_binaries_checksum != current_binaries_checksum:
            raise RuntimeError('INCONSISTENT BINARY CHECKSUMS')
        logger.info("SKIPPING TYPESHED SERIALIZATION")


if __name__ == '__main__':
    main()
