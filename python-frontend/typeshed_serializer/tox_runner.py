import contextlib
import os
from os.path import isfile, join
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

CHECKSUM_FILE = "checksum"
SERIALIZER_PATH = "serializer"


def compute_checksum() -> str:
    _hash = hashlib.md5()
    filenames = [os.path.join(SERIALIZER_PATH, file) for file in os.listdir(SERIALIZER_PATH) if isfile(join(SERIALIZER_PATH, file)) and file.endswith(".py")]
    resources = [os.path.join(root, file) for root, _, files in os.walk("resources") for file in files if file.endswith(".pyi")]
    all_files = sorted([*filenames, *resources])
    for fn in all_files:
        with contextlib.suppress(IsADirectoryError):
            _hash.update(Path(fn).read_bytes())
    return _hash.hexdigest()


def read_previous_checksum() -> Optional[str]:
    if not Path(CHECKSUM_FILE).is_file():
        return None
    with open(CHECKSUM_FILE, 'r') as file:
        return file.readline()


def update_checksum():
    with open(CHECKSUM_FILE, 'w') as file:
        file.write(compute_checksum())


def main():
    previous_checksum = read_previous_checksum()
    current_checksum = compute_checksum()
    if previous_checksum is None or previous_checksum != current_checksum:
        print("STARTING TYPESHED SERIALIZATION ... ")
        print(f"Previous checksum {previous_checksum}")
        print(f"Current checksum {current_checksum}")
        subprocess.run(["tox"])
    else:
        print("SKIPPING TYPESHED SERIALIZATION")


if __name__ == '__main__':
    main()
