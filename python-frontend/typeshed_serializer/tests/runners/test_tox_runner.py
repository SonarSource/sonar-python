#
# SonarQube Python Plugin
# Copyright (C) 2011-2024 SonarSource SA
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

import unittest
from unittest import mock
from unittest.mock import mock_open
import os
from six import PY2

from runners import tox_runner
from runners.tox_runner import CHECKSUM_FILE

CURRENT_PATH = os.path.dirname(__file__)


class ToxRunnerTest(unittest.TestCase):
    MODULE_NAME = 'runners.tox_runner'
    PATH_IS_FILE_FUNCTION = f'{MODULE_NAME}.Path.is_file'
    COMPUTE_CHECKSUM_FUNCTION = f'{MODULE_NAME}.compute_checksum'
    READ_PREVIOUS_CHECKSUM_FUNCTION = f'{MODULE_NAME}.read_previous_checksum'
    SUBPROCESS_CALL = f'{MODULE_NAME}.subprocess'
    BUILTIN_OPEN_FUNCTION = '__builtin__.open' if PY2 else 'builtins.open'

    FILE_NAMES = ['a/test', 'b/file', 'requirements.txt']
    FAKEMODULE_PATH = os.path.join(CURRENT_PATH, "../resources/fakemodule.pyi")
    FAKEMODULE_IMPORTED_PATH = os.path.join(CURRENT_PATH, "../resources/fakemodule_imported.pyi")
    TEST_RESOURCES_FILE_NAMES = [FAKEMODULE_PATH, FAKEMODULE_IMPORTED_PATH]
    FILE_CONTENT = bytes("test\n end", 'utf-8')

    def test_fetching_python_files(self):
        folder = "test"
        with mock.patch('os.listdir') as mocked_listdir, mock.patch(f'{self.MODULE_NAME}.isfile') as mocked_isfile:
            mocked_listdir.return_value = ['folder1', 'folder2', 'file', 'file1.py', 'otherfile.cpp', 'file2.py']
            mocked_isfile.side_effect = [False, False, True, True, True, True]
            fns = tox_runner.fetch_python_file_names(folder)
            expected = ['test/file1.py', 'test/file2.py']
            self.assertListEqual(fns, expected)

    def test_fetching_resources(self):
        folder_name = 'test'
        extension = '.test'
        with mock.patch('os.walk') as mocked_walk:
            mocked_walk.return_value = [('folder2', '', [f'__init__{extension}', f'file1{extension}']),
                                        ('folder1', '', ['file', f'__init__{extension}']),
                                        ('folder3', '', ['otherfile.cpp', 'file2.testother', 'filetest'])]
            fns = tox_runner.fetch_resource_file_names(folder_name, extension)
            expected = [f'folder2/__init__{extension}', f'folder2/file1{extension}', f'folder1/__init__{extension}']
            mocked_walk.assert_called_once_with(folder_name)
            self.assertListEqual(fns, expected)

    def test_fetch_config_file_names(self):
        fns = tox_runner.fetch_config_file_names()
        expected = ['requirements.txt', 'tox.ini']
        self.assertListEqual(fns, expected)

    def test_fetch_source_file_names(self):
        folder = "test"
        with mock.patch(f'{self.MODULE_NAME}.fetch_python_file_names') as mock_fetch_python, \
                mock.patch(f'{self.MODULE_NAME}.fetch_resource_file_names') as mock_fetch_resource, \
                mock.patch(f'{self.MODULE_NAME}.fetch_config_file_names') as mock_fetch_config:
            mock_fetch_resource.return_value = ['a/1', 'b/2', 'b/4']
            mock_fetch_python.return_value = ['a/2', 'a/4', 'b/1', 'b/3']
            mock_fetch_config.return_value = ['z', '_1']
            fns = tox_runner.fetch_source_file_names(folder)
            self.assertListEqual(fns, ['_1', 'a/1', 'a/2', 'a/4', 'b/1', 'b/2', 'b/3', 'b/4', 'z'])
            mock_fetch_python.assert_called_with(folder)
            mock_fetch_config.assert_called()
            mock_fetch_resource.assert_called_with(tox_runner.RESOURCES_FOLDER_PATH, tox_runner.PYTHON_STUB_EXTENSION)

    def test_fetch_binary_file_name(self):
        with mock.patch(f'{self.MODULE_NAME}.fetch_resource_file_names') as mock_fetch_resource:
            mock_fetch_resource.return_value = ['b/1', 'a/2', 'a/1']
            fns = tox_runner.fetch_binary_file_names()
            self.assertListEqual(fns, ['a/1', 'a/2', 'b/1'])
            mock_fetch_resource.assert_called_with(tox_runner.BINARY_FOLDER_PATH, tox_runner.PROTOBUF_EXTENSION)

    def test_read_previous_checksum_non_existant_file(self):
        checksum_file = 'non_existant'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile:
            mocked_isfile.return_value = False
            assert tox_runner.read_previous_checksum(checksum_file) == (None, None)

    def test_read_previous_checksum_file_exists(self):
        source_checksum = '123'
        binary_checksum = '456'
        file_data = f"{source_checksum}\n{binary_checksum}"
        checksum_file = 'test_checksum'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile, \
                mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=file_data)) as mocked_open:
            mocked_isfile.return_value = True
            assert tox_runner.read_previous_checksum(checksum_file) == (source_checksum, binary_checksum)
            mocked_open.assert_called_with(checksum_file, 'r')

    def test_read_previous_checksum_file_missing_line(self):
        source_checksum = '123'
        checksum_file = 'test_checksum'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile, \
                mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=source_checksum)) as mocked_open:
            mocked_isfile.return_value = True
            assert tox_runner.read_previous_checksum(checksum_file) == (source_checksum, None)
            mocked_open.assert_called_with(checksum_file, 'r')

    def test_update_checksum(self):
        binary_file_names = ['test.protobuf', 'other.protobuf']
        source_checksum = '123'
        binaries_checksum = '456'
        checksums = [source_checksum, binaries_checksum]

        def feed_checksum(_fn, _f):
            return checksums.pop(0)

        with mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open()) as mocked_open, \
                mock.patch(f'{self.MODULE_NAME}.fetch_binary_file_names') as mock_binary_files, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum:
            mocked_checksum.side_effect = feed_checksum
            mock_files.return_value = self.FILE_NAMES
            mock_binary_files.return_value = binary_file_names
            tox_runner.update_checksum()
            mocked_file = mocked_open()
            mocked_open.assert_any_call(CHECKSUM_FILE, 'w')
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocked_checksum.assert_any_call(binary_file_names, tox_runner.read_file)
            mocked_file.writelines.assert_any_call([f"{source_checksum}\n", binaries_checksum])

    def test_normalized_text_files_rn(self):
        with mock.patch(f'{self.MODULE_NAME}.Path.read_text') as mock_read_text:
            mock_read_text.return_value = "\r\ntest\r\n end\r\n"
            text = tox_runner.normalize_text_files("test")
            assert text == self.FILE_CONTENT

    def test_normalized_text_files_r(self):
        with mock.patch(f'{self.MODULE_NAME}.Path.read_text') as mock_read_text:
            mock_read_text.return_value = "\rtest\r end\r"
            text = tox_runner.normalize_text_files("test")
            assert text == self.FILE_CONTENT

    def test_normalized_text_files(self):
        with mock.patch(f'{self.MODULE_NAME}.Path.read_text') as mock_read_text:
            mock_read_text.return_value = "\ntest\n end\n"
            text = tox_runner.normalize_text_files("test")
            assert text == self.FILE_CONTENT

    def test_read_file(self):
        file_bytes = bytes("\ntest end\n", 'utf-8')
        with mock.patch(f'{self.MODULE_NAME}.Path.read_bytes') as mock_read_bytes:
            mock_read_bytes.return_value = file_bytes
            text = tox_runner.read_file("test")
            assert text == file_bytes

    def test_compute_checksum(self):
        checksum1 = tox_runner.compute_checksum(self.TEST_RESOURCES_FILE_NAMES, tox_runner.normalize_text_files)
        checksum2 = tox_runner.compute_checksum(self.TEST_RESOURCES_FILE_NAMES, tox_runner.normalize_text_files)
        assert checksum1 == checksum2

    def test_compute_different_checksum(self):
        checksum1 = tox_runner.compute_checksum(self.TEST_RESOURCES_FILE_NAMES, tox_runner.normalize_text_files)
        checksum2 = tox_runner.compute_checksum([self.FAKEMODULE_IMPORTED_PATH], tox_runner.normalize_text_files)
        assert checksum1 != checksum2

    def test_tox_runner_unchanged_checksums(self):
        checksum = ('123', '456')
        computed_checksum = ['123', '456']

        def feed_checksum(_fn, _f):
            return computed_checksum.pop(0)

        with mock.patch(self.READ_PREVIOUS_CHECKSUM_FUNCTION) as mocked_previous_checksum, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(f'{self.MODULE_NAME}.fetch_binary_file_names') as mock_binary_files, \
                mock.patch(self.SUBPROCESS_CALL) as mocked_subprocess:
            mocked_previous_checksum.return_value = checksum
            mock_binary_files.return_value = self.FILE_NAMES
            mock_files.return_value = self.FILE_NAMES
            mocked_checksum.side_effect = feed_checksum
            tox_runner.main()
            mock_files.assert_called_once()
            mock_binary_files.assert_called_once()
            mocked_previous_checksum.assert_any_call(tox_runner.CHECKSUM_FILE)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.read_file)
            mocked_subprocess.run.assert_called_with(['python', '-m', 'tox', '-e', 'py39'], check=True)

    def test_tox_runner_different_binary_checksums(self):
        previous_checksum = '123'
        binaries_checksum = '456'
        checksums = [previous_checksum, binaries_checksum]
        previous_checksums = (previous_checksum, None)

        def feed_checksum(_fn, _f):
            return checksums.pop(0)

        with mock.patch(self.READ_PREVIOUS_CHECKSUM_FUNCTION) as mocked_previous_checksum, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(f'{self.MODULE_NAME}.fetch_binary_file_names') as mock_binary_files:
            mocked_previous_checksum.return_value = previous_checksums
            mock_binary_files.return_value = self.FILE_NAMES
            mock_files.return_value = self.FILE_NAMES
            mocked_checksum.side_effect = feed_checksum
            self.assertRaises(RuntimeError, tox_runner.main)
            mock_files.assert_called_once()
            mock_binary_files.assert_called_once()
            mocked_previous_checksum.assert_any_call(tox_runner.CHECKSUM_FILE)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.read_file)

    def test_tox_runner_modified_checksum(self):
        with mock.patch(self.READ_PREVIOUS_CHECKSUM_FUNCTION) as mocked_previous_checksum, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(self.SUBPROCESS_CALL) as mocked_subprocess:
            mock_files.return_value = self.FILE_NAMES
            mocked_previous_checksum.return_value = ('123', '456')
            mocked_checksum.return_value = ('789', '456')
            tox_runner.main()
            mocked_previous_checksum.assert_called_with(tox_runner.CHECKSUM_FILE)
            mocked_checksum.assert_called_with(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocked_subprocess.run.assert_called_with(['python', '-m', 'tox'], check=True)

    def test_skip_tests(self):
        checksum = ('123', '456')
        computed_checksum = ['123', '456']

        def feed_checksum(_fn, _f):
            return computed_checksum.pop(0)

        with mock.patch(self.READ_PREVIOUS_CHECKSUM_FUNCTION) as mocked_previous_checksum, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(f'{self.MODULE_NAME}.fetch_binary_file_names') as mock_binary_files, \
                mock.patch(self.SUBPROCESS_CALL) as mocked_subprocess:
            mocked_previous_checksum.return_value = checksum
            mock_binary_files.return_value = self.FILE_NAMES
            mock_files.return_value = self.FILE_NAMES
            mocked_checksum.side_effect = feed_checksum
            tox_runner.main(skip_tests=True)
            mock_files.assert_called_once()
            mock_binary_files.assert_called_once()
            mocked_previous_checksum.assert_any_call(tox_runner.CHECKSUM_FILE)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocked_checksum.assert_any_call(self.FILE_NAMES, tox_runner.read_file)
            mocked_subprocess.run.assert_not_called()


    def test_fail_fast(self):
        previous_checksum = '123'
        binaries_checksum = '456'
        checksums = [previous_checksum, binaries_checksum]
        previous_checksums = (None, None)

        def feed_checksum(_fn, _f):
            return checksums.pop(0)

        with mock.patch(self.READ_PREVIOUS_CHECKSUM_FUNCTION) as mocked_previous_checksum, \
                mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum, \
                mock.patch(f'{self.MODULE_NAME}.fetch_source_file_names') as mock_files, \
                mock.patch(f'{self.MODULE_NAME}.fetch_binary_file_names') as mock_binary_files, \
                mock.patch(self.SUBPROCESS_CALL) as mocked_subprocess, \
                self.assertRaises(RuntimeError) as error:
            mocked_previous_checksum.return_value = previous_checksums
            mock_binary_files.return_value = self.FILE_NAMES
            mock_files.return_value = self.FILE_NAMES
            mocked_checksum.side_effect = feed_checksum
            tox_runner.main(skip_tests=False, fail_fast=True)
            mocked_subprocess.run.assert_not_called()
        self.assertEqual(str(error.exception), 'INCONSISTENT SOURCES CHECKSUMS')
