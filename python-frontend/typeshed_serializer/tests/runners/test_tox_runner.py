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

import unittest
from unittest import mock
from unittest.mock import mock_open
import os
from contextlib import ExitStack


from runners import tox_runner
from runners.tox_runner import SERIALIZER_SOURCE_CHECKSUM_FILE

CURRENT_PATH = os.path.dirname(__file__)


class ToxRunnerTest(unittest.TestCase):
    MODULE_NAME = 'runners.tox_runner'
    PATH_IS_FILE_FUNCTION = f'{MODULE_NAME}.Path.is_file'
    COMPUTE_CHECKSUM_FUNCTION = f'{MODULE_NAME}.compute_checksum'
    READ_PREVIOUS_CHECKSUM_FUNCTION = f'{MODULE_NAME}.read_previous_checksum'
    SUBPROCESS_CALL = f'{MODULE_NAME}.subprocess'
    BUILTIN_OPEN_FUNCTION = 'builtins.open'

    FILE_NAMES = [os.path.join('a', 'test'), os.path.join('b', 'file'), 'requirements.txt']
    FAKEMODULE_PATH = os.path.join(CURRENT_PATH, "../resources/fakemodule.pyi")
    FAKEMODULE_IMPORTED_PATH = os.path.join(CURRENT_PATH, "../resources/fakemodule_imported.pyi")
    TEST_RESOURCES_FILE_NAMES = [FAKEMODULE_PATH, FAKEMODULE_IMPORTED_PATH]
    FILE_CONTENT = bytes("test\n end", 'utf-8')

    def _setup_basic_main_test(self, prev_checksum="123", curr_checksum="123", changed_serializers=None):
        """Set up a basic main() test with common mocks and configuration.

        This helper creates all the standard mocks needed to test tox_runner.main() function
        with different scenarios. It returns a context manager stack and a mock container object
        that provides clean access to individual mocks without requiring tuple unpacking.

        Args:
            prev_checksum (str): Previous SERIALIZER_SOURCE checksum value to simulate in files (default: "123")
            curr_checksum (str): Current SERIALIZER_SOURCE checksum value to simulate computation (default: "123")
            changed_serializers (list): List of serializer names that should appear changed (default: empty list)

        Returns:
            tuple: (ExitStack, MockContainer) where:
                - ExitStack: Context manager for all mocks
                - MockContainer: Object with named attributes for each mock

        Usage:
            stack, mocks = self._setup_basic_main_test(prev_checksum="old", curr_checksum="new")
            with stack:
                tox_runner.main()
                mocks.subprocess.run.assert_called_once()
        """
        stack = ExitStack()

        class MockContainer:
            """Container for organized access to mock objects.

            Attributes:
                prev_checksum: Mock for reading previous checksum from file
                checksum: Mock for computing current serializer code checksums
                files: Mock for fetching source file names
                check_changes: Mock for checking resource folder changes
                subprocess: Mock for subprocess.run calls
            """
            def __init__(self, test_instance):
                # Mock previous checksum reading (from saved files)
                self.prev_checksum = stack.enter_context(mock.patch(test_instance.READ_PREVIOUS_CHECKSUM_FUNCTION, return_value=prev_checksum))
                # Mock current checksum computation
                self.checksum = stack.enter_context(mock.patch(test_instance.COMPUTE_CHECKSUM_FUNCTION, return_value=curr_checksum))
                # Mock serializer source file discovery
                self.files = stack.enter_context(mock.patch(f"{test_instance.MODULE_NAME}.fetch_source_file_names", return_value=test_instance.FILE_NAMES))
                # Mock compute and compare checksums function call for serializer source
                serializer_changed = prev_checksum != curr_checksum
                self.compute_compare = stack.enter_context(mock.patch(f"{test_instance.MODULE_NAME}.compute_and_compare_checksums", return_value=serializer_changed))
                # Mock resource folder change detection
                self.check_changes = stack.enter_context(mock.patch(f"{test_instance.MODULE_NAME}.detect_required_serializations", return_value=changed_serializers or []))
                # Mock subprocess calls (tox execution)
                self.subprocess = stack.enter_context(mock.patch(test_instance.SUBPROCESS_CALL))
                self.update_folder_checksums_for_changed_serializers = stack.enter_context(mock.patch(f"{test_instance.MODULE_NAME}.update_folder_checksums_for_changed_serializers"))

        return stack, MockContainer(self)

    def test_fetching_python_files(self):
        folder = "test"
        with mock.patch('os.listdir') as mocked_listdir, mock.patch(f'{self.MODULE_NAME}.isfile') as mocked_isfile:
            mocked_listdir.return_value = ['folder1', 'folder2', 'file', 'file1.py', 'otherfile.cpp', 'file2.py']
            mocked_isfile.side_effect = [False, False, True, True, True, True]
            fns = tox_runner.fetch_python_file_names(folder)
            expected = [os.path.join("test", "file1.py"), os.path.join("test", "file2.py")]
            self.assertListEqual(fns, expected)

    def test_fetching_resources(self):
        folder_name = 'test'
        extension = '.test'
        with mock.patch('os.walk') as mocked_walk:
            mocked_walk.return_value = [('folder2', '', [f'__init__{extension}', f'file1{extension}']),
                                        ('folder1', '', ['file', f'__init__{extension}']),
                                        ('folder3', '', ['otherfile.cpp', 'file2.testother', 'filetest'])]
            fns = tox_runner.fetch_resource_file_names(folder_name, extension)
            expected = [os.path.join("folder2", f'__init__{extension}'), os.path.join("folder2", f'file1{extension}'), os.path.join("folder1", f'__init__{extension}')]
            mocked_walk.assert_called_once_with(folder_name)
            self.assertListEqual(fns, expected)

    def test_fetch_config_file_names(self):
        fns = tox_runner.fetch_config_file_names()
        self.assertEqual(len(fns), 2)
        self.assertTrue(fns[0].endswith('requirements.txt'))
        self.assertTrue(fns[1].endswith('tox.ini'))

    def test_fetch_source_file_names(self):
        folder = "test"
        with mock.patch(f'{self.MODULE_NAME}.fetch_python_file_names') as mock_fetch_python, \
                mock.patch(f'{self.MODULE_NAME}.fetch_config_file_names') as mock_fetch_config:
            mock_fetch_python.return_value = [os.path.join('a', '2'), os.path.join('a', '4'), os.path.join('b', '1'), os.path.join('b', '3')]
            mock_fetch_config.return_value = ['z', '_1']
            fns = tox_runner.fetch_source_file_names(folder)
            expected = ['_1', os.path.join('a', '2'), os.path.join('a', '4'), os.path.join('b', '1'), os.path.join('b', '3'), 'z']
            self.assertListEqual(fns, expected)
            mock_fetch_python.assert_called_with(folder)
            mock_fetch_config.assert_called()

    def test_read_previous_checksum_non_existant_file(self):
        checksum_file = 'non_existant'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile:
            mocked_isfile.return_value = False
            assert tox_runner.read_previous_checksum(checksum_file) == None

    def test_read_previous_checksum_file_exists(self):
        source_checksum = '123'
        file_data = f"{source_checksum}"
        checksum_file = 'test_checksum'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile, \
                mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=file_data)) as mocked_open:
            mocked_isfile.return_value = True
            assert tox_runner.read_previous_checksum(checksum_file) == source_checksum
            mocked_open.assert_called_with(checksum_file, 'r')

    def test_read_previous_checksum_empty(self):
        empty_source_checksum = f''
        checksum_file = 'test_checksum'
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile, \
                mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=empty_source_checksum)) as mocked_open:
            mocked_isfile.return_value = True
            assert tox_runner.read_previous_checksum(checksum_file) == None
            mocked_open.assert_called_with(checksum_file, 'r')

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
        # Test scenario: source code unchanged, no folder changes -> run tests only
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=[]
        )
        with stack:
            tox_runner.main()

            mocks.files.assert_called_once()
            mocks.prev_checksum.assert_any_call(tox_runner.SERIALIZER_SOURCE_CHECKSUM_FILE)
            mocks.checksum.assert_any_call(self.FILE_NAMES, tox_runner.normalize_text_files)
            mocks.subprocess.run.assert_called_with(
                ["tox", "-e", "py39"], check=True
            )

    def test_tox_runner_modified_checksum(self):
        # Test scenario: source code changed -> run full serialization
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="789",
            changed_serializers=[]
        )
        with stack:
            tox_runner.main()

            mocks.prev_checksum.assert_called_with(tox_runner.SERIALIZER_SOURCE_CHECKSUM_FILE)
            mocks.checksum.assert_called_with(self.FILE_NAMES, tox_runner.normalize_text_files)
            expected_calls = [
                mock.call(['tox', '-e', 'serialize'], check=True),
                mock.call(['tox', '-e', 'py39'], check=True)
            ]
            mocks.subprocess.run.assert_has_calls(expected_calls)

    def test_skip_tests(self):
        # Test scenario: no changes but skip_tests=True -> no subprocess calls at all
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=[]
        )
        with stack:
            tox_runner.main(skip_tests=True)

            mocks.files.assert_called_once()
            mocks.prev_checksum.assert_any_call(tox_runner.SERIALIZER_SOURCE_CHECKSUM_FILE)
            mocks.checksum.assert_any_call(
                self.FILE_NAMES, tox_runner.normalize_text_files
            )
            mocks.subprocess.run.assert_not_called()

    def test_fail_fast(self):
        # Test scenario: source code changed with fail_fast=True -> should raise RuntimeError
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="789",
            changed_serializers=[]
        )

        with (
            stack,
            self.assertRaises(RuntimeError) as error,
        ):
            tox_runner.main(skip_tests=False, fail_fast=True)
            mocks.subprocess.run.assert_not_called()
        self.assertEqual(str(error.exception), 'INCONSISTENT SOURCES CHECKSUMS')


    def test_write_folder_checksum(self):
        folder_name = "test_folder"
        source_checksum = "123abc"
        binary_checksum = "456def"

        with (
            mock.patch("os.makedirs") as mock_makedirs,
            mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open()) as mocked_open,
        ):
            tox_runner.write_folder_checksum(
                folder_name, source_checksum, binary_checksum
            )

            mock_makedirs.assert_called_once_with(
                tox_runner.CHECKSUMS_DIR, exist_ok=True
            )
            expected_file_path = os.path.join(
                tox_runner.CHECKSUMS_DIR, "test_folder.checksum"
            )
            mocked_open.assert_called_with(expected_file_path, "w")
            mocked_file = mocked_open()
            mocked_file.writelines.assert_called_once_with(
                [f"{source_checksum}\n", binary_checksum]
            )

    def test_read_previous_folder_checksum_non_existant_file(self):
        folder_name = "test_folder"
        with mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile:
            mocked_isfile.return_value = False
            result = tox_runner.read_previous_folder_checksum(folder_name)
            assert result == (None, None)

    def test_read_previous_folder_checksum_file_exists(self):
        folder_name = "test_folder"
        source_checksum = "123abc"
        binary_checksum = "456def"
        file_data = f"{source_checksum}\n{binary_checksum}"
        with (
            mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile,
            mock.patch(
                self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=file_data)
            ) as mocked_open,
        ):
            mocked_isfile.return_value = True
            result = tox_runner.read_previous_folder_checksum(folder_name)
            assert result == (source_checksum, binary_checksum)
            expected_file_path = os.path.join(
                tox_runner.CHECKSUMS_DIR, "test_folder.checksum"
            )
            mocked_open.assert_called_with(expected_file_path, "r")

    def test_read_previous_folder_checksum_missing_binary_line(self):
        folder_name = "test_folder"
        source_checksum = "123abc"
        file_data = source_checksum
        with (
            mock.patch(self.PATH_IS_FILE_FUNCTION) as mocked_isfile,
            mock.patch(
                self.BUILTIN_OPEN_FUNCTION, mock_open(read_data=file_data)
            ) as mocked_open,
        ):
            mocked_isfile.return_value = True
            result = tox_runner.read_previous_folder_checksum(folder_name)
            assert result == (source_checksum, None)

    def test_update_all_checksums_with_resources_folders(self):
        source_checksum = "123"
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "binary_folder": "custom_protobuf",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(self.BUILTIN_OPEN_FUNCTION, mock_open()) as mocked_open,
            mock.patch(f"{self.MODULE_NAME}.fetch_source_file_names") as mock_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(f"{self.MODULE_NAME}.RESOURCES_FOLDERS", mock_resources_folders),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.write_folder_checksum"
            ) as mock_write_checksum,
        ):
            mock_files.return_value = self.FILE_NAMES
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            mocked_checksum.side_effect = [source_checksum, "456", "789"]

            tox_runner.update_all_checksums()

            mocked_open.assert_called_with(SERIALIZER_SOURCE_CHECKSUM_FILE, "w")
            mock_write_checksum.assert_called_once_with("custom", "456", "789")

    def test_all_checksums_match_only_tests_launched(self):
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
                "binary_folder": "custom_protobuf",
            },
            "importer": {
                "serializer": "importer",
                "source_path": "importer",
                "binary_folder": "third_party_protobuf_mypy",
            },
        }

        # Test scenario: all checksums match -> run tests only
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=[]
        )
        with (
            stack,
            mock.patch(f"{self.MODULE_NAME}.RESOURCES_FOLDERS", mock_resources_folders),
            mock.patch(f"{self.MODULE_NAME}.write_folder_checksum") as mock_write_checksum,
        ):
            tox_runner.main()

            # Should only run tests
            mocks.subprocess.run.assert_called_with(
                ["tox", "-e", "py39"], check=True
            )

            # Should not update any checksums
            mock_write_checksum.assert_not_called()

    def test_detect_required_serializations_no_changes(self):
        # Test when all folder checksums match - no changes detected
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Folder has files and checksums match
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            current_source_checksum = "123"
            current_binary_checksum = "456"
            mocked_checksum.side_effect = [
                current_source_checksum,
                current_binary_checksum,
            ]
            mock_read_checksum.return_value = (
                current_source_checksum,
                current_binary_checksum,
            )

            result = tox_runner.detect_required_serializations()

            self.assertEqual(result, [])
            mock_resources_files.assert_called_once_with("custom")
            mock_binary_files.assert_called_once_with("custom")
            mock_read_checksum.assert_called_once_with("custom")

    def test_detect_required_serializations_source_changed(self):
        # Test when source checksum has changed
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Folder has files and source checksum changed
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            current_source_checksum = "123"
            current_binary_checksum = "456"
            previous_source_checksum = "999"  # Different
            previous_binary_checksum = "456"
            mocked_checksum.side_effect = [
                current_source_checksum,
                current_binary_checksum,
            ]
            mock_read_checksum.return_value = (
                previous_source_checksum,
                previous_binary_checksum,
            )

            result = tox_runner.detect_required_serializations()

            self.assertEqual(result, ["custom"])

    def test_detect_required_serializations_binary_changed(self):
        # Test when binary checksum has changed
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Folder has files and binary checksum changed
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            current_source_checksum = "123"
            current_binary_checksum = "456"
            previous_source_checksum = "123"
            previous_binary_checksum = "999"  # Different
            mocked_checksum.side_effect = [
                current_source_checksum,
                current_binary_checksum,
            ]
            mock_read_checksum.return_value = (
                previous_source_checksum,
                previous_binary_checksum,
            )

            result = tox_runner.detect_required_serializations()

            self.assertEqual(result, ["custom"])

    def test_detect_required_serializations_empty_folder_skipped(self):
        # Test that empty folders are skipped
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Empty folder - no files
            mock_resources_files.return_value = []

            result = tox_runner.detect_required_serializations()

            self.assertEqual(result, [])
            mock_resources_files.assert_called_once_with("custom")
            # Should not call other functions for empty folders
            mock_binary_files.assert_not_called()
            mock_read_checksum.assert_not_called()
            mocked_checksum.assert_not_called()

    def test_detect_required_serializations_fail_fast_binary_inconsistent(self):
        # Test fail-fast when binary checksum changed but source didn't
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Folder has files and binary changed but source didn't (inconsistent state)
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            current_source_checksum = "123"
            current_binary_checksum = "456"
            previous_source_checksum = "123"  # Same (unchanged)
            previous_binary_checksum = "999"  # Different (changed)
            mocked_checksum.side_effect = [
                current_source_checksum,
                current_binary_checksum,
            ]
            mock_read_checksum.return_value = (
                previous_source_checksum,
                previous_binary_checksum,
            )

            # Should raise RuntimeError with fail_fast=True
            with self.assertRaises(RuntimeError) as error:
                tox_runner.detect_required_serializations(fail_fast=True)

            self.assertEqual(
                str(error.exception), "INCONSISTENT RESOURCE FOLDER BINARY CHECKSUMS"
            )

    def test_detect_required_serializations_no_fail_fast_binary_inconsistent(self):
        # Test no fail-fast when binary checksum changed but source didn't (without fail_fast)
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
            }
        }

        with (
            mock.patch(
                f"{self.MODULE_NAME}.RESOURCES_FOLDERS",
                mock_resources_folders,
            ),
            mock.patch(
                f"{self.MODULE_NAME}.fetch_resources_subfolder_files"
            ) as mock_resources_files,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(
                f"{self.MODULE_NAME}.fetch_binary_files_for_folder"
            ) as mock_binary_files,
            mock.patch(
                f"{self.MODULE_NAME}.read_previous_folder_checksum"
            ) as mock_read_checksum,
        ):
            # Folder has files and binary changed but source didn't (inconsistent state)
            mock_resources_files.return_value = [os.path.join("custom", "test.pyi")]
            mock_binary_files.return_value = [os.path.join("custom_protobuf", "test.protobuf")]
            current_source_checksum = "123"
            current_binary_checksum = "456"
            previous_source_checksum = "123"  # Same (unchanged)
            previous_binary_checksum = "999"  # Different (changed)
            mocked_checksum.side_effect = [
                current_source_checksum,
                current_binary_checksum,
            ]
            mock_read_checksum.return_value = (
                previous_source_checksum,
                previous_binary_checksum,
            )

            # Should return changed serializer without fail_fast=False (default)
            result = tox_runner.detect_required_serializations(fail_fast=False)
            self.assertEqual(result, ["custom"])

    def test_main_fail_fast_resource_folder_binary_inconsistent(self):
        # Test main function fail-fast when resource folder has binary inconsistency
        source_checksum = "123"

        with (
            mock.patch(
                self.READ_PREVIOUS_CHECKSUM_FUNCTION
            ) as mocked_previous_checksum,
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mocked_checksum,
            mock.patch(f"{self.MODULE_NAME}.fetch_source_file_names") as mock_files,
            mock.patch(
                f"{self.MODULE_NAME}.detect_required_serializations"
            ) as mock_check_changes,
            mock.patch(self.SUBPROCESS_CALL) as mocked_subprocess,
        ):
            # Serializer source code unchanged (same checksum)
            mocked_previous_checksum.return_value = source_checksum
            mocked_checksum.return_value = source_checksum
            mock_files.return_value = self.FILE_NAMES

            # Resource folder check raises RuntimeError
            mock_check_changes.side_effect = RuntimeError(
                "INCONSISTENT RESOURCE FOLDER BINARY CHECKSUMS"
            )

            # Should propagate the RuntimeError
            with self.assertRaises(RuntimeError) as error:
                tox_runner.main(fail_fast=True)

            self.assertEqual(
                str(error.exception), "INCONSISTENT RESOURCE FOLDER BINARY CHECKSUMS"
            )
            # Should not call subprocess
            mocked_subprocess.run.assert_not_called()

    def test_dry_run_unchanged_checksums(self):
        # Test dry-run scenario: no changes -> should display test command only
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=[]
        )
        with stack:
            tox_runner.main(dry_run=True)

            # Should not call subprocess in dry run mode
            mocks.subprocess.run.assert_not_called()

    def test_dry_run_modified_checksum(self):
        # Test dry-run scenario: source changed -> should display full serialization command
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="789",
            changed_serializers=[]
        )
        with stack:
            tox_runner.main(dry_run=True)

            # Should not call subprocess in dry run mode
            mocks.subprocess.run.assert_not_called()

    def test_dry_run_selective_serialization(self):
        # Test dry-run scenario: folder changed -> should display selective serialization command
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=["custom"]
        )
        with stack:
            tox_runner.main(dry_run=True)

            # Should not call subprocess in dry run mode
            mocks.subprocess.run.assert_not_called()

    def test_selective_serialization_when_one_serializer_changed(self):
        # Test dry-run scenario: folder changed -> should display selective serialization command
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=["custom"]
        )
        with stack:
            tox_runner.main()

            expected_calls = [
                mock.call(['tox', '-e', 'selective-serialize', '--', 'custom'], check=True),
                mock.call(['tox', '-e', 'py39'], check=True)
            ]
            mocks.subprocess.run.assert_has_calls(expected_calls)
            mocks.update_folder_checksums_for_changed_serializers.assert_any_call(["custom"])


    def test_selective_serialization_when_two_serializer_changed(self):
        # Test dry-run scenario: folder changed -> should display selective serialization command
        stack, mocks = self._setup_basic_main_test(
            prev_checksum="123",
            curr_checksum="123",
            changed_serializers=["custom", "import"]
        )
        with stack:
            tox_runner.main()

            expected_calls = [
                mock.call(['tox', '-e', 'selective-serialize', '--', 'custom,import'], check=True),
                mock.call(['tox', '-e', 'py39'], check=True)
            ]
            mocks.subprocess.run.assert_has_calls(expected_calls)
            mocks.update_folder_checksums_for_changed_serializers.assert_any_call(["custom", "import"])


    def test_update_folder_checksums_for_changed_serializers(self):
        changed_serializers = ["custom", "stdlib"]
        mock_resources_folders = {
            "custom": {
                "serializer": "custom",
                "source_path": "custom",
                "binary_folder": "custom_protobuf",
            },
            "importer": {
                "serializer": "importer",
                "source_path": "importer",
                "binary_folder": "third_party_protobuf_mypy",
            },
            "typeshed_stdlib": {
                "serializer": "stdlib",
                "source_path": os.path.join("typeshed", "stdlib"),
                "binary_folder": "stdlib_protobuf",
            },
        }

        with (
            mock.patch(self.COMPUTE_CHECKSUM_FUNCTION) as mock_checksum,
            mock.patch(f"{self.MODULE_NAME}.RESOURCES_FOLDERS", mock_resources_folders),
            mock.patch(f"{self.MODULE_NAME}.fetch_resources_subfolder_files") as mock_resources_files,
            mock.patch(f"{self.MODULE_NAME}.fetch_binary_files_for_folder") as mock_binary_files,
            mock.patch(f"{self.MODULE_NAME}.write_folder_checksum") as mock_write_checksum,
        ):
            mock_resources_files.return_value = ["file1.py", "file2.py"]
            mock_binary_files.return_value = ["binary1.protobuf"]
            mock_checksum.side_effect = ["source123", "binary456", "source789", "binary012"]

            tox_runner.update_folder_checksums_for_changed_serializers(changed_serializers)

            # Should be called twice - once for "custom" and once for "typeshed_stdlib"
            assert mock_write_checksum.call_count == 2
            mock_write_checksum.assert_any_call("custom", "source123", "binary456")
            mock_write_checksum.assert_any_call("typeshed_stdlib", "source789", "binary012")

    def test_get_serialize_command_to_run_source_changed(self):
        # Test when source checksum has changed
        command = tox_runner.get_serialize_command_to_run("old_checksum", "new_checksum", [])
        expected = ['tox', '-e', 'serialize']
        self.assertEqual(command, expected)

    def test_get_serialize_command_to_run_serializers_changed(self):
        # Test when serializers have changed but source is the same
        command = tox_runner.get_serialize_command_to_run("same_checksum", "same_checksum", ["custom", "stdlib"])
        expected = ['tox', '-e', 'selective-serialize', '--', 'custom,stdlib']
        self.assertEqual(command, expected)

    def test_get_serialize_command_to_run_no_changes(self):
        # Test when nothing has changed
        command = tox_runner.get_serialize_command_to_run("same_checksum", "same_checksum", [])
        self.assertIsNone(command)

    def test_get_serialize_command_to_run_source_priority(self):
        # Test that source changes take priority over serializer changes
        command = tox_runner.get_serialize_command_to_run("old_checksum", "new_checksum", ["custom"])
        expected = ['tox', '-e', 'serialize']
        self.assertEqual(command, expected)

    def test_fetch_resources_subfolder_files(self):
        # Test fetching files from a subfolder
        with (
            mock.patch('os.path.exists', return_value=True),
            mock.patch('os.walk') as mock_walk
        ) :
            base_path = os.path.join('path', 'to', 'custom')
            mock_walk.return_value = [
                # hidden file and dir should not be returned
                (base_path, ['subdir', '.hidden-dir'], ['.hidden-file.py', 'file1.py', 'file2.pyi']),
                (os.path.join(base_path, 'subdir'), [], ['file3.py'])
            ]

            result = tox_runner.fetch_resources_subfolder_files('custom')
            expected = [
                os.path.join(base_path, 'file1.py'),
                os.path.join(base_path, 'file2.pyi'),
                os.path.join(base_path, 'subdir', 'file3.py')
            ]
            self.assertEqual(result, expected)

            # Assert there was one call and parameter ends with '../resources/custom'
            mock_walk.assert_called_once()
            call_args = mock_walk.call_args[0][0]
            expected_ending = os.path.join('resources', 'custom')
            self.assertTrue(call_args.endswith(expected_ending))

    def test_compute_and_compare_checksums_changed(self):
        # Test when checksums are different (files changed)
        files = ['file1.py', 'file2.py']
        with mock.patch('runners.tox_runner.compute_checksum', return_value='new_checksum'), \
             mock.patch('runners.tox_runner.logger') as mock_logger:
            
            result = tox_runner.compute_and_compare_checksums(
                'SOURCE', 'custom', files, tox_runner.normalize_text_files, 'old_checksum'
            )
            
            self.assertTrue(result)
            mock_logger.info.assert_any_call('SOURCE FILES CHANGED in folder: custom')
            mock_logger.info.assert_any_call('Previous checksum: old_checksum')
            mock_logger.info.assert_any_call('Current checksum: new_checksum')
            mock_logger.info.assert_any_call('Checksum is computed over 2 files')

    def test_compute_and_compare_checksums_unchanged(self):
        # Test when checksums are the same (files unchanged)
        files = ['file1.py', 'file2.py']
        with mock.patch('runners.tox_runner.compute_checksum', return_value='same_checksum'), \
             mock.patch('runners.tox_runner.logger') as mock_logger:
            
            result = tox_runner.compute_and_compare_checksums(
                'BINARY', 'importer', files, tox_runner.read_file, 'same_checksum'
            )
            
            self.assertFalse(result)
            mock_logger.info.assert_any_call('BINARY FILES UNCHANGED in folder: importer - Computed over 2 files - Checksum: same_checksum')

