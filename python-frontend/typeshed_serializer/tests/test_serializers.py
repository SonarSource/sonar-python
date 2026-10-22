#
# SonarQube Python Plugin
# Copyright (C) SonarSource Sàrl
# mailto:info AT sonarsource DOT com
#
# You can redistribute and/or modify this program under the terms of
# the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import logging
import os
import tempfile
from unittest.mock import Mock, patch

from serializer import symbols, serializers
from serializer.serializers import (
    CustomStubsSerializer,
    MicrosoftStubsSerializer,
    TypeshedSerializer,
    ImporterSerializer,
    SUPPORTED_PYTHON_VERSIONS,
    get_sources,
    walk_typeshed_stdlib,
)
from tests import conftest
from tests.conftest import MOCK_THIRD_PARTY_STUBS_LIST


def test_supported_python_versions():
    assert SUPPORTED_PYTHON_VERSIONS == ((3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15))


def test_build_mypy_model(typeshed_stdlib):
    assert typeshed_stdlib is not None


def test_serialize_typeshed_stdlib(typeshed_stdlib, caplog):
    serializers.walk_typeshed_stdlib = Mock(return_value=(typeshed_stdlib, set()))
    save_module_mock = Mock()
    with patch("serializer.serializers.symbols.save_module", save_module_mock):
        TypeshedSerializer().serialize()
        assert serializers.walk_typeshed_stdlib.call_count == 1
        assert symbols.save_module.call_count == len(typeshed_stdlib.files)

def test_build_for_every_python_version(typeshed_stdlib, caplog):
    serializers.walk_typeshed_stdlib= Mock(return_value=(typeshed_stdlib, set()))
    number_of_supported_python_versions = len(SUPPORTED_PYTHON_VERSIONS)
    with caplog.at_level(logging.INFO):
        modules = TypeshedSerializer().build_for_every_python_version()
        assert (
            serializers.walk_typeshed_stdlib.call_count
            == number_of_supported_python_versions
        )
        assert len(caplog.messages) == number_of_supported_python_versions
        assert "Building for python version " in caplog.text 
        assert len(modules.keys()) == number_of_supported_python_versions

def test_microsoft_stubs_serializer(microsoft_stubs):
    save_module_mock = Mock()
    with patch("serializer.serializers.symbols.save_module", save_module_mock):
        stubs_serializer = MicrosoftStubsSerializer()
        stubs_serializer.get_build_result = Mock(return_value=(microsoft_stubs, set()))
        stubs_serializer.serialize()
        assert stubs_serializer.get_build_result.call_count == 1
        assert symbols.save_module.call_count == 306


def test_custom_stubs_serializer(typeshed_custom_stubs):
    save_module_mock = Mock()
    with patch("serializer.serializers.symbols.save_module", save_module_mock):
        custom_stubs_serializer = CustomStubsSerializer()
        custom_stubs_serializer.get_build_result = Mock(
            return_value=(typeshed_custom_stubs, set())
        )
        custom_stubs_serializer.serialize()
        assert custom_stubs_serializer.get_build_result.call_count == 1
        expected_serialized_modules = sum(
            file != serializers.SONAR_CUSTOM_BASE_STUB_MODULE
            and typeshed_custom_stubs.files.get(file).path.startswith(
                custom_stubs_serializer.path
            )
            for file in typeshed_custom_stubs.files
        )
        assert symbols.save_module.call_count == expected_serialized_modules
        assert expected_serialized_modules > 0


def test_importer_serializer():
    build_mock = Mock()
    build_mock.BuildSource = Mock()
    build_mock.build = Mock()
    importer_serializer = ImporterSerializer()
    with patch("serializer.serializers.build", build_mock):
        importer_serializer.get_build_result()
        assert build_mock.BuildSource.call_count == 1
        assert build_mock.build.call_count == 1
    assert (
        importer_serializer.is_exception("sonar_third_party_libs", None, set()) is True
    )
    assert importer_serializer.is_exception("other", None, set()) is False


def test_third_parties_are_serialized_without_excluded_packages(typeshed_third_parties):
    stub_modules = set()
    excluded = ["pywin32"]
    stubs_without_excluded = [
        folder for folder in MOCK_THIRD_PARTY_STUBS_LIST if folder not in excluded
    ]
    for stub_folder in stubs_without_excluded:
        stub_path = os.path.join(
            conftest.CURRENT_PATH, conftest.MOCK_THIRD_PARTY_STUBS_LOCATION, stub_folder
        )
        _, dirs, files = os.walk(stub_path).__next__()
        stub_modules = stub_modules.union(
            [dir for dir in dirs if not dir.startswith("@")]
        )
        stub_modules = stub_modules.union(
            [file.replace(".pyi", "") for file in files if file.endswith(".pyi")]
        )
    for third_party_stub in stub_modules:
        assert third_party_stub in typeshed_third_parties
    for third_party_stub_name in typeshed_third_parties:
        assert "win32" not in third_party_stub_name
    assert len(typeshed_third_parties) == 12


def test_save_merged_symbols():
    save_module_mock = Mock()
    with patch("serializer.serializers.symbols.save_module", save_module_mock):
        ts_serializer = TypeshedSerializer()
        merged_module_symbol = symbols.MergedModuleSymbol("abc", {}, {}, {}, {})
        ts_serializer.get_merged_modules = Mock(
            return_value={"abc": merged_module_symbol}
        )
        ts_serializer.serialize_merged_modules()
        assert symbols.save_module.call_count == 1
        assert symbols.save_module.mock_calls[0].args[0] == merged_module_symbol

def test_get_sources():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_test_structure = {
            "module1.pyi": "# Module 1 stub",
            "module2.pyi": "# Module 2 stub", 
            "regular_file.py": "# Not a stub file",
            "text_file.txt": "# Text file",
            "subpackage/__init__.pyi": "# Subpackage init",
            "subpackage/submodule.pyi": "# Submodule stub",
            "subpackage/python2/old_module.pyi": "# Python 2 stub (should be excluded)",
            "excluded_module.pyi": "# Module to be excluded by filter",
        }
        
        for file_path, content in file_test_structure.items():
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
                 
        source_list, source_paths = get_sources(temp_dir)
        assert len(source_list) == 5
        assert len(source_paths) == 5

        module_names = [source.module for source in source_list]
        expected_modules = {
            "module1", "module2", "excluded_module", "subpackage", "subpackage.submodule"
        }
        assert set(module_names) == expected_modules
        
        def exclude_filter(root, file_str, file_path, fq_module_name):
            return "excluded" not in file_str and "excluded" not in fq_module_name
        
        source_list_filtered, source_paths_filtered = get_sources(temp_dir, exclude_filter)
        
        assert len(source_list_filtered) == 4
        assert len(source_paths_filtered) == 4
        
        filtered_module_names = [source.module for source in source_list_filtered]
        expected_filtered_modules = {
            "module1", "module2", "subpackage", "subpackage.submodule"
        }
        assert set(filtered_module_names) == expected_filtered_modules


def test_stdlib_module_availability_respects_package_and_module_ranges():
    versions = {"profiling": ((3, 15), None), "math": ((3, 0), None),
                "math.integer": ((3, 15), None), "lib2to3": ((3, 0), (3, 12))}
    available = serializers.is_stdlib_module_available
    assert available("math", (3, 14), versions)
    assert not available("math.integer", (3, 14), versions)
    assert available("math.integer", (3, 15), versions)
    assert not available("profiling.sampling", (3, 14), versions)
    assert available("profiling.sampling", (3, 15), versions)
    assert available("lib2to3.refactor", (3, 12), versions)
    assert not available("lib2to3.refactor", (3, 13), versions)


def test_merged_module_records_availability_even_if_all_members_are_unconditional(tmp_path):
    from serializer.symbols_merger import merge_modules

    for name in ("stable", "new"):
        (tmp_path / f"{name}.pyi").write_text("def available() -> int: ...\n", encoding="utf-8")
    versions = {"stable": ((3, 10), None), "new": ((3, 15), None)}
    models = {}
    with patch.object(serializers, "STDLIB_PATH", str(tmp_path)), \
         patch.object(serializers, "load_stdlib_py_versions", return_value=versions):
        for version in ((3, 14), (3, 15)):
            result, source_paths = walk_typeshed_stdlib(serializers.get_options(version))
            models[f"{version[0]}{version[1]}"] = {
                name: symbols.ModuleSymbol(module)
                for name, module in result.files.items()
                if module.path in source_paths
            }

    assert set(models["314"]) == {"stable"}
    assert set(models["315"]) == {"stable", "new"}
    merged = merge_modules({"stable", "new"}, models)
    new_module = merged["new"].to_proto()
    stable_module = merged["stable"].to_proto()
    assert list(new_module.valid_for) == ["315"]
    assert [(function.name, list(function.valid_for)) for function in new_module.functions] == [("available", ["315"])]
    assert not stable_module.valid_for
    assert [(function.name, list(function.valid_for)) for function in stable_module.functions] == [("available", [])]
