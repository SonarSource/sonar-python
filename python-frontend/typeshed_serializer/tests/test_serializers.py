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

import logging
import os
from unittest.mock import Mock, patch

from serializer import symbols, serializers
from serializer.serializers import (
    CustomStubsSerializer,
    MicrosoftStubsSerializer,
    TypeshedSerializer,
    ImporterSerializer,
)
from tests import conftest
from tests.conftest import MOCK_THIRD_PARTY_STUBS_LIST


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
    NUMBER_OF_PYTHON_VERSION_SUPPORTED = 6
    with caplog.at_level(logging.INFO):
        modules = TypeshedSerializer().build_for_every_python_version()
        assert serializers.walk_typeshed_stdlib.call_count == NUMBER_OF_PYTHON_VERSION_SUPPORTED
        assert len(caplog.messages) == NUMBER_OF_PYTHON_VERSION_SUPPORTED
        assert "Building for python version " in caplog.text 
        assert len(modules.keys()) == NUMBER_OF_PYTHON_VERSION_SUPPORTED

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
        # Not every files from "typeshed_custom_stubs" build are serialized, as some are builtins
        assert symbols.save_module.call_count == 146


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
