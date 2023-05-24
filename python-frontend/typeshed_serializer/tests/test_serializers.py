#
# SonarQube Python Plugin
# Copyright (C) 2011-2023 SonarSource SA
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

from unittest import mock
from unittest.mock import Mock, patch

from serializer import symbols, serializers
import os

from serializer.serializers import CustomStubsSerializer, TypeshedSerializer
from tests import conftest
from tests.conftest import MOCK_THIRD_PARTY_STUBS_LIST


def test_build_mypy_model(typeshed_stdlib):
    assert typeshed_stdlib is not None


def test_serialize_typeshed_stdlib(typeshed_stdlib):
    serializers.walk_typeshed_stdlib = Mock(return_value=(typeshed_stdlib, set()))
    save_module_mock = Mock()
    with patch('serializer.serializers.symbols.save_module', save_module_mock):
        TypeshedSerializer().serialize()
        assert serializers.walk_typeshed_stdlib.call_count == 1
        assert symbols.save_module.call_count == len(typeshed_stdlib.files)


def test_custom_stubs_serializer(typeshed_custom_stubs):
    save_module_mock = Mock()
    with patch('serializer.serializers.symbols.save_module', save_module_mock):
        custom_stubs_serializer = CustomStubsSerializer()
        custom_stubs_serializer.get_build_result = Mock(return_value=(typeshed_custom_stubs, set()))
        custom_stubs_serializer.serialize()
        assert custom_stubs_serializer.get_build_result.call_count == 1
        # Not every files from "typeshed_custom_stubs" build are serialized, as some are builtins
        assert symbols.save_module.call_count == 79


def test_all_third_parties_are_serialized(typeshed_third_parties):
    stub_modules = set()
    for stub_folder in MOCK_THIRD_PARTY_STUBS_LIST:
        stub_path = os.path.join(conftest.CURRENT_PATH, conftest.MOCK_THIRD_PARTY_STUBS_LOCATION, stub_folder)
        _, dirs, files = os.walk(stub_path).__next__()
        stub_modules = stub_modules.union([dir for dir in dirs if not dir.startswith("@")])
        stub_modules = stub_modules.union([file.replace(".pyi", "") for file in files if file.endswith(".pyi")])
    for third_party_stub in stub_modules:
        assert third_party_stub in typeshed_third_parties


def test_save_merged_symbols():
    save_module_mock = Mock()
    with patch('serializer.serializers.symbols.save_module', save_module_mock):
        ts_serializer = TypeshedSerializer()
        merged_module_symbol = symbols.MergedModuleSymbol('abc', {}, {}, {}, {})
        ts_serializer.get_merged_modules = Mock(return_value={'abc': merged_module_symbol})
        ts_serializer.serialize_merged_modules()
        assert symbols.save_module.call_count == 1
        assert symbols.save_module.mock_calls[0].args[0] == merged_module_symbol
