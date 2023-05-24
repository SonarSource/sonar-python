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

import os
from unittest import mock

import pytest
from mypy import build

from serializer import serializers
from serializer.serializers import CustomStubsSerializer, TypeshedSerializer, get_options

CURRENT_PATH = os.path.dirname(__file__)
MOCK_THIRD_PARTY_STUBS_LOCATION = "resources/mock_third_parties"
MOCK_THIRD_PARTY_STUBS_LIST = os.listdir(os.path.join(CURRENT_PATH, MOCK_THIRD_PARTY_STUBS_LOCATION))


@pytest.fixture(scope="session")
def typeshed_stdlib():
    with mock.patch('serializer.serializers.CURRENT_PATH', CURRENT_PATH):
        serializers.STDLIB_PATH = "resources/mock_stdlib"
        serializers.CURRENT_PATH = CURRENT_PATH
        build_result, _ = serializers.walk_typeshed_stdlib()
    return build_result


@pytest.fixture(scope="session")
def typeshed_custom_stubs():
    build_result, _ = CustomStubsSerializer().get_build_result()
    assert len(build_result.errors) == 0
    return build_result


@pytest.fixture(scope="session")
def fake_module_36_38():
    modules = {
        "fakemodule": os.path.join(CURRENT_PATH, "resources/fakemodule.pyi"),
        "fakemodule_imported": os.path.join(CURRENT_PATH, "resources/fakemodule_imported.pyi")
    }
    model_36 = build_modules(modules, python_version=(3, 6))
    model_38 = build_modules(modules, python_version=(3, 8))
    return [model_36.get("fakemodule"), model_38.get("fakemodule")]


@pytest.fixture(scope="session")
def typeshed_third_parties():
    serializers.THIRD_PARTIES_STUBS = MOCK_THIRD_PARTY_STUBS_LIST
    return TypeshedSerializer(is_third_parties=True).get_merged_modules()


def build_modules(modules: dict[str, str], python_version=(3, 8)):
    opt = get_options(python_version)
    module_sources = []
    for module_fqn in modules.keys():
        module_sources.append(load_single_module(modules.get(module_fqn), module_fqn))
    build_result = build.build(module_sources, opt)
    return build_result.files


def load_single_module(module_path: str, module_fqn):
    module_source = build.BuildSource(module_path, module_fqn)
    return module_source
