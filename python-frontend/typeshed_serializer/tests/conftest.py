#
# SonarQube Python Plugin
# Copyright (C) 2011-2022 SonarSource SA
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
from unittest.mock import Mock

import pytest
from mypy import build

from serializer import typeshed_serializer, symbols_merger


@pytest.fixture(scope="session")
def typeshed_stdlib():
    build_result, _ = typeshed_serializer.walk_typeshed_stdlib()
    return build_result


@pytest.fixture(scope="session")
def typeshed_custom_stubs():
    build_result, _ = typeshed_serializer.walk_custom_stubs()
    assert len(build_result.errors) == 0
    return build_result


@pytest.fixture(scope="session")
def fake_module_36_38():
    fake_module_path = os.path.join(os.path.dirname(__file__), "resources/fakemodule.pyi")
    typeshed_serializer.load_single_module = Mock(return_value=build.BuildSource(fake_module_path, "fakemodule"))
    fake_module_36 = typeshed_serializer.build_single_module('fakemodule', python_version=(3, 6))
    fake_module_38 = typeshed_serializer.build_single_module('fakemodule', python_version=(3, 8))
    return [fake_module_36, fake_module_38]


@pytest.fixture(scope="session")
def typeshed_third_parties():
    return symbols_merger.merge_multiple_python_versions(is_third_parties=True)
