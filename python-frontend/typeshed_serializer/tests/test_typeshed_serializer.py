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

from unittest.mock import Mock

from serializer import typeshed_serializer

from serializer.serializers import CustomStubsSerializer, MicrosoftStubsSerializer, TypeshedSerializer, ImporterSerializer


def test_typeshed_serializer():
    TypeshedSerializer.serialize_merged_modules = Mock()
    CustomStubsSerializer.serialize = Mock()
    ImporterSerializer.serialize = Mock()
    MicrosoftStubsSerializer.serialize = Mock()
    typeshed_serializer.main()
    assert TypeshedSerializer.serialize_merged_modules.call_count == 2
    assert CustomStubsSerializer.serialize.call_count == 1
    assert ImporterSerializer.serialize.call_count == 1
    assert MicrosoftStubsSerializer.serialize.call_count == 1
