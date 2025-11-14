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

from unittest.mock import Mock
from serializer import typeshed_serializer
from serializer.serializers import CustomStubsSerializer, MicrosoftStubsSerializer, TypeshedSerializer, ImporterSerializer


def test_main_with_all_serializers():
    """Test main function with default parameters (all serializers)"""
    TypeshedSerializer.serialize_merged_modules = Mock()
    CustomStubsSerializer.serialize = Mock()
    ImporterSerializer.serialize = Mock()
    MicrosoftStubsSerializer.serialize = Mock()

    typeshed_serializer.main()

    assert TypeshedSerializer.serialize_merged_modules.call_count == 2  # stdlib + third_party
    assert CustomStubsSerializer.serialize.call_count == 1
    assert ImporterSerializer.serialize.call_count == 1
    assert MicrosoftStubsSerializer.serialize.call_count == 1


def test_main_with_specific_serializers():
    """Test main function with specific serializers"""
    TypeshedSerializer.serialize_merged_modules = Mock()
    CustomStubsSerializer.serialize = Mock()
    ImporterSerializer.serialize = Mock()
    MicrosoftStubsSerializer.serialize = Mock()

    # Test with only stdlib and custom
    typeshed_serializer.main(["stdlib", "custom"])

    assert TypeshedSerializer.serialize_merged_modules.call_count == 1  # only stdlib
    assert CustomStubsSerializer.serialize.call_count == 1
    assert ImporterSerializer.serialize.call_count == 0
    assert MicrosoftStubsSerializer.serialize.call_count == 0


def test_main_with_third_party_serializer():
    """Test main function with third_party serializer"""
    TypeshedSerializer.serialize_merged_modules = Mock()
    CustomStubsSerializer.serialize = Mock()
    ImporterSerializer.serialize = Mock()
    MicrosoftStubsSerializer.serialize = Mock()

    typeshed_serializer.main(["third_party"])

    # TypeshedSerializer should be called with is_third_parties=True
    assert TypeshedSerializer.serialize_merged_modules.call_count == 1
    assert CustomStubsSerializer.serialize.call_count == 0
    assert ImporterSerializer.serialize.call_count == 0
    assert MicrosoftStubsSerializer.serialize.call_count == 0


def test_main_with_empty_serializers_list():
    """Test main function with empty serializers list"""
    TypeshedSerializer.serialize_merged_modules = Mock()
    CustomStubsSerializer.serialize = Mock()
    ImporterSerializer.serialize = Mock()
    MicrosoftStubsSerializer.serialize = Mock()

    typeshed_serializer.main([])

    # No serializers should be called
    assert TypeshedSerializer.serialize_merged_modules.call_count == 0
    assert CustomStubsSerializer.serialize.call_count == 0
    assert ImporterSerializer.serialize.call_count == 0
    assert MicrosoftStubsSerializer.serialize.call_count == 0
