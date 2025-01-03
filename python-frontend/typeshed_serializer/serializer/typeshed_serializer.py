#
# SonarQube Python Plugin
# Copyright (C) 2011-2025 SonarSource SA
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

from serializer.serializers import MicrosoftStubsSerializer, TypeshedSerializer, CustomStubsSerializer, ImporterSerializer


def main():
    TypeshedSerializer().serialize_merged_modules()
    TypeshedSerializer(is_third_parties=True).serialize_merged_modules()
    CustomStubsSerializer().serialize()
    ImporterSerializer().serialize()
    MicrosoftStubsSerializer().serialize()


if __name__ == '__main__':
    main()
