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

import sys
from typing import List, Optional
from serializer.serializers import (
    MicrosoftStubsSerializer,
    TypeshedSerializer,
    CustomStubsSerializer,
    ImporterSerializer,
)


def main(serializers: Optional[List[str]] = None):
    """
    Run typeshed serialization for specified serializers.

    Args:
        serializers: List of serializer names to run. If None, run all serializers.
                    Valid names: 'stdlib', 'third_party', 'custom', 'importer', 'microsoft'
    """
    if serializers is None:
        serializers = ["stdlib", "third_party", "custom", "importer", "microsoft"]

    if "stdlib" in serializers:
        TypeshedSerializer().serialize_merged_modules()

    if "third_party" in serializers:
        TypeshedSerializer(is_third_parties=True).serialize_merged_modules()

    if "custom" in serializers:
        CustomStubsSerializer().serialize()

    if "importer" in serializers:
        ImporterSerializer().serialize()

    if "microsoft" in serializers:
        MicrosoftStubsSerializer().serialize()


if __name__ == "__main__":
    # Parse command line arguments for selective execution
    if len(sys.argv) > 1:
        serializers = sys.argv[1].split(",")
        main(serializers)
    else:
        main()
