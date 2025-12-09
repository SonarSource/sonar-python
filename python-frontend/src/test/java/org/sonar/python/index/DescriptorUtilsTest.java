/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.index;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.sonar.python.types.protobuf.SymbolsProtos;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class DescriptorUtilsTest {

  @ParameterizedTest
  @EnumSource(SymbolsProtos.TypeKind.class)
  void testSymbolTypeKindToTypeAnnotationKind(SymbolsProtos.TypeKind protoKind) {
    TypeAnnotationDescriptor.TypeKind typeKind = DescriptorUtils.symbolTypeKindToTypeAnnotationKind(protoKind);
    if (protoKind == SymbolsProtos.TypeKind.UNRECOGNIZED) {
      assertNull(typeKind);
    } else {
      assertEquals(protoKind.name(), typeKind.name());
    }
  }

  @ParameterizedTest
  @EnumSource(TypeAnnotationDescriptor.TypeKind.class)
  void testTypeAnnotationKindToSymbolKind(TypeAnnotationDescriptor.TypeKind typeKind) {
    SymbolsProtos.TypeKind protoKind = DescriptorUtils.typeAnnotationKindToSymbolKind(typeKind);
    assertEquals(typeKind.name(), protoKind.name());
  }
}

