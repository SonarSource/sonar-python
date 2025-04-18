/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2.typeshed;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.types.protobuf.SymbolsProtos;

class TypeShedUtilsTest {

  @Test
  void normalizedFqnTest() {
    Assertions.assertThat(TypeShedUtils.normalizedFqn("builtins.int")).isEqualTo("int");
    Assertions.assertThat(TypeShedUtils.normalizedFqn("something.else")).isEqualTo("something.else");
    Assertions.assertThat(TypeShedUtils.normalizedFqn("")).isNull();
  }

  @Test
  void getTypesNormalizedFqnBasicTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("builtins.int")
        .build()))
      .isEqualTo("int");
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("something.else")
        .build()))
      .isEqualTo("something.else");
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("")
        .build()))
      .isNull();
  }

  @Test
  void getTypesNormalizedFqnBasicForInstanceKindTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.INSTANCE)
        .setFullyQualifiedName("builtins.int")
        .build()))
      .isEqualTo("int");

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.INSTANCE)
        .setFullyQualifiedName("typing._SpecialForm")
        .build()))
      .isEqualTo("typing._SpecialForm");

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.INSTANCE)
        .setFullyQualifiedName("")
        .build()))
      .isNull();
  }

  @Test
  void getTypesNormalizedFqnForBuiltinsKindsTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPE)
        .build()))
      .isEqualTo("type");
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TUPLE)
        .build()))
      .isEqualTo("tuple");
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.NONE)
        .build()))
      .isEqualTo("NoneType");
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPED_DICT)
        .build()))
      .isEqualTo("dict");
  }

  @Test
  void getTypesNormalizedFqnForTypeAliasKindTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPE_ALIAS)
        .addArgs(
          SymbolsProtos.Type.newBuilder()
            .setKind(SymbolsProtos.TypeKind.INSTANCE)
            .setFullyQualifiedName("builtins.int")
            .build()
        )
        .build()))
      .isEqualTo("int");
  }

  @Test
  void getTypesNormalizedFqnForTypeVarKindTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPE_VAR)
        .setFullyQualifiedName("builtins.int")
        .build()))
      .isEqualTo("int");

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPE_VAR)
        .setFullyQualifiedName("builtins.object")
        .build()))
      .isNull();

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.TYPE_VAR)
        .setFullyQualifiedName("_ctypes._CanCastTo")
        .build()))
      .isNull();
  }

  @Test
  void getTypesNormalizedFqnForUnsupportedKindsTest() {
    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.CALLABLE)
        .build()))
      .isNull();

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.UNION)
        .build()))
      .isNull();

    Assertions.assertThat(TypeShedUtils.getTypesNormalizedFqn(SymbolsProtos.Type.newBuilder()
        .setKind(SymbolsProtos.TypeKind.LITERAL)
        .build()))
      .isNull();
  }


}
