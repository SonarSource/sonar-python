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

class ParameterSymbolToDescriptorConverterTest {
  
  @Test
  void test1() {
    var converter = new ParameterSymbolToDescriptorConverter();
    var ps1 = createParameterSymbol("p1",
      false,
      SymbolsProtos.ParameterKind.KEYWORD_ONLY,
      null);
    var pd1 = converter.convert(ps1);
    Assertions.assertThat(pd1.name()).isEqualTo("p1");
    Assertions.assertThat(pd1.annotatedType()).isNull();
    Assertions.assertThat(pd1.hasDefaultValue()).isFalse();
    Assertions.assertThat(pd1.isKeywordOnly()).isTrue();
    Assertions.assertThat(pd1.isPositionalOnly()).isFalse();
    Assertions.assertThat(pd1.isPositionalVariadic()).isFalse();
    Assertions.assertThat(pd1.isKeywordVariadic()).isFalse();
  }

  @Test
  void test2() {
    var converter = new ParameterSymbolToDescriptorConverter();
    var ps2 = createParameterSymbol("p2",
      false,
      SymbolsProtos.ParameterKind.KEYWORD_ONLY,
      "builtins.int");
    var pd2 = converter.convert(ps2);
    Assertions.assertThat(pd2.name()).isEqualTo("p2");
    Assertions.assertThat(pd2.annotatedType()).isEqualTo("int");
    Assertions.assertThat(pd2.hasDefaultValue()).isFalse();
    Assertions.assertThat(pd2.isKeywordOnly()).isTrue();
    Assertions.assertThat(pd2.isPositionalOnly()).isFalse();
    Assertions.assertThat(pd2.isPositionalVariadic()).isFalse();
    Assertions.assertThat(pd2.isKeywordVariadic()).isFalse();
  }

  @Test
  void test3() {
    var converter = new ParameterSymbolToDescriptorConverter();
    var ps3 = createParameterSymbol("p3",
      false,
      SymbolsProtos.ParameterKind.POSITIONAL_ONLY,
      "builtins.str");
    var pd3 = converter.convert(ps3);
    Assertions.assertThat(pd3.name()).isEqualTo("p3");
    Assertions.assertThat(pd3.annotatedType()).isEqualTo("str");
    Assertions.assertThat(pd3.hasDefaultValue()).isFalse();
    Assertions.assertThat(pd3.isKeywordOnly()).isFalse();
    Assertions.assertThat(pd3.isPositionalOnly()).isTrue();
    Assertions.assertThat(pd3.isPositionalVariadic()).isFalse();
    Assertions.assertThat(pd3.isKeywordVariadic()).isFalse();
  }

  @Test
  void test4() {
    var converter = new ParameterSymbolToDescriptorConverter();
    var ps4 = createParameterSymbol("p4",
      false,
      SymbolsProtos.ParameterKind.VAR_POSITIONAL,
      "builtins.int");
    var pd4 = converter.convert(ps4);
    Assertions.assertThat(pd4.name()).isEqualTo("p4");
    Assertions.assertThat(pd4.annotatedType()).isEqualTo("int");
    Assertions.assertThat(pd4.hasDefaultValue()).isFalse();
    Assertions.assertThat(pd4.isKeywordOnly()).isFalse();
    Assertions.assertThat(pd4.isPositionalOnly()).isFalse();
    Assertions.assertThat(pd4.isPositionalVariadic()).isTrue();
    Assertions.assertThat(pd4.isKeywordVariadic()).isFalse();
  }

  @Test
  void test5() {
    var converter = new ParameterSymbolToDescriptorConverter();
    var ps5 = createParameterSymbol("p5",
      true,
      SymbolsProtos.ParameterKind.VAR_KEYWORD,
      "builtins.int");
    var pd5 = converter.convert(ps5);
    Assertions.assertThat(pd5.name()).isEqualTo("p5");
    Assertions.assertThat(pd5.annotatedType()).isEqualTo("int");
    Assertions.assertThat(pd5.hasDefaultValue()).isTrue();
    Assertions.assertThat(pd5.isKeywordOnly()).isFalse();
    Assertions.assertThat(pd5.isPositionalOnly()).isFalse();
    Assertions.assertThat(pd5.isPositionalVariadic()).isFalse();
    Assertions.assertThat(pd5.isKeywordVariadic()).isTrue();
  }

  private static SymbolsProtos.ParameterSymbol createParameterSymbol(String name,
    boolean hasDefault,
    SymbolsProtos.ParameterKind kind,
    String type) {
    var builder = SymbolsProtos.ParameterSymbol.newBuilder()
      .setName(name)
      .setHasDefault(hasDefault)
      .setKind(kind);

    if (type != null) {
      builder.setTypeAnnotation(SymbolsProtos.Type.newBuilder().setFullyQualifiedName(type).build());
    }

    return builder.build();
  }

}
