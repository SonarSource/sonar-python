/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

class FunctionSymbolToDescriptorConverterTest {

  @Test
  void test() {
    var converter = new FunctionSymbolToDescriptorConverter();

    var symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setName("foo")
      .setFullyQualifiedName("module.foo")
      .setIsAsynchronous(false)
      .setIsClassMethod(false)
      .setIsStatic(false)
      .setHasDecorators(true)
      .addResolvedDecoratorNames("decorator")
      .setReturnAnnotation(
        SymbolsProtos.Type.newBuilder()
          .setFullyQualifiedName("builtins.int")
          .build()
      )
      .addParameters(
        SymbolsProtos.ParameterSymbol.newBuilder()
          .setName("p1")
          .build()
      )
      .build();
    var descriptor = converter.convert(symbol);

    Assertions.assertThat(descriptor.name()).isEqualTo("foo");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module.foo");
    Assertions.assertThat(descriptor.isAsynchronous()).isFalse();
    Assertions.assertThat(descriptor.isInstanceMethod()).isFalse();
    Assertions.assertThat(descriptor.hasDecorators()).isTrue();
    Assertions.assertThat(descriptor.annotatedReturnTypeName()).isEqualTo("int");
    Assertions.assertThat(descriptor.parameters()).hasSize(1);
    Assertions.assertThat(descriptor.parameters().get(0).name()).isEqualTo("p1");
  }

  @Test
  void instanceMethodTest() {
    var converter = new FunctionSymbolToDescriptorConverter();

    var symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setIsClassMethod(false)
      .setIsStatic(false)
      .build();
    var descriptor = converter.convert(symbol, true);
    Assertions.assertThat(descriptor.isInstanceMethod()).isTrue();

    symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setIsClassMethod(true)
      .setIsStatic(false)
      .build();
    descriptor = converter.convert(symbol, true);
    Assertions.assertThat(descriptor.isInstanceMethod()).isFalse();

    symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setIsClassMethod(false)
      .setIsStatic(true)
      .build();
    descriptor = converter.convert(symbol, true);
    Assertions.assertThat(descriptor.isInstanceMethod()).isFalse();

    symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setIsClassMethod(true)
      .setIsStatic(true)
      .build();
    descriptor = converter.convert(symbol, true);
    Assertions.assertThat(descriptor.isInstanceMethod()).isFalse();

    symbol = SymbolsProtos.FunctionSymbol.newBuilder()
      .setIsClassMethod(true)
      .setIsStatic(true)
      .build();
    descriptor = converter.convert(symbol, false);
    Assertions.assertThat(descriptor.isInstanceMethod()).isFalse();
  }

}
