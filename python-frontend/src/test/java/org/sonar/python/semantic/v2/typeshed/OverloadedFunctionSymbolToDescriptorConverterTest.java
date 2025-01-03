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

import java.util.function.Function;
import java.util.stream.Collectors;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

class OverloadedFunctionSymbolToDescriptorConverterTest {


  @Test
  void test() {
    var symbol = SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
      .setName("something")
      .setFullname("module.something")
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("f1")
        .setIsStatic(true)
        .build())
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("f2")
        .build())
      .build();
    var converter = new OverloadedFunctionSymbolToDescriptorConverter(new FunctionSymbolToDescriptorConverter());

    var descriptor = converter.convert(symbol, true);
    Assertions.assertThat(descriptor.name()).isEqualTo("something");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module.something");
    Assertions.assertThat(descriptor.alternatives()).hasSize(2);

    var candidatesByName = descriptor.alternatives()
      .stream()
      .collect(Collectors.toMap(Descriptor::name, Function.identity()));

    var f1 = (FunctionDescriptor) candidatesByName.get("f1");
    var f2 = (FunctionDescriptor) candidatesByName.get("f2");
    Assertions.assertThat(f1.isInstanceMethod()).isFalse();
    Assertions.assertThat(f2.isInstanceMethod()).isTrue();
  }

  @Test
  void builtinTest() {
    var symbol = SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
      .setName("int")
      .setFullname("builtins.int")
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder().build())
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder().build())
      .build();
    var converter = new OverloadedFunctionSymbolToDescriptorConverter(new FunctionSymbolToDescriptorConverter());

    var descriptor = converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("int");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("int");
    Assertions.assertThat(descriptor.alternatives()).hasSize(2);
  }

  @Test
  void singleCandidateExceptionTest() {
    var symbol = SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder().build())
      .build();
    var converter = new OverloadedFunctionSymbolToDescriptorConverter(new FunctionSymbolToDescriptorConverter());

    Assertions.assertThatThrownBy(() -> converter.convert(symbol))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Overloaded function symbols should have at least two definitions.");
  }

}
