/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.semantic.v2.typeshed;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.types.protobuf.SymbolsProtos;

class OverloadedFunctionSymbolToDescriptorConverterTest {


  @Test
  void test() {
    var symbol = SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
      .setName("something")
      .setFullname("module.something")
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder().build())
      .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder().build())
      .build();
    var converter = new OverloadedFunctionSymbolToDescriptorConverter(new FunctionSymbolToDescriptorConverter());

    var descriptor = converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("something");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module.something");
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
