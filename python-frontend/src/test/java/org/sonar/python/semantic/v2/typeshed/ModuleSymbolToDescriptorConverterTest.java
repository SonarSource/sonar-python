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
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

class ModuleSymbolToDescriptorConverterTest {
  @Test
  void test() {
    var converter = new ModuleSymbolToDescriptorConverter(PythonVersionUtils.allVersions());
    var symbol = SymbolsProtos.ModuleSymbol.newBuilder()
      .setFullyQualifiedName("module")
      .addVars(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v1")
        .setFullyQualifiedName("module.v1")
        .build())
      .addVars(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v2")
        .setFullyQualifiedName("module.v2")
        .build())
      .addVars(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v2")
        .setFullyQualifiedName("module.v2")
        .build())
      .addFunctions(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("foo")
        .setFullyQualifiedName("module.foo")
        .build())
      .addOverloadedFunctions(SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
        .setName("overloaded_foo")
        .setFullname("module.overloaded_foo")
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo")
          .setFullyQualifiedName("module.overloaded_foo")
          .build())
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo")
          .setFullyQualifiedName("module.overloaded_foo")
          .build())
        .build()
      )
      .addClasses(
        SymbolsProtos.ClassSymbol.newBuilder()
          .setName("MyClass")
          .setFullyQualifiedName("module.MyClass")
          .build()
      )
      .build();
    var descriptor = converter.convert(symbol);
    Assertions.assertThat(descriptor).isNotNull();
    Assertions.assertThat(descriptor.name()).isEqualTo("module");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module");
    Assertions.assertThat(descriptor.members().get("v1")).isInstanceOf(VariableDescriptor.class);
    Assertions.assertThat(descriptor.members().get("v2")).isInstanceOf(AmbiguousDescriptor.class);
    Assertions.assertThat(descriptor.members().get("foo")).isInstanceOf(FunctionDescriptor.class);
    Assertions.assertThat(descriptor.members().get("overloaded_foo")).isInstanceOf(AmbiguousDescriptor.class);
    Assertions.assertThat(descriptor.members().get("MyClass")).isInstanceOf(ClassDescriptor.class);
  }

  @Test
  void nullSymbolTest() {
    var converter = new ModuleSymbolToDescriptorConverter(PythonVersionUtils.allVersions());
    var descriptor = converter.convert(null);
    Assertions.assertThat(descriptor).isNull();
  }
}
