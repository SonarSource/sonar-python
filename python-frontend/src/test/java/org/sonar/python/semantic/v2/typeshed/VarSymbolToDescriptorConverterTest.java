/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

class VarSymbolToDescriptorConverterTest {

  @Test
  void test() {
    var symbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("something")
      .setFullyQualifiedName("module.something")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("module.something_else")
        .build())
      .build();
    var converter = new VarSymbolToDescriptorConverter();

    var descriptor = (VariableDescriptor) converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("something");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module.something");
    Assertions.assertThat(descriptor.annotatedType()).isEqualTo("module.something_else");
  }

  @Test
  void builtinVarTest() {
    var symbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("int")
      .setFullyQualifiedName("builtins.int")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("builtins.int")
        .build())
      .build();
    var converter = new VarSymbolToDescriptorConverter();

    var descriptor = (VariableDescriptor) converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("int");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("int");
    Assertions.assertThat(descriptor.annotatedType()).isEqualTo("int");
  }

  @Test
  void test_is_type_alias_propagated() {
    var converter = new VarSymbolToDescriptorConverter();

    // is_type_alias=true must propagate through to VariableDescriptor.isTypeAlias()
    var aliasSymbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("Text")
      .setFullyQualifiedName("typing.Text")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("builtins.str")
        .build())
      .setIsTypeAlias(true)
      .build();

    var aliasDescriptor = (VariableDescriptor) converter.convert(aliasSymbol);
    Assertions.assertThat(aliasDescriptor.name()).isEqualTo("Text");
    Assertions.assertThat(aliasDescriptor.fullyQualifiedName()).isEqualTo("typing.Text");
    // TypeShedUtils.getTypesNormalizedFqn strips the "builtins." prefix
    Assertions.assertThat(aliasDescriptor.annotatedType()).isEqualTo("str");
    Assertions.assertThat(aliasDescriptor.isTypeAlias()).isTrue();

    // is_type_alias=false (default) must NOT set the flag on regular vars
    var regularSymbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("something")
      .setFullyQualifiedName("module.something")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("builtins.str")
        .build())
      .build();

    var regularDescriptor = (VariableDescriptor) converter.convert(regularSymbol);
    Assertions.assertThat(regularDescriptor.isTypeAlias()).isFalse();
  }

  @Test
  void test_typed_dict_exception() {
    var converter = new VarSymbolToDescriptorConverter();
    var symbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("TypedDict")
      .setFullyQualifiedName("typing.TypedDict")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("something")
        .build())
      .build();

    var descriptor = (VariableDescriptor) converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("TypedDict");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("typing.TypedDict");
    Assertions.assertThat(descriptor.annotatedType()).isNull();

    symbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("TypedDict")
      .setFullyQualifiedName("typing_extensions.TypedDict")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("something")
        .build())
      .build();

    descriptor = (VariableDescriptor) converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("TypedDict");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("typing_extensions.TypedDict");
    Assertions.assertThat(descriptor.annotatedType()).isNull();

    symbol = SymbolsProtos.VarSymbol.newBuilder()
      .setName("TypedDict")
      .setFullyQualifiedName("unrelated.TypedDict")
      .setTypeAnnotation(SymbolsProtos.Type.newBuilder()
        .setFullyQualifiedName("something")
        .build())
      .build();

    descriptor = (VariableDescriptor) converter.convert(symbol);
    Assertions.assertThat(descriptor.name()).isEqualTo("TypedDict");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("unrelated.TypedDict");
    Assertions.assertThat(descriptor.annotatedType()).isEqualTo("something");
  }

}
