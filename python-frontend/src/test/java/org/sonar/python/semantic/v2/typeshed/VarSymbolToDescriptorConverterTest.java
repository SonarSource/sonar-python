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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.TypeShedTypeTable;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;

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

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 2})
  void preservesIndexedUnionAnnotationThroughAliases(int aliasDepth) {
    var intType = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.INSTANCE).setFullyQualifiedName("builtins.int").build();
    var noneType = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.NONE).build();
    var unionType = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.UNION).addArgTypeIds(1).addArgTypeIds(2).build();
    var module = SymbolsProtos.ModuleSymbol.newBuilder().addTypeTable(intType).addTypeTable(noneType).addTypeTable(unionType);
    for (int i = 0; i < aliasDepth; i++) {
      module.addTypeTable(SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.TYPE_ALIAS)
        .addArgTypeIds(module.getTypeTableCount()));
    }
    var symbol = SymbolsProtos.VarSymbol.newBuilder().setName("errno").setTypeAnnotationId(module.getTypeTableCount()).build();
    var converter = new VarSymbolToDescriptorConverter(TypeShedTypeTable.from(module.build()));

    var descriptor = (VariableDescriptor) converter.convert(symbol, "builtins.OSError");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("OSError.errno");
    Assertions.assertThat(descriptor.annotatedType()).isNull();
    var annotation = descriptor.typeAnnotationDescriptor();
    for (int i = 0; i < aliasDepth; i++) {
      Assertions.assertThat(annotation).isNotNull();
      Assertions.assertThat(annotation.kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.TYPE_ALIAS);
      annotation = annotation.args().get(0);
    }
    Assertions.assertThat(annotation.kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.UNION);
    Assertions.assertThat(annotation.args()).extracting(TypeAnnotationDescriptor::kind)
      .containsExactly(TypeAnnotationDescriptor.TypeKind.INSTANCE, TypeAnnotationDescriptor.TypeKind.NONE);
    Assertions.assertThat(annotation.args().get(0).fullyQualifiedName()).isEqualTo("int");
    assertDescriptorToProtobuf(descriptor);

    Assertions.assertThat(((VariableDescriptor) converter.convert(symbol.toBuilder().setIsTypeAlias(true).build()))
      .typeAnnotationDescriptor()).isNull();
    Assertions.assertThat(((VariableDescriptor) converter.convert(symbol.toBuilder().setIsImportedModule(true).build()))
      .typeAnnotationDescriptor()).isNull();
  }

  @Test
  void keepsSingleTypeAnnotationsOnTheQualifiedNamePath() {
    var intType = SymbolsProtos.Type.newBuilder().setFullyQualifiedName("builtins.int").build();
    var alias = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.TYPE_ALIAS).addArgs(intType).build();
    var symbol = SymbolsProtos.VarSymbol.newBuilder().setName("value").setTypeAnnotation(alias).build();

    var descriptor = (VariableDescriptor) new VarSymbolToDescriptorConverter().convert(symbol);

    Assertions.assertThat(descriptor.annotatedType()).isEqualTo("int");
    Assertions.assertThat(descriptor.typeAnnotationDescriptor()).isNull();
  }

}
