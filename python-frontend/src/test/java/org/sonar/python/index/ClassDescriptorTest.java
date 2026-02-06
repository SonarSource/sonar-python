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


import java.util.Set;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.types.v2.TypesTestUtils.lastClassDef;
import static org.sonar.python.types.v2.TypesTestUtils.lastClassDefWithName;

class ClassDescriptorTest {

  @Test
  void classDescriptor() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class A:",
      "  def foo(): ...");
    assertThat(classDescriptor.superClasses()).isEmpty();
    assertThat(classDescriptor.hasSuperClassWithoutDescriptor()).isFalse();
    assertThat(classDescriptor.hasMetaClass()).isFalse();
    assertThat(classDescriptor.metaclassFQN()).isNull();
    assertThat(classDescriptor.supportsGenerics()).isFalse();
    assertThat(classDescriptor.hasDecorators()).isFalse();
    FunctionDescriptor fooMethod = ((FunctionDescriptor) classDescriptor.members().iterator().next());
    assertThat(fooMethod.name()).isEqualTo("foo");
    assertThat(fooMethod.fullyQualifiedName()).isEqualTo("my_package.mod.A.foo");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithSuperClass() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class B: ...",
      "class A(B): ...");
    assertThat(classDescriptor.superClasses()).containsExactly("my_package.mod.B");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithSuperClassWithoutSymbol() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "def foo(): ...",
      "class A(foo()): ...");
    assertThat(classDescriptor.superClasses()).isEmpty();
    assertThat(classDescriptor.hasSuperClassWithoutDescriptor()).isTrue();
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithMetaclass() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class B(type): ...",
      "class A(metaclass=B): ...");
    assertThat(classDescriptor.hasMetaClass()).isTrue();
    assertThat(classDescriptor.metaclassFQN()).isEqualTo("my_package.mod.B");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorDecorator() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "@foo",
      "class A: ...");
    assertThat(classDescriptor.hasDecorators()).isTrue();
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void multipleClassSymbolNotAmbiguous() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      """
        class A: ...
        class A: ...
        """
    );
    assertThat(classDescriptor.name()).isEqualTo("A");
    assertThat(classDescriptor.fullyQualifiedName()).isEqualTo("my_package.mod.A");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorGenerics() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "from typing import Generic",
      "class A(Generic[str]): ...");
    assertThat(classDescriptor.supportsGenerics()).isFalse();
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithVariousMembers() {
    ClassDescriptor classDescriptor = lastClassDescriptorWithName("A",
      "class A:",
      "  def foo(): ...",
      "  foo = 42",
      "  bar = 24",
      "  qix = 42",
      "  def qix(): ...",
      "  class Nested: ..."
    );
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void protobufSerializationWithoutLocationNorFQN() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("foo")
      .withFullyQualifiedName("mod.foo")
      .build();
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorBuilderWithIsSelf() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .withIsSelf(true)
      .build();
    assertThat(classDescriptor.isSelf()).isTrue();
  }

  @Test
  void classDescriptorWithAttributes() {
    VariableDescriptor attrDescriptor = new VariableDescriptor("int", "builtins.int", "builtins.int");
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyGeneric")
      .withFullyQualifiedName("mod.MyGeneric")
      .withAttributes(java.util.List.of(attrDescriptor))
      .build();
    assertThat(classDescriptor.attributes()).hasSize(1);
    assertThat(classDescriptor.attributes().get(0).fullyQualifiedName()).isEqualTo("builtins.int");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithNestedAttributes() {
    VariableDescriptor nestedAttr = new VariableDescriptor("str", "builtins.str", "builtins.str");
    VariableDescriptor attrDescriptor = new VariableDescriptor("list", "builtins.list", "builtins.list", false, java.util.List.of(nestedAttr), java.util.List.of());
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyGeneric")
      .withFullyQualifiedName("mod.MyGeneric")
      .withAttributes(java.util.List.of(attrDescriptor))
      .build();
    assertThat(classDescriptor.attributes()).hasSize(1);
    VariableDescriptor firstAttr = (VariableDescriptor) classDescriptor.attributes().get(0);
    assertThat(firstAttr.attributes()).hasSize(1);
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorDefaultsToEmptyAttributes() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .build();
    assertThat(classDescriptor.attributes()).isEmpty();
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithMetaClasses() {
    ClassDescriptor metaclassDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("ABCMeta")
      .withFullyQualifiedName("abc.ABCMeta")
      .build();
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .withHasMetaClass(true)
      .withMetaClasses(java.util.List.of(metaclassDescriptor))
      .build();
    assertThat(classDescriptor.metaClasses()).hasSize(1);
    assertThat(classDescriptor.metaClasses().get(0).fullyQualifiedName()).isEqualTo("abc.ABCMeta");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithMultipleMetaClasses() {
    ClassDescriptor metaclass1 = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("ABCMeta")
      .withFullyQualifiedName("abc.ABCMeta")
      .build();
    ClassDescriptor metaclass2 = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("EnumMeta")
      .withFullyQualifiedName("enum.EnumMeta")
      .build();
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .withHasMetaClass(true)
      .withMetaClasses(java.util.List.of(metaclass1, metaclass2))
      .build();
    assertThat(classDescriptor.metaClasses()).hasSize(2);
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorDefaultsToEmptyMetaClasses() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .build();
    assertThat(classDescriptor.metaClasses()).isEmpty();
    assertDescriptorToProtobuf(classDescriptor);
  }

  public static ClassDescriptor lastClassDescriptor(String... code) {
    return lastClassDescriptorWithName(null, code);
  }

  public static ClassDescriptor lastClassDescriptorWithName(@Nullable String name, String... code) {
    ClassDef classDef;
    ClassType classType;
    SymbolV2 symbol;
    if (name == null) {
      classDef = lastClassDef(code);
    } else {
      classDef = lastClassDefWithName(name, code);
    }
    classType = (ClassType) classDef.name().typeV2();
    symbol = classDef.name().symbolV2();
    PythonTypeToDescriptorConverter converter = new PythonTypeToDescriptorConverter();
    ClassDescriptor classDescriptor = (ClassDescriptor) converter.convert("my_package.mod", symbol, Set.of(classType));
    assertThat(classDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);
    assertThat(classDescriptor.name()).isEqualTo(classType.name());
    assertThat(classDescriptor.definitionLocation()).isNotNull();
    assertThat(classType.definitionLocation()).contains(classDescriptor.definitionLocation());
    return classDescriptor;
  }
}
