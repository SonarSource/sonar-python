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
package org.sonar.python.index;


import java.util.Collections;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.types.v2.ClassType;

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
    ClassDescriptor classDescriptor = new ClassDescriptor(
      "foo",
      "mod.foo",
      Collections.emptyList(),
      Collections.emptySet(),
      false,
      null,
      false,
      false,
      null,
      false
    );
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
