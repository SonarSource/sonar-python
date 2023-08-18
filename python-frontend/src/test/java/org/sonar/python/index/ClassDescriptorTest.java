/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.index;


import java.util.Collections;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastClassSymbol;
import static org.sonar.python.PythonTestUtils.lastClassSymbolWithName;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.index.DescriptorUtils.descriptor;

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
    assertThat(fooMethod.fullyQualifiedName()).isEqualTo("package.mod.A.foo");
    assertDescriptorToProtobuf(classDescriptor);
  }

  @Test
  void classDescriptorWithSuperClass() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class B: ...",
      "class A(B): ...");
    assertThat(classDescriptor.superClasses()).containsExactly("package.mod.B");
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
    assertThat(classDescriptor.metaclassFQN()).isEqualTo("package.mod.B");
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
  void classDescriptorGenerics() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "from typing import Generic",
      "class A(Generic[str]): ...");
    assertThat(classDescriptor.supportsGenerics()).isTrue();
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
      null,
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
    ClassSymbol classSymbol;
    if (name == null) {
      classSymbol = lastClassSymbol(code);
    } else {
      classSymbol = lastClassSymbolWithName(name, code);
    }
    ClassDescriptor classDescriptor = (ClassDescriptor) descriptor(classSymbol);
    assertThat(classDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);
    assertThat(classDescriptor.name()).isEqualTo(classSymbol.name());
    assertThat(classDescriptor.fullyQualifiedName()).isEqualTo(classSymbol.fullyQualifiedName());
    assertThat(classDescriptor.definitionLocation()).isNotNull();
    assertThat(classDescriptor.definitionLocation()).isEqualTo(classSymbol.definitionLocation());
    return classDescriptor;
  }
}
