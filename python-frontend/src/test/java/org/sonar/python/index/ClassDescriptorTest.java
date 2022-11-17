/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastClassSymbol;
import static org.sonar.python.PythonTestUtils.lastClassSymbolWithName;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobuf;
import static org.sonar.python.index.DescriptorUtils.descriptor;

public class ClassDescriptorTest {

  @Test
  public void classDescriptor() {
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
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorWithSuperClass() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class B: ...",
      "class A(B): ...");
    assertThat(classDescriptor.superClasses()).containsExactly("package.mod.B");
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorWithSuperClassWithoutSymbol() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "def foo(): ...",
      "class A(foo()): ...");
    assertThat(classDescriptor.superClasses()).isEmpty();
    assertThat(classDescriptor.hasSuperClassWithoutDescriptor()).isTrue();
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorWithMetaclass() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "class B(type): ...",
      "class A(metaclass=B): ...");
    assertThat(classDescriptor.hasMetaClass()).isTrue();
    assertThat(classDescriptor.metaclassFQN()).isEqualTo("package.mod.B");
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorDecorator() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "@foo",
      "class A: ...");
    assertThat(classDescriptor.hasDecorators()).isTrue();
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorGenerics() {
    ClassDescriptor classDescriptor = lastClassDescriptor(
      "from typing import Generic",
      "class A(Generic[str]): ...");
    assertThat(classDescriptor.supportsGenerics()).isTrue();
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void classDescriptorWithVariousMembers() {
    ClassDescriptor classDescriptor = lastClassDescriptorWithName("A",
      "class A:",
      "  def foo(): ...",
      "  foo = 42",
      "  bar = 24",
      "  qix = 42",
      "  def qix(): ...",
      "  class Nested: ..."
    );
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
  }

  @Test
  public void protobufSerializationWithoutLocationNorFQN() {
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
    assertClassDescriptors(classDescriptor, fromProtobuf(toProtobuf(classDescriptor)));
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

  void assertClassDescriptors(ClassDescriptor first, ClassDescriptor second) {
    assertThat(first.hasDecorators()).isEqualTo(second.hasDecorators());
    assertThat(first.hasSuperClassWithoutDescriptor()).isEqualTo(second.hasSuperClassWithoutDescriptor());
    assertThat(first.hasMetaClass()).isEqualTo(second.hasMetaClass());
    assertThat(first.supportsGenerics()).isEqualTo(second.supportsGenerics());
    assertThat(first.name()).isEqualTo(second.name());
    assertThat(first.fullyQualifiedName()).isEqualTo(second.fullyQualifiedName());
    assertThat(first.superClasses()).isEqualTo(second.superClasses());
    assertThat(first.members()).usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(second.members());
    assertThat(first.definitionLocation()).usingRecursiveComparison().isEqualTo(second.definitionLocation());
    assertThat(first.metaclassFQN()).isEqualTo(second.metaclassFQN());
  }
}
