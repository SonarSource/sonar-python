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
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.index.ClassDescriptorTest.lastClassDescriptor;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.index.FunctionDescriptorTest.lastFunctionDescriptor;
import static org.sonar.python.types.v2.TypesTestUtils.lastName;

class AmbiguousDescriptorTest {

  @Test
  void test_basic_ambiguous_descriptor() {
    AmbiguousDescriptor ambiguousDescriptor = lastAmbiguousDescriptor(
      """
        if x:
          class A: ...
        else:
          class A: ...
        A
        """
    );
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactly("A", "A");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::fullyQualifiedName).containsExactly("my_package.mod.A", "my_package.mod.A");
    assertDescriptorToProtobuf(ambiguousDescriptor);
  }

  @Test
  void test_ambiguous_descriptor_different_kinds() {
    AmbiguousDescriptor ambiguousDescriptor = lastAmbiguousDescriptor(
      """
        if foo:
          class A: ...
        elif bar:
          A: int = 42
        else:
          def A(): ...
        A
        """
    );
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactly("A", "A", "A");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::fullyQualifiedName).containsExactly("my_package.mod.A", "my_package.mod.A", "my_package.mod.A");
    assertDescriptorToProtobuf(ambiguousDescriptor);
  }

  @Test
  void test_flattened_ambiguous_descriptor() {
    AmbiguousDescriptor firstAmbiguousSymbol = lastAmbiguousDescriptor(
      """
        if x:
          class A: ...
        else:
          class A: ...
        A
        """
    );
    ClassDescriptor classDescriptor = lastClassDescriptor("class A: ...");
    AmbiguousDescriptor ambiguousDescriptor = AmbiguousDescriptor.create(firstAmbiguousSymbol, classDescriptor);
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactly("A", "A", "A");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::fullyQualifiedName).containsExactly("my_package.mod.A", "my_package.mod.A", "my_package.mod.A");
    assertDescriptorToProtobuf(ambiguousDescriptor);
  }

  @Test
  void test_single_descriptor_illegal_argument() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def func(): ...");
    assertThatThrownBy(() -> AmbiguousDescriptor.create(functionDescriptor)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void test_nested_ambiguous_descriptors_illegal_argument() {
    AmbiguousDescriptor ambiguousDescriptor = new AmbiguousDescriptor("foo", "foo", Collections.emptySet());
      Set<Descriptor> descriptors = Set.of(ambiguousDescriptor);
    assertThatThrownBy(() -> new AmbiguousDescriptor("foo", "foo", descriptors)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void test_different_names_illegal_argument() {
    FunctionDescriptor functionDescriptorA = lastFunctionDescriptor("def a(): ...");
    FunctionDescriptor functionDescriptorB = lastFunctionDescriptor("def b(): ...");
    assertThatThrownBy(() -> AmbiguousDescriptor.create(functionDescriptorA, functionDescriptorB)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void ambiguous_descriptor_creation_different_name_same_fqn() {
    Descriptor fooDesc = new VariableDescriptor("foo", "mod.bar", null, false);
    Descriptor barDesc = new VariableDescriptor("bar", "mod.bar", null, false);
    assertThatThrownBy(() -> AmbiguousDescriptor.create(fooDesc, barDesc)).isInstanceOf(IllegalArgumentException.class);
  }

  private AmbiguousDescriptor lastAmbiguousDescriptor(String... code) {
    Name name = lastName(code);
    PythonType pythonType = name.typeV2();
    if (!(pythonType instanceof UnionType unionType)) {
      throw new AssertionError("Symbol is not ambiguous.");
    }
    SymbolV2 symbol = name.symbolV2();
    PythonTypeToDescriptorConverter converter = new PythonTypeToDescriptorConverter();
    AmbiguousDescriptor ambiguousDescriptor = (AmbiguousDescriptor) converter.convert("my_package.mod", symbol, Set.of(unionType));
    assertThat(ambiguousDescriptor.name()).isEqualTo(name.name());
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo("my_package.mod." + symbol.name());
    return ambiguousDescriptor;
  }
}
