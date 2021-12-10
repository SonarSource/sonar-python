/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Arrays;
import java.util.HashSet;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.PythonTestUtils.lastSymbolFromDef;
import static org.sonar.python.index.ClassDescriptorTest.lastClassDescriptor;
import static org.sonar.python.index.DescriptorUtils.descriptor;
import static org.sonar.python.index.FunctionDescriptorTest.lastFunctionDescriptor;

public class AmbiguousDescriptorTest {

  @Test
  public void test_basic_ambiguous_descriptor() {
    AmbiguousDescriptor ambiguousDescriptor = lastAmbiguousDescriptor(
      "class A: ...",
      "class A: ...");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactly("A", "A");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::fullyQualifiedName).containsExactly("package.mod.A", "package.mod.A");
  }

  @Test
  public void test_flattened_ambiguous_descriptor() {
    AmbiguousDescriptor firstAmbiguousSymbol = lastAmbiguousDescriptor(
      "class A: ...",
      "class A: ...");
    ClassDescriptor classDescriptor = lastClassDescriptor("class A: ...");
    AmbiguousDescriptor ambiguousDescriptor = AmbiguousDescriptor.create(firstAmbiguousSymbol, classDescriptor);
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactly("A", "A", "A");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::fullyQualifiedName).containsExactly("package.mod.A", "package.mod.A", "package.mod.A");
  }

  @Test
  public void test_single_descriptor_illegal_argument() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def func(): ...");
    assertThatThrownBy(() -> AmbiguousDescriptor.create(functionDescriptor)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  public void test_different_names_illegal_argument() {
    FunctionDescriptor functionDescriptorA = lastFunctionDescriptor("def a(): ...");
    FunctionDescriptor functionDescriptorB = lastFunctionDescriptor("def b(): ...");
    assertThatThrownBy(() -> AmbiguousDescriptor.create(functionDescriptorA, functionDescriptorB)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  public void ambiguous_descriptor_creation_different_name_same_fqn() {
    SymbolImpl foo = new SymbolImpl("foo", "mod.bar");
    SymbolImpl bar = new SymbolImpl("bar", "mod.bar");
    Descriptor fooDesc = descriptor(foo);
    Descriptor barDesc = descriptor(bar);
    AmbiguousDescriptor ambiguousDescriptor = AmbiguousDescriptor.create(new HashSet<>(Arrays.asList(fooDesc, barDesc)));
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo("mod.bar");
    assertThat(ambiguousDescriptor.name()).isEmpty();
  }

  private AmbiguousDescriptor lastAmbiguousDescriptor(String... code) {
    Symbol ambiguousSymbol = lastSymbolFromDef(code);
    if (!(ambiguousSymbol instanceof AmbiguousSymbol)) {
      throw new AssertionError("Symbol is not ambiguous.");
    }
    AmbiguousDescriptor ambiguousDescriptor = (AmbiguousDescriptor) descriptor(ambiguousSymbol);
    assertThat(ambiguousDescriptor.name()).isEqualTo(ambiguousSymbol.name());
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo(ambiguousSymbol.fullyQualifiedName());
    return ambiguousDescriptor;
  }
}
