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


import java.util.Collection;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastFunctionSymbol;
import static org.sonar.python.index.DescriptorUtils.descriptors;

public class FunctionDescriptorTest {

  @Test
  public void functionDescriptor() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x): ...");
    assertThat(functionDescriptor.decorators()).isEmpty();
    assertThat(functionDescriptor.hasDecorators()).isFalse();
    assertThat(functionDescriptor.annotatedReturnTypeName()).isNull();
    assertThat(functionDescriptor.isInstanceMethod()).isFalse();
    assertThat(functionDescriptor.isAsynchronous()).isFalse();
    assertThat(functionDescriptor.parameters()).hasSize(1);
  }

  @Test
  public void parameters() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x): ...");
    FunctionDescriptor.Parameter x = functionDescriptor.parameters().get(0);
    assertThat(x.name()).isEqualTo("x");
    assertThat(x.hasDefaultValue()).isFalse();
    assertThat(x.isKeywordOnly()).isFalse();
    assertThat(x.isPositionalOnly()).isFalse();
    assertThat(x.isVariadic()).isFalse();
  }

  @Test
  public void parameterWithDefaultValue() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x=42): ...");
    FunctionDescriptor.Parameter x = functionDescriptor.parameters().get(0);
    assertThat(x.hasDefaultValue()).isTrue();
  }

  @Test
  public void parameterWithType() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
  }

  @Test
  public void parameterWithPositionOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, /, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.isPositionalOnly()).isTrue();
  }

  @Test
  public void parameterWithKeywordOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, *, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(1);
    assertThat(parameter.isKeywordOnly()).isTrue();
  }

  @Test
  public void parameterWithPositional() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
  }

  @Test
  public void variadicParameter() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(*x): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.isVariadic()).isTrue();
  }

  @Test
  public void decorators() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "@bar",
      "def foo(x): ...");
    assertThat(functionDescriptor.hasDecorators()).isTrue();
    assertThat(functionDescriptor.decorators()).containsExactly("bar");
  }

  @Test
  public void asyncFunctions() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("async def foo(): ...");
    assertThat(functionDescriptor.isAsynchronous()).isTrue();
  }

  @Test
  public void method() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "class A:",
      "  def foo(self): ...");
    assertThat(functionDescriptor.isInstanceMethod()).isTrue();
  }

  private FunctionDescriptor lastFunctionDescriptor(String... code) {
    FunctionSymbol functionSymbol = lastFunctionSymbol(code);
    Collection<Descriptor> descriptors = descriptors(functionSymbol);
    assertThat(descriptors).extracting(Descriptor::kind).containsExactly(Descriptor.Kind.FUNCTION);
    FunctionDescriptor functionDescriptor = ((FunctionDescriptor) descriptors.iterator().next());
    assertThat(functionDescriptor.name()).isEqualTo(functionSymbol.name());
    assertThat(functionDescriptor.fullyQualifiedName()).isEqualTo(functionSymbol.fullyQualifiedName());
    assertThat(functionDescriptor.definitionLocation()).isNotNull();
    assertThat(functionDescriptor.definitionLocation()).isEqualTo(functionSymbol.definitionLocation());
    return functionDescriptor;
  }
}
