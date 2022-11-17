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


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastFunctionSymbol;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobuf;
import static org.sonar.python.index.DescriptorUtils.descriptor;

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
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
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
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void parameterWithDefaultValue() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x=42): ...");
    FunctionDescriptor.Parameter x = functionDescriptor.parameters().get(0);
    assertThat(x.hasDefaultValue()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void parameterWithType() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void parameterWithPositionOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, /, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.isPositionalOnly()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void parameterWithKeywordOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, *, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(1);
    assertThat(parameter.isKeywordOnly()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void parameterWithPositional() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void variadicParameter() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(*x, **y): ...");
    FunctionDescriptor.Parameter parameter1 = functionDescriptor.parameters().get(0);
    FunctionDescriptor.Parameter parameter2 = functionDescriptor.parameters().get(1);
    assertThat(parameter1.isVariadic()).isTrue();
    assertThat(parameter2.isVariadic()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void decorators() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "@bar",
      "def foo(x): ...");
    assertThat(functionDescriptor.hasDecorators()).isTrue();
    assertThat(functionDescriptor.decorators()).containsExactly("bar");
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void asyncFunctions() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("async def foo(): ...");
    assertThat(functionDescriptor.isAsynchronous()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void method() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "class A:",
      "  def foo(self): ...");
    assertThat(functionDescriptor.isInstanceMethod()).isTrue();
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  @Test
  public void protobufSerializationWithoutLocationAndWithAnnotatedReturnType() {
    // FIXME: Annotated type name is never set for regular project decorators (only Typeshed) - SONARPY-1202
    List<FunctionDescriptor.Parameter> parameters = new ArrayList<>();
    parameters.add(new FunctionDescriptor.Parameter(null, "str", false, false, false, false, false, null));
    FunctionDescriptor functionDescriptor = new FunctionDescriptor(
      "foo",
      "mod.foo",
      parameters,
      false,
      false,
      Collections.emptyList(),
      false,
      null,
      "str"
    );
    assertFunctionDescriptors(functionDescriptor, fromProtobuf(toProtobuf(functionDescriptor)));
  }

  public static FunctionDescriptor lastFunctionDescriptor(String... code) {
    FunctionSymbol functionSymbol = lastFunctionSymbol(code);
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) descriptor(functionSymbol);
    assertThat(functionDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(functionDescriptor.name()).isEqualTo(functionSymbol.name());
    assertThat(functionDescriptor.fullyQualifiedName()).isEqualTo(functionSymbol.fullyQualifiedName());
    assertThat(functionDescriptor.definitionLocation()).isNotNull();
    assertThat(functionDescriptor.definitionLocation()).isEqualTo(functionSymbol.definitionLocation());
    return functionDescriptor;
  }

  void assertFunctionDescriptors(FunctionDescriptor first, FunctionDescriptor second) {
    assertThat(first.isAsynchronous()).isEqualTo(second.isAsynchronous());
    assertThat(first.isInstanceMethod()).isEqualTo(second.isInstanceMethod());
    assertThat(first.hasDecorators()).isEqualTo(second.hasDecorators());
    assertThat(first.name()).isEqualTo(second.name());
    assertThat(first.fullyQualifiedName()).isEqualTo(second.fullyQualifiedName());
    assertThat(first.annotatedReturnTypeName()).isEqualTo(second.annotatedReturnTypeName());
    assertThat(first.decorators()).containsExactlyElementsOf(second.decorators());
    assertThat(first.parameters()).usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(second.parameters());
    assertThat(first.definitionLocation()).usingRecursiveComparison().isEqualTo(second.definitionLocation());
  }
}
