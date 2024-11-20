/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.types.v2.FunctionType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.types.v2.TypesTestUtils.lastFunctionDef;

class FunctionDescriptorTest {

  @Test
  void functionDescriptor() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x): ...");
    assertThat(functionDescriptor.decorators()).isEmpty();
    assertThat(functionDescriptor.hasDecorators()).isFalse();
    assertThat(functionDescriptor.annotatedReturnTypeName()).isNull();
    assertThat(functionDescriptor.isInstanceMethod()).isFalse();
    assertThat(functionDescriptor.isAsynchronous()).isFalse();
    assertThat(functionDescriptor.parameters()).hasSize(1);
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameters() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x): ...");
    FunctionDescriptor.Parameter x = functionDescriptor.parameters().get(0);
    assertThat(x.name()).isEqualTo("x");
    assertThat(x.hasDefaultValue()).isFalse();
    assertThat(x.isKeywordOnly()).isFalse();
    assertThat(x.isPositionalOnly()).isFalse();
    assertThat(x.isVariadic()).isFalse();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameterWithDefaultValue() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x=42): ...");
    FunctionDescriptor.Parameter x = functionDescriptor.parameters().get(0);
    assertThat(x.hasDefaultValue()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameterWithType() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameterWithPositionOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, /, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.isPositionalOnly()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameterWithKeywordOnly() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x, *, y): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(1);
    assertThat(parameter.isKeywordOnly()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void parameterWithPositional() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(x: str): ...");
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.annotatedType()).isEqualTo("str");
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void variadicParameter() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("def foo(*x, **y): ...");
    FunctionDescriptor.Parameter parameter1 = functionDescriptor.parameters().get(0);
    FunctionDescriptor.Parameter parameter2 = functionDescriptor.parameters().get(1);
    assertThat(parameter1.isVariadic()).isTrue();
    assertThat(parameter2.isVariadic()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void unknown_decorators_have_no_name() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "@bar",
      "def foo(x): ...");
    assertThat(functionDescriptor.hasDecorators()).isTrue();
    // Empty decorator name due to it not being resolved
    assertThat(functionDescriptor.decorators()).isEmpty();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void decorator_from_function() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      """
        def bar(): ...
        
        @bar
        def foo(x): ...
        """
    );
    assertThat(functionDescriptor.hasDecorators()).isTrue();
    assertThat(functionDescriptor.decorators()).containsExactly("my_package.mod.bar");
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void asyncFunctions() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor("async def foo(): ...");
    assertThat(functionDescriptor.isAsynchronous()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void method() {
    FunctionDescriptor functionDescriptor = lastFunctionDescriptor(
      "class A:",
      "  def foo(self): ...");
    assertThat(functionDescriptor.isInstanceMethod()).isTrue();
    assertDescriptorToProtobuf(functionDescriptor);
  }

  @Test
  void protobufSerializationWithoutLocationAndWithAnnotatedReturnType() {
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
    assertDescriptorToProtobuf(functionDescriptor);
  }

  public static FunctionDescriptor lastFunctionDescriptor(String... code) {
    FunctionDef functionDef = lastFunctionDef(code);
    SymbolV2 symbol = functionDef.name().symbolV2();
    FunctionType functionType = (FunctionType) functionDef.name().typeV2();

    PythonTypeToDescriptorConverter converter = new PythonTypeToDescriptorConverter();
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) converter.convert("my_package.mod", symbol, Set.of(functionType));

    assertThat(functionDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(functionDescriptor.name()).isEqualTo(symbol.name());
    assertThat(functionDescriptor.fullyQualifiedName()).isEqualTo(functionType.fullyQualifiedName());
    assertThat(functionDescriptor.definitionLocation()).isNotNull();
    assertThat(functionType.definitionLocation()).contains(functionDescriptor.definitionLocation());
    return functionDescriptor;
  }
}
