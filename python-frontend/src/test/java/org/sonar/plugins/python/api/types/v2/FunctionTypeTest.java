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
package org.sonar.plugins.python.api.types.v2;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class FunctionTypeTest {

  static PythonFile pythonFile = PythonTestUtils.pythonFile("mod");

  @Test
  void arity() {
    FunctionType functionType = functionType("def fn(): pass");
    assertThat(functionType.isAsynchronous()).isFalse();
    assertThat(functionType.parameters()).isEmpty();
    assertThat(functionType.displayName()).contains("Callable");
    assertThat(functionType).hasToString("FunctionType[fn]");
    assertThat(functionType.unwrappedType()).isEqualTo(functionType);
    assertThat(functionType.instanceDisplayName()).isEmpty();
    String fileId = SymbolUtils.pathOf(pythonFile).toString();
    assertThat(functionType.definitionLocation()).contains(new LocationInFile(fileId, 1, 4, 1, 6));

    functionType = functionType("async def fn(p1, p2, p3): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2", "p3");
    assertThat(functionType.hasVariadicParameter()).isFalse();
    assertThat(functionType.isInstanceMethod()).isFalse();
    assertThat(functionType.isAsynchronous()).isTrue();
    assertThat(functionType.hasDecorators()).isFalse();
    assertThat(functionType.parameters()).extracting(ParameterV2::hasDefaultValue).containsExactly(false, false, false);
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, false, false);

    functionType = functionType("def fn(p1, *, p2): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2");
    assertThat(functionType.parameters()).extracting(ParameterV2::hasDefaultValue).containsExactly(false, false);
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, true);

    functionType = functionType("def fn(p1, /, p2): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2");
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, false);
    assertThat(functionType.parameters()).extracting(ParameterV2::isPositionalOnly).containsExactly(true, false);

    functionType = functionType("def fn(p1, /, p2, *, p3): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2", "p3");
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, false, true);
    assertThat(functionType.parameters()).extracting(ParameterV2::isPositionalOnly).containsExactly(true, false, false);

    functionType = functionType("def fn(p1, /, p2, *p3, p4 = False): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2", "p3", "p4");
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, false, false, true);
    assertThat(functionType.parameters()).extracting(ParameterV2::isPositionalOnly).containsExactly(true, false, false, false);
    assertThat(functionType.parameters()).extracting(ParameterV2::isVariadic).containsExactly(false, false, true, false);

    functionType = functionType("def fn(p1, p2=42): pass");
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2");
    assertThat(functionType.parameters()).extracting(ParameterV2::hasDefaultValue).containsExactly(false, true);
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, false);

    functionType = functionType("def fn(p1, *, p2=42): pass");
    assertThat(functionType.hasVariadicParameter()).isFalse();
    assertThat(functionType.parameters()).extracting(ParameterV2::name).containsExactly("p1", "p2");
    assertThat(functionType.parameters()).extracting(ParameterV2::hasDefaultValue).containsExactly(false, true);
    assertThat(functionType.parameters()).extracting(ParameterV2::isKeywordOnly).containsExactly(false, true);

    functionType = functionType("def fn((p1,p2,p3)): pass");
    assertThat(functionType.parameters()).hasSize(1);
    assertThat(functionType.parameters().get(0).name()).isNull();
    assertThat(functionType.parameters().get(0).hasDefaultValue()).isFalse();
    assertThat(functionType.parameters().get(0).isKeywordOnly()).isFalse();
    assertThat(functionType.parameters().get(0).isVariadic()).isFalse();
    assertThat(functionType.parameters().get(0).isKeywordVariadic()).isFalse();
    assertThat(functionType.parameters().get(0).isPositionalVariadic()).isFalse();
    assertThat(functionType.parameters().get(0).location()).isEqualTo(new LocationInFile(fileId, 1, 7, 1, 17));

    functionType = functionType("def fn(**kwargs): pass");
    assertThat(functionType.parameters()).hasSize(1);
    assertThat(functionType.hasVariadicParameter()).isTrue();
    assertThat(functionType.parameters().get(0).name()).isEqualTo("kwargs");
    assertThat(functionType.parameters().get(0).hasDefaultValue()).isFalse();
    assertThat(functionType.parameters().get(0).isKeywordOnly()).isFalse();
    assertThat(functionType.parameters().get(0).isVariadic()).isTrue();
    assertThat(functionType.parameters().get(0).isKeywordVariadic()).isTrue();
    assertThat(functionType.parameters().get(0).isPositionalVariadic()).isFalse();

    functionType = functionType("def fn(p1, *args): pass");
    assertThat(functionType.hasVariadicParameter()).isTrue();

    functionType = functionType("class A:\n  def method(self, p1): pass");
    assertThat(functionType.isInstanceMethod()).isTrue();

    functionType = functionType("class A:\n  def method(*args, p1): pass");
    assertThat(functionType.isInstanceMethod()).isTrue();

    functionType = functionType("class A:\n  @staticmethod\n  def method((a, b), c): pass");
    assertThat(functionType.isInstanceMethod()).isFalse();

    functionType = functionType("class A:\n  @staticmethod\n  def method(p1, p2): pass");
    assertThat(functionType.isInstanceMethod()).isFalse();

    functionType = functionType("class A:\n  @classmethod\n  def method(self, p1): pass");
    assertThat(functionType.isInstanceMethod()).isFalse();
    assertThat(functionType.hasDecorators()).isTrue();

    functionType = functionType("class A:\n  @dec\n  def method(self, p1): pass");
    assertThat(functionType.isInstanceMethod()).isTrue();
    assertThat(functionType.hasDecorators()).isTrue();

    functionType = functionType("class A:\n  @some[\"thing\"]\n  def method(self, p1): pass");
    assertThat(functionType.isInstanceMethod()).isTrue();
    assertThat(functionType.hasDecorators()).isTrue();
  }

  @Test
  void fullyQualifiedName() {
    FunctionType functionType = functionType("def foo(): ...");
    assertThat(functionType.name()).isEqualTo("foo");
    assertThat(functionType.fullyQualifiedName()).isEqualTo("my_package.mod.foo");

    functionType = functionType("""
      class MyClass:
        def bar(): ...
      """);
    assertThat(functionType.name()).isEqualTo("bar");
    assertThat(functionType.fullyQualifiedName()).isEqualTo("my_package.mod.MyClass.bar");

    functionType = functionType("""
      class MyClass:
        class MyNestedClass:
          def qix(): ...
      """);
    assertThat(functionType.name()).isEqualTo("qix");
    assertThat(functionType.fullyQualifiedName()).isEqualTo("my_package.mod.MyClass.MyNestedClass.qix");

    FileInput fileInput = parseAndInferTypes(pythonFile, """
      def baz():
        def nested(): ...
      """);
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(1);
    functionType = (FunctionType) functionDef.name().typeV2();
    assertThat(functionType.name()).isEqualTo("nested");
    assertThat(functionType.fullyQualifiedName()).isEqualTo("my_package.mod.baz.nested");
  }

  @Test
  void declaredTypes() {
    // TODO: SONARPY-1776 handle declared return type
    FunctionType functionType = functionType("def fn(p1: int): pass");
    assertThat(functionType.returnType()).isEqualTo(PythonType.UNKNOWN);
    assertThat(functionType.parameters()).extracting(ParameterV2::declaredType).extracting(TypeWrapper::type).extracting(PythonType::unwrappedType).containsExactly(INT_TYPE);
  }

  @Test
  void declared_return_type() {
    FunctionType functionType = functionType("def fn() -> int: ...");
    assertThat(functionType.returnType().unwrappedType()).isEqualTo(INT_TYPE);
    functionType = functionType("def fn() -> unknown: ...");
    assertThat(functionType.returnType()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void decorators() {
    FunctionType functionType = functionType("@something\ndef fn(p1, *args): pass");
    assertThat(functionType.hasDecorators()).isTrue();

    functionType = functionType("@something[\"else\"]\ndef fn(p1, *args): pass");
    assertThat(functionType.hasDecorators()).isTrue();
  }

  @Test
  void owner() {
    FileInput fileInput = parseWithoutSymbols(
      "class A:",
      "  def foo(self): pass"
    );
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "").inferTypes(fileInput);

    ClassDef classDef = (ClassDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF)).get(0);
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);

    FunctionType functionType = (FunctionType) functionDef.name().typeV2();
    ClassType classType = (ClassType) classDef.name().typeV2();
    assertThat(functionType.owner()).isEqualTo(classType);

    functionType = functionType("def foo(): ...");
    assertThat(functionType.owner()).isNull();
  }

  @Test
  void location() {
    FunctionType functionType = functionType("def fn(param: int, (a: str, b)): pass");
    String fileId = SymbolUtils.pathOf(pythonFile).toString();
    assertThat(functionType.definitionLocation()).contains(new LocationInFile(fileId, 1, 4, 1, 6));
    assertThat(functionType.parameters().get(0).location()).isEqualTo(new LocationInFile(fileId, 1, 7, 1, 17));
    assertThat(functionType.parameters().get(1).location()).isEqualTo(new LocationInFile(fileId, 1, 19, 1, 30));
  }

  public static FunctionType functionType(String... code) {
    FileInput fileInput = parseWithoutSymbols(code);
    var symbolTable = new SymbolTableBuilderV2(fileInput)
      .build();
    new TypeInferenceV2(PROJECT_LEVEL_TYPE_TABLE, pythonFile, symbolTable, "my_package").inferTypes(fileInput);
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    return (FunctionType) functionDef.name().typeV2();
  }

}
