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
package org.sonar.python.semantic.v2.types;

import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.tree.CallExpressionImpl;
import org.sonar.python.tree.TokenImpl;
import org.sonar.python.tree.UnaryExpressionImpl;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.python.PythonTestUtils.getFirstDescendant;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class TrivialTypePropagationVisitorTest {
  private TrivialTypeInferenceVisitor trivialTypeInferenceVisitor;
  private TrivialTypePropagationVisitor trivialTypePropagationVisitor;

  @BeforeEach
  void setup() {
    trivialTypeInferenceVisitor = new TrivialTypeInferenceVisitor(PROJECT_LEVEL_TYPE_TABLE, pythonFile("mod"), "mod");
    trivialTypePropagationVisitor = new TrivialTypePropagationVisitor(PROJECT_LEVEL_TYPE_TABLE);
  }

  static Stream<Arguments> testSources() {
    return Stream.of(
      Arguments.of("-1", TypesTestUtils.INT_TYPE),
      Arguments.of("-1.0", TypesTestUtils.FLOAT_TYPE),
      Arguments.of("-(True)", TypesTestUtils.INT_TYPE),
      Arguments.of("-(1j)", TypesTestUtils.COMPLEX_TYPE),

      Arguments.of("+1", TypesTestUtils.INT_TYPE),
      Arguments.of("+1.0", TypesTestUtils.FLOAT_TYPE),
      Arguments.of("+(1j)", TypesTestUtils.COMPLEX_TYPE),
      Arguments.of("+(True)", TypesTestUtils.INT_TYPE),

      Arguments.of("~1", TypesTestUtils.INT_TYPE),
      Arguments.of("~(True)", TypesTestUtils.INT_TYPE),

      Arguments.of("not 1", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not 1.0", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not (2j)", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not (True)", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not x", TypesTestUtils.BOOL_TYPE)
    );
  }

  @MethodSource("testSources")
  @ParameterizedTest
  void test(String code, PythonType expectedType) {
    var expr = lastExpression(code);
    expr.accept(trivialTypeInferenceVisitor);
    expr.accept(trivialTypePropagationVisitor);
    assertThat(expr.typeV2())
      .isInstanceOfSatisfying(ObjectType.class, objectType ->
        assertThat(objectType.type()).isEqualTo(expectedType));
  }

  static Stream<Arguments> testUnknownReturnSources() {
    return Stream.of(
      Arguments.of("~x"),
      Arguments.of("~1.0"),
      Arguments.of("~(1j)"),
      Arguments.of("~(3+2j)"),
      Arguments.of("-x"),
      Arguments.of("+x")
    );
  }

  @ParameterizedTest
  @MethodSource("testUnknownReturnSources")
  void testUnknownReturn(String code) {
    var expr = lastExpression(code);
    expr.accept(trivialTypeInferenceVisitor);
    expr.accept(trivialTypePropagationVisitor);
    assertThat(expr.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  static Stream<Arguments> customNumberClassTestSource() {
    return Stream.of(
      Arguments.of("+(MyNum())", PythonType.UNKNOWN),
      Arguments.of("-(MyNum())", PythonType.UNKNOWN),
      Arguments.of("not (MyNum())", ObjectType.fromType(TypesTestUtils.BOOL_TYPE)),
      Arguments.of("~(MyNum())", PythonType.UNKNOWN)
    );
  }

  @MethodSource("customNumberClassTestSource")
  @ParameterizedTest
  void testCustomNumberClass(String code, PythonType expectedType) {
    var expr = lastExpression("class MyNum: pass", code);
    expr.accept(trivialTypeInferenceVisitor);
    expr.accept(trivialTypePropagationVisitor);

    assertThat(expr.typeV2()).isEqualTo(expectedType);
  }

  @Test
  void testNotOfCustomClass() {
    var expr = lastExpression("class MyNum: pass", "not MyNum()");
    expr.accept(trivialTypeInferenceVisitor);
    expr.accept(trivialTypePropagationVisitor);

    assertThat(expr.typeV2()).isInstanceOfSatisfying(ObjectType.class, objectType ->
      assertThat(objectType.type()).isEqualTo(TypesTestUtils.BOOL_TYPE));
  }

  @Test
  void testUnknownOperator() {
    var operator = mock(TokenImpl.class);
    when(operator.value()).thenReturn("invalid_operator");
    UnaryExpressionImpl expr = new UnaryExpressionImpl(operator, lastExpression("1"));
    expr.typeV2(TypesTestUtils.INT_TYPE);

    expr.accept(trivialTypePropagationVisitor);
    assertThat(expr.typeV2()).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void selfType_collapsedForInstanceMethodCall() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      a = A()
      a.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, tree -> tree instanceof CallExpression ce && ce.callee() instanceof QualifiedExpression);
    callExpr.accept(trivialTypePropagationVisitor);

    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classA = (ClassType) classDef.name().typeV2();

    // After collapsing, the type should be ObjectType[ClassType[A]], not ObjectType[SelfType[...]]
    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) resultType;
    assertThat(objectType.type()).isEqualTo(classA);
  }

  @Test
  void selfType_collapsedWithInheritance() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      class B(A):
        def bar(self): ...
      b = B()
      b.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    callExpr.accept(trivialTypePropagationVisitor);

    ClassDef classDefB = getFirstDescendant(fileInput, tree -> tree instanceof ClassDef classDef && classDef.name().name().equals("B"));
    ClassType classB = (ClassType) classDefB.name().typeV2();

    // After collapsing, the type should be ObjectType[ClassType[B]] (receiver type), not A
    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    ObjectType objectType = (ObjectType) resultType;
    assertThat(objectType.type()).isInstanceOf(ClassType.class);
    assertThat(objectType.type()).isNotInstanceOf(SelfType.class);
    assertThat(objectType.type()).isEqualTo(classB);
  }

  @Test
  void selfType_notCollapsedForStaticMethod() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        @staticmethod
        def foo() -> Self: ... # Self is not valid here, and should not be collapsed
      A.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    callExpr.accept(trivialTypePropagationVisitor);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void selfType_returnUnchangedIfNotSelfType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self) -> int: ...
      a = A()
      a.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    callExpr.accept(trivialTypePropagationVisitor);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType)
      .isInstanceOf(ObjectType.class)
      .extracting(PythonType::unwrappedType)
      .isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void selfType_recieverIsUnionType() {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      class B:
        def bar(self) -> Self: ...
      if ...:
        C = A
      else:
        C = B
      """);

    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var modFile = pythonFile("mod.py");
    projectLevelSymbolTable.addModule(fileInput, "", modFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    FileInput testFileInput = parseAndInferTypes(projectLevelTypeTable, pythonFile("test.py"), """
      from mod import C
      C.foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(testFileInput, CallExpression.class::isInstance);
    QualifiedExpression qualifiedExpr = (QualifiedExpression) callExpr.callee();
    assertThat(qualifiedExpr.qualifier().typeV2()).isInstanceOf(UnionType.class);

    assertThat(callExpr.typeV2()).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void selfType_receiverIsClassType() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...

      A.foo()
      """);
    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    callExpr.accept(trivialTypePropagationVisitor);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void selfType_reassignedFunctionYieldsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      from typing import Self
      class A:
        def foo(self) -> Self: ...
      bar = A.foo
      bar()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    callExpr.accept(trivialTypePropagationVisitor);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void callExpression_classInstantiation() {
    FileInput fileInput = parseAndInferTypes("""
      class MyClass:
        pass
      MyClass()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);
    ClassDef classDef = getFirstDescendant(fileInput, ClassDef.class::isInstance);
    ClassType classType = (ClassType) classDef.name().typeV2();

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) resultType).type()).isEqualTo(classType);
  }

  @Test
  void callExpression_functionCall() {
    FileInput fileInput = parseAndInferTypes("""
      def foo() -> int: ...
      foo()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) resultType).unwrappedType()).isEqualTo(TypesTestUtils.INT_TYPE);
  }

  @Test
  void callExpression_unknownCallee() {
    FileInput fileInput = parseAndInferTypes("""
      unknown_func()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void callExpression_methodCall() {
    FileInput fileInput = parseAndInferTypes("""
      class MyClass:
        def method(self) -> str: ...
      obj = MyClass()
      obj.method()
      """);

    CallExpressionImpl callExpr = getLastDescendant(fileInput, CallExpression.class::isInstance);

    PythonType resultType = callExpr.typeV2();
    assertThat(resultType).isInstanceOf(ObjectType.class);
    assertThat(((ObjectType) resultType).unwrappedType()).isEqualTo(TypesTestUtils.STR_TYPE);
  }

}
