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
package org.sonar.python.semantic.v2.types.typecalculator;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class QualifiedExpressionCalculatorTest {

  private final TypePredicateContext typePredicateContext = TypePredicateContext.of(PROJECT_LEVEL_TYPE_TABLE);
  private final QualifiedExpressionCalculator calculator = new QualifiedExpressionCalculator(typePredicateContext);

  @Test
  void calculate_basicMemberAccess_returnsMethodType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isInstanceOf(FunctionType.class);
    assertThat(result.name()).isEqualTo("foo");
  }

  @Test
  void calculate_propertyDecorator_returnsReturnType() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        @property
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void calculate_propertySubclassDecorator_returnsReturnType() {
    FileInput fileInput = parseAndInferTypes("""
      class subclass_property(property):...

      class A:
        @subclass_property
        def foo(self) -> int:
          return 42
      a = A()
      a.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
  }

  @Test
  void calculate_unknownQualifier_returnsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      unknown.foo
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void calculate_nonExistentMember_returnsUnknown() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        pass
      a = A()
      a.non_existent
      """);

    var qualifiedExpression = getQualifiedExpressionFromStatement(fileInput);
    PythonType result = calculator.calculate(qualifiedExpression);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  private static QualifiedExpression getQualifiedExpressionFromStatement(FileInput fileInput) {
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.QUALIFIED_EXPR));
  }
}
