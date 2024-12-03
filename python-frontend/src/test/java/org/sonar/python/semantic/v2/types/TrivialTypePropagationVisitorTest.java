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
package org.sonar.python.semantic.v2.types;

import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.tree.TokenImpl;
import org.sonar.python.tree.UnaryExpressionImpl;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;

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
      Arguments.of("not (MyNum())", new ObjectType(TypesTestUtils.BOOL_TYPE)),
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

}