/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2.types;

import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
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
      Arguments.of("~1.0", TypesTestUtils.INT_TYPE),
      Arguments.of("~(3+2j)", TypesTestUtils.INT_TYPE),
      Arguments.of("~(1j)", TypesTestUtils.INT_TYPE),
      Arguments.of("~(True)", TypesTestUtils.INT_TYPE),

      Arguments.of("not 1", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not 1.0", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not (2j)", TypesTestUtils.BOOL_TYPE),
      Arguments.of("not (True)", TypesTestUtils.BOOL_TYPE)
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

  static Stream<Arguments> customNumberClassTestSource() {
    return Stream.of(
      Arguments.of("+(MyNum())", PythonType.UNKNOWN),
      Arguments.of("-(MyNum())", PythonType.UNKNOWN),
      Arguments.of("not (MyNum())", new ObjectType(TypesTestUtils.BOOL_TYPE)),
      Arguments.of("~(MyNum())", new ObjectType(TypesTestUtils.INT_TYPE))
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
}