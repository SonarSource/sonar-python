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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.tree.BinaryExpressionImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;

class TypeDependenciesCalculatorTest {


  @Test
  void hasTypeDependenciesTest() {
    var calculator = new TypeDependenciesCalculator();
    var expression = lastExpression("a or b");
    var hasTypeDependencies = calculator.hasTypeDependencies(expression);
    assertThat(hasTypeDependencies).isTrue();

    expression = lastExpression("a()");
    hasTypeDependencies = calculator.hasTypeDependencies(expression);
    assertThat(hasTypeDependencies).isTrue();

    expression = lastExpression("a");
    hasTypeDependencies = calculator.hasTypeDependencies(expression);
    assertThat(hasTypeDependencies).isFalse();
  }

  @Test
  void getTypeDependenciesTest() {
    var calculator = new TypeDependenciesCalculator();
    var binaryExpression = (BinaryExpression) lastExpression("a or b");
    var dependencies = calculator.getTypeDependencies(binaryExpression);
    assertThat(dependencies).containsExactly(binaryExpression.leftOperand(), binaryExpression.rightOperand());


    binaryExpression = (BinaryExpression) lastExpression("a + b");
    dependencies = calculator.getTypeDependencies(binaryExpression);
    assertThat(dependencies).containsExactly(binaryExpression.leftOperand(), binaryExpression.rightOperand());

    binaryExpression = (BinaryExpressionImpl) lastExpression("a @ b");
    dependencies = calculator.getTypeDependencies(binaryExpression);
    assertThat(dependencies).isEmpty();

    var callExpression = (CallExpression) lastExpression("a()");
    dependencies = calculator.getTypeDependencies(callExpression);
    assertThat(dependencies).containsExactly(callExpression.callee());

    var expression = lastExpression("a");
    dependencies = calculator.getTypeDependencies(expression);
    assertThat(dependencies).isEmpty();
  }


}
