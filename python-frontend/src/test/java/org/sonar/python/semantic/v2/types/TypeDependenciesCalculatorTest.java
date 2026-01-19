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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastStatement;

class TypeDependenciesCalculatorTest {

  private final TypeDependenciesCalculator calculator = new TypeDependenciesCalculator();

  @Test
  void binaryExpression() {
    var expr = (BinaryExpression) lastExpression("a or b");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.leftOperand(), expr.rightOperand());

    expr = (BinaryExpression) lastExpression("a + b");
    dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.leftOperand(), expr.rightOperand());
  }

  @Test
  void qualifiedExpression() {
    var expr = (QualifiedExpression) lastExpression("a.b");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.qualifier());
  }

  @Test
  void awaitExpression() {
    var expr = (AwaitExpression) lastExpression("await a");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());
  }

  @Test
  void assignmentExpression() {
    var paren = (ParenthesizedExpression) lastExpression("(x := 1)");
    var expr = (AssignmentExpression) paren.expression();
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());
  }

  @Test
  void callExpression() {
    var expr = (CallExpression) lastExpression("a()");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.callee());
  }

  @Test
  void conditionalExpression() {
    var expr = (ConditionalExpression) lastExpression("a if cond else b");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.trueExpression(), expr.falseExpression());
  }

  @Test
  void parenthesizedExpression() {
    var expr = (ParenthesizedExpression) lastExpression("(a)");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());
  }

  @Test
  void sliceExpression() {
    var expr = (SliceExpression) lastExpression("a[1:2]");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.object());
  }

  @Test
  void subscriptionExpression() {
    var expr = (SubscriptionExpression) lastExpression("a[0]");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.object());
  }

  @Test
  void unaryExpression() {
    var expr = (UnaryExpression) lastExpression("-a");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());

    expr = (UnaryExpression) lastExpression("not a");
    dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());
  }

  @Test
  void expressionList() {
    var subscription = (SubscriptionExpression) lastExpression("a[1, 2]");
    var exprList = subscription.subscripts();
    var dependencies = calculator.getTypeDependencies(exprList);
    assertThat(dependencies).containsExactlyElementsOf(exprList.expressions());
  }

  @Test
  void comprehensionExpression() {
    var expr = (ComprehensionExpression) lastExpression("[x for x in a]");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.resultExpression());
  }

  @Test
  void dictCompExpression() {
    var expr = (DictCompExpression) lastExpression("{k: v for k, v in items}");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.keyExpression(), expr.valueExpression());
  }

  @Test
  void tuple() {
    var expr = (Tuple) lastExpression("(a, b)");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactlyElementsOf(expr.elements());
  }

  @Test
  void unpackingExpression() {
    var list = (ListLiteral) lastExpression("[*a]");
    var expr = (UnpackingExpression) list.elements().expressions().get(0);
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactly(expr.expression());
  }

  @Test
  void yieldExpression() {
    var yieldStmt = (YieldStatement) lastStatement("yield a");
    var expr = yieldStmt.yieldExpression();
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).containsExactlyElementsOf(expr.expressions());
  }

  @Test
  void unknownExpression_returnsEmpty() {
    var expr = lastExpression("a");
    var dependencies = calculator.getTypeDependencies(expr);
    assertThat(dependencies).isEmpty();
  }

}
