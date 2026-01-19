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

import java.util.ArrayList;
import java.util.List;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AwaitExpression;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.YieldExpression;

public class TypeDependenciesCalculator {

  public List<Expression> getTypeDependencies(Expression expression) {
    if (expression instanceof BinaryExpression e) {
      return List.of(e.leftOperand(), e.rightOperand());
    } else if (expression instanceof QualifiedExpression e) {
      return List.of(e.qualifier());
    } else if (expression instanceof AwaitExpression e) {
      return List.of(e.expression());
    } else if (expression instanceof AssignmentExpression e) {
      return List.of(e.expression());
    } else if (expression instanceof CallExpression e) {
      return List.of(e.callee());
    } else if (expression instanceof ConditionalExpression e) {
      return List.of(e.trueExpression(), e.falseExpression());
    } else if (expression instanceof ParenthesizedExpression e) {
      return List.of(e.expression());
    } else if (expression instanceof SliceExpression e) {
      return List.of(e.object());
    } else if (expression instanceof SubscriptionExpression e) {
      return List.of(e.object());
    } else if (expression instanceof UnaryExpression e) {
      return List.of(e.expression());
    } else if (expression instanceof UnpackingExpression e) {
      return List.of(e.expression());
    } else if (expression instanceof ExpressionList e) {
      return new ArrayList<>(e.expressions());
    } else if (expression instanceof ComprehensionExpression e) {
      return List.of(e.resultExpression());
    } else if (expression instanceof DictCompExpression e) {
      return List.of(e.keyExpression(), e.valueExpression());
    } else if (expression instanceof Tuple e) {
      return new ArrayList<>(e.elements());
    } else if (expression instanceof YieldExpression e) {
      return new ArrayList<>(e.expressions());
    }
    return List.of();
  }

}
