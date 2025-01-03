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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1764")
public class IdenticalExpressionOnBinaryOperatorCheck extends PythonSubscriptionCheck {

  private static final List<Tree.Kind> kinds = Arrays.asList(Tree.Kind.MINUS, Tree.Kind.DIVISION, Tree.Kind.FLOOR_DIVISION, Tree.Kind.MODULO,
    Tree.Kind.SHIFT_EXPR, Tree.Kind.BITWISE_AND, Tree.Kind.BITWISE_OR, Tree.Kind.BITWISE_XOR, Tree.Kind.AND, Tree.Kind.OR, Tree.Kind.COMPARISON, Tree.Kind.IS,Tree.Kind.IN);

  @Override
  public void initialize(Context context) {
    kinds.forEach(k -> context.registerSyntaxNodeConsumer(k, IdenticalExpressionOnBinaryOperatorCheck::checkBinaryExpression));
  }

  private static void checkBinaryExpression(SubscriptionContext ctx) {
    BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
    Expression leftOperand = binaryExpression.leftOperand();
    Expression rightOperand = binaryExpression.rightOperand();
    Token operator = binaryExpression.operator();
    if (CheckUtils.areEquivalent(leftOperand, rightOperand) && !"<<".equals(operator.value()) && !isException(leftOperand)) {
      ctx.addIssue(rightOperand, "Correct one of the identical sub-expressions on both sides of operator \"" + operator.value() + "\".")
        .secondary(leftOperand, "");
    }
  }

  private static boolean isException(Expression leftOperand) {
    // Avoid raising issue if operands are function calls or within try/except blocks
    return leftOperand.is(Tree.Kind.CALL_EXPR) || TreeUtils.hasDescendant(leftOperand, t -> t.is(Tree.Kind.CALL_EXPR))
      || TreeUtils.firstAncestorOfKind(leftOperand, Tree.Kind.TRY_STMT) != null;
  }
}
