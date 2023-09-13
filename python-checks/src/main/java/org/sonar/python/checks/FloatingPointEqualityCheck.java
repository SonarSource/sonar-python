/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Set;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S1244")
public class FloatingPointEqualityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not perform equality checks with floating point values.";
  private static final String QUICK_FIX_MESSAGE = "Replace with %smath.isclose().";

  private static final String QUICK_FIX_REPLACE = "%smath.isclose(%s, %s, rel_tol=1e-09, abs_tol=1e-09)";

  private static final Tree.Kind[] BINARY_OPERATION_KINDS = { Tree.Kind.PLUS, Tree.Kind.MINUS, Tree.Kind.MULTIPLICATION,
      Tree.Kind.DIVISION };

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT,
        ctx -> reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, this::checkFloatingPointEquality);
  }

  private void checkFloatingPointEquality(SubscriptionContext ctx) {
    BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
    String operator = binaryExpression.operator().value();
    if (("==".equals(operator) || "!=".equals(operator)) && isAnyOperandFloatingPoint(binaryExpression)) {
      PreciseIssue issue = ctx.addIssue(binaryExpression, MESSAGE);
      issue.addQuickFix(createQuickFix(binaryExpression, operator));
    }
  }

  private boolean isAnyOperandFloatingPoint(BinaryExpression binaryExpression) {
    Expression leftOperand = binaryExpression.leftOperand();
    Expression rightOperand = binaryExpression.rightOperand();

    return isFloat(leftOperand) || isFloat(rightOperand) ||
        isAssignedFloat(leftOperand) || isAssignedFloat(rightOperand) ||
        isBinaryOperationWithFloat(leftOperand) || isBinaryOperationWithFloat(rightOperand);
  }

  private static boolean isFloat(Expression expression) {
    return expression.is(Tree.Kind.NUMERIC_LITERAL) && expression.type().equals(InferredTypes.FLOAT);
  }

  private boolean isAssignedFloat(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Set<Expression> values = reachingDefinitionsAnalysis.valuesAtLocation((Name) expression);
      if(!values.isEmpty()){
        return values.stream().allMatch(value -> isFloat(value));
      }
    }
    return false;
  }

  private boolean isBinaryOperationWithFloat(Expression expression) {
    if (expression.is(BINARY_OPERATION_KINDS)) {
      return isAnyOperandFloatingPoint((BinaryExpression) expression);
    }
    return false;
  }

  private static PythonQuickFix createQuickFix(BinaryExpression binaryExpression, String operator) {
    String notToken = "!=".equals(operator) ? "!" : "";
    String message = String.format(QUICK_FIX_MESSAGE, notToken);
    String mathIsCloseText = String.format(QUICK_FIX_REPLACE, notToken,
        TreeUtils.treeToString(binaryExpression.leftOperand(), false),
        TreeUtils.treeToString(binaryExpression.rightOperand(), false));

    return PythonQuickFix.newQuickFix(message)
        .addTextEdit(TextEditUtils.replace(binaryExpression, mathIsCloseText))
        .build();
  }
}
