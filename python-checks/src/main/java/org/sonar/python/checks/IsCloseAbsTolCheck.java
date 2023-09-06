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

import java.util.Optional;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6727")
public class IsCloseAbsTolCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide the abs_tol parameter when using math.isclose to compare a value to 0";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR,
        ctx -> checkForIsCloseAbsTolArgument(ctx, (CallExpression) ctx.syntaxNode()));
  }

  private static void checkForIsCloseAbsTolArgument(SubscriptionContext ctx, CallExpression call) {
    Optional.ofNullable(call.calleeSymbol())
        .filter(symbol -> "math.isclose".equals(symbol.fullyQualifiedName()) &&
            anyArgumentIsZero(call) &&
            TreeUtils.argumentByKeyword("abs_tol", call.arguments()) == null)
        .ifPresent(s -> ctx.addIssue(call, MESSAGE));
  }

  private static boolean anyArgumentIsZero(CallExpression call) {
    RegularArgument firstArg = TreeUtils.nthArgumentOrKeyword(0, "a", call.arguments());
    RegularArgument secondArg = TreeUtils.nthArgumentOrKeyword(1, "b", call.arguments());
    return (firstArg != null && isLitteralZeroOrAssignedZero(firstArg.expression())) ||
        (secondArg != null && isLitteralZeroOrAssignedZero(secondArg.expression()));
  }

  private static boolean isLitteralZeroOrAssignedZero(Expression expression) {
    return isZero(expression) || isAssignedZero(expression);
  }

  private static boolean isAssignedZero(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Expression assignedValue = Expressions.singleAssignedValue((Name) expression);
      return assignedValue != null && isZero(assignedValue);
    }
    return false;
  }

  private static boolean isZero(Expression expression) {
    return expression.is(Tree.Kind.NUMERIC_LITERAL) && "0".equals(((NumericLiteral) (expression)).valueAsString());
  }
}
