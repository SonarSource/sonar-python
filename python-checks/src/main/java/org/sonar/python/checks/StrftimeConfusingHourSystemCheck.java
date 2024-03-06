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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6883")
public class StrftimeConfusingHourSystemCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).";
  public static final String MESSAGE_12_HOURS = "Use %I (12-hour clock) with %p (AM/PM).";
  private static final String MESSAGE_SECONDARY_LOCATION = "Wrong format created here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, StrftimeConfusingHourSystemCheck::checkCallExpr);
  }

  private static void checkExpression(SubscriptionContext context, Expression expression) {
    checkExpression(context, expression, expression);
  }

  private static void checkExpression(SubscriptionContext context, Expression expression, Tree primaryLocation) {
    if (expression.is(Tree.Kind.NAME)) {
      Expressions.singleAssignedNonNameValue((Name) expression).ifPresent(a -> checkExpression(context, a, primaryLocation));
    }
    if (expression.is(Tree.Kind.STRING_LITERAL)) {
      StringLiteral stringLiteral = (StringLiteral) expression;
      if (stringLiteral.stringElements().stream().anyMatch(stringElement -> "f".equalsIgnoreCase(stringElement.prefix()))) {
        return;
      }
      String value = stringLiteral.trimmedQuotesValue();
      String effectiveMessage = null;
      if (value.contains("%H") && value.contains("%p")) {
        effectiveMessage = MESSAGE;
      } else if (value.contains("%I") && !value.contains("%p")) {
        effectiveMessage = MESSAGE_12_HOURS;
      }
      if (effectiveMessage != null) {
        var issue = context.addIssue(primaryLocation, effectiveMessage);
        if (primaryLocation != expression) {
          issue.secondary(stringLiteral, MESSAGE_SECONDARY_LOCATION);
        }
      }
    }
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    String fullyQualifiedName = calleeSymbol.fullyQualifiedName();
    if (!"datetime.time.strftime".equals(fullyQualifiedName)) {
      return;
    }

    RegularArgument formatArgument = TreeUtils.nthArgumentOrKeyword(0, "format", callExpression.arguments());
    if (formatArgument == null) {
      return;
    }
    checkExpression(context, formatArgument.expression());
  }
}
