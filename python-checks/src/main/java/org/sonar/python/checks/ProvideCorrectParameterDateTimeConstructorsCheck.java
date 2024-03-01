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
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6882")
public class ProvideCorrectParameterDateTimeConstructorsCheck extends PythonSubscriptionCheck {
  private static final int MIN_YEAR = 1;
  private static final int MAX_YEAR = 9999;
  private static final String MESSAGE = "The %s parameter must be valid.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ProvideCorrectParameterDateTimeConstructorsCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    if ("datetime.date".equals(calleeSymbol.fullyQualifiedName())) {
      checkDate(context, callExpression);
    } else if ("datetime.time".equals(calleeSymbol.fullyQualifiedName())) {
      checkTime(context, callExpression);
    } else if ("datetime.datetime".equals(calleeSymbol.fullyQualifiedName())) {
      checkDate(context, callExpression);
      checkTime(context, callExpression, 3);
    }
  }

  private static void checkTime(SubscriptionContext context, CallExpression callExpression) {
    checkTime(context, callExpression, 0);
  }

  private static void checkTime(SubscriptionContext context, CallExpression callExpression, int parameterOffset) {
    RegularArgument hourArgument = TreeUtils.nthArgumentOrKeyword(parameterOffset, "hour", callExpression.arguments());
    RegularArgument minuteArgument = TreeUtils.nthArgumentOrKeyword(parameterOffset + 1, "minute", callExpression.arguments());
    RegularArgument secondArgument = TreeUtils.nthArgumentOrKeyword(parameterOffset + 2, "second", callExpression.arguments());
    RegularArgument microsecondArgument = TreeUtils.nthArgumentOrKeyword(parameterOffset + 3, "microsecond", callExpression.arguments());

    if (hourArgument != null) {
      checkArgument(hourArgument, 0, 23, context, "hour");
    }
    if (minuteArgument != null) {
      checkArgument(minuteArgument, 0, 59, context, "minute");
    }
    if (secondArgument != null) {
      checkArgument(secondArgument, 0, 59, context, "second");
    }
    if (microsecondArgument != null) {
      checkArgument(microsecondArgument, 0, 999_999, context, "microsecond");
    }
  }

  private static void checkArgument(RegularArgument argument, long min, long max, SubscriptionContext context, String name) {
    if (!argument.expression().is(Tree.Kind.NUMERIC_LITERAL) && !argument.expression().is(Tree.Kind.UNARY_MINUS)) {
      return;
    }
    long value;
    if (argument.expression().is(Tree.Kind.NUMERIC_LITERAL)) {
      value = ((NumericLiteral) argument.expression()).valueAsLong();
    } else {
      UnaryExpression unaryExpression = (UnaryExpression) argument.expression();
      if (!unaryExpression.expression().is(Tree.Kind.NUMERIC_LITERAL)) {
        return;
      }
      value = -((NumericLiteral) unaryExpression.expression()).valueAsLong();
    }
    if (value < min || value > max) {
      context.addIssue(argument, String.format(MESSAGE, name));
    }
  }

  private static void checkDate(SubscriptionContext context, CallExpression callExpression) {
    RegularArgument yearArgument = TreeUtils.nthArgumentOrKeyword(0, "year", callExpression.arguments());
    RegularArgument monthArgument = TreeUtils.nthArgumentOrKeyword(1, "month", callExpression.arguments());
    RegularArgument dayArgument = TreeUtils.nthArgumentOrKeyword(2, "day", callExpression.arguments());

    if (yearArgument == null || monthArgument == null || dayArgument == null) {
      return;
    }

    checkArgument(yearArgument, MIN_YEAR, MAX_YEAR, context, "year");
    checkArgument(monthArgument, 1, 12, context, "month");
    checkArgument(dayArgument, 1, 31, context, "day");
  }

}
