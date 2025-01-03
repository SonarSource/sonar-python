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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6882")
public class IncorrectParameterDatetimeConstructorsCheck extends PythonSubscriptionCheck {
  private static final int MIN_YEAR = 1;
  private static final int MAX_YEAR = 9999;
  private static final String MESSAGE = "Provide a correct value for the `%s` parameter.";
  private static final String MESSAGE_SECONDARY_LOCATION = "An invalid value is assigned here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, IncorrectParameterDatetimeConstructorsCheck::checkCallExpr);
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
      checkArgument(context, hourArgument, 0, 23, "hour");
    }
    if (minuteArgument != null) {
      checkArgument(context, minuteArgument, 0, 59, "minute");
    }
    if (secondArgument != null) {
      checkArgument(context, secondArgument, 0, 59, "second");
    }
    if (microsecondArgument != null) {
      checkArgument(context, microsecondArgument, 0, 999_999, "microsecond");
    }
  }

  private static class ValueWithExpression {
    private final long value;
    private final Tree expression;

    public ValueWithExpression(long value, Tree expression) {
      this.value = value;
      this.expression = expression;
    }

    public long value() {
      return value;
    }

    public Tree expression() {
      return expression;
    }
  }

  private static ValueWithExpression getValue(Expression expression) {
    if (expression.is(Tree.Kind.NUMERIC_LITERAL)) {
      return new ValueWithExpression(((NumericLiteral) expression).valueAsLong(), expression);
    } else if (expression.is(Tree.Kind.UNARY_MINUS)) {
      UnaryExpression unaryExpression = (UnaryExpression) expression;
      if (!unaryExpression.expression().is(Tree.Kind.NUMERIC_LITERAL)) {
        return null;
      }
      return new ValueWithExpression(-((NumericLiteral) unaryExpression.expression()).valueAsLong(), unaryExpression);
    } else if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedNonNameValue((Name) expression).map(IncorrectParameterDatetimeConstructorsCheck::getValue).orElse(null);
    }
    return null;
  }

  private static void checkArgument(SubscriptionContext context, RegularArgument argument, long min, long max, String name) {
    ValueWithExpression valueWithExpression = getValue(argument.expression());
    if (valueWithExpression == null) {
      return;
    }
    long value = valueWithExpression.value();
    Tree secondaryLocation = argument.expression() == valueWithExpression.expression() ? null : valueWithExpression.expression();
    if (value < min || value > max) {
      PreciseIssue issue = context.addIssue(argument, String.format(MESSAGE, name));
      if (secondaryLocation != null) {
        issue.secondary(secondaryLocation, MESSAGE_SECONDARY_LOCATION);
      }
    }
  }

  private static void checkDate(SubscriptionContext context, CallExpression callExpression) {
    RegularArgument yearArgument = TreeUtils.nthArgumentOrKeyword(0, "year", callExpression.arguments());
    RegularArgument monthArgument = TreeUtils.nthArgumentOrKeyword(1, "month", callExpression.arguments());
    RegularArgument dayArgument = TreeUtils.nthArgumentOrKeyword(2, "day", callExpression.arguments());

    if (yearArgument == null || monthArgument == null || dayArgument == null) {
      return;
    }

    checkArgument(context, yearArgument, MIN_YEAR, MAX_YEAR, "year");
    checkArgument(context, monthArgument, 1, 12, "month");
    checkArgument(context, dayArgument, 1, 31, "day");
  }
}
