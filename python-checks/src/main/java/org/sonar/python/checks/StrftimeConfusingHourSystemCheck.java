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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
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
    Expressions.ifNameGetSingleAssignedNonNameValue(expression)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .filter(stringLiteral -> stringLiteral.stringElements().stream().noneMatch(stringElement -> "f".equalsIgnoreCase(stringElement.prefix())))
      .ifPresent(stringLiteral -> checkDateFormatStringLiteral(context, primaryLocation, stringLiteral));
  }

  private static void checkDateFormatStringLiteral(SubscriptionContext context, Tree primaryLocation, StringLiteral stringLiteral) {
    String value = stringLiteral.trimmedQuotesValue();
    String effectiveMessage = null;
    if (value.contains("%H") && value.contains("%p")) {
      effectiveMessage = MESSAGE;
    } else if (value.contains("%I") && !value.contains("%p")) {
      effectiveMessage = MESSAGE_12_HOURS;
    }
    if (effectiveMessage != null) {
      var issue = context.addIssue(primaryLocation, effectiveMessage);
      if (primaryLocation != stringLiteral) {
        issue.secondary(stringLiteral, MESSAGE_SECONDARY_LOCATION);
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
