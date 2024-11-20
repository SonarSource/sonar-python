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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6900")
public class NumpyWeekMaskValidationCheck extends PythonSubscriptionCheck {
  private static final Pattern PATTERN_STRING1 = Pattern.compile("^[01]{7}$");
  private static final Pattern PATTERN_STRING2 = Pattern
    .compile("^(Mon|Tue|Wed|Thu|Fri|Sat|Sun|\\s|\\\\t|\\\\n|\\\\x0b|\\\\x0c|\\\\r)*+$");
  private static final Set<String> VALID_WEEKMASK_ARRAY_VALUES = Set.of("0", "1");
  private static final String MESSAGE_ARRAY = "Array must have 7 elements, all of which are 0 or 1.";
  private static final String MESSAGE_STRING = "String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.";
  private static final String MESSAGE_SECONDARY_LOCATION = "Invalid mask is created here.";
  private static final Map<String, Integer> FUNCTIONS_PARAMETER_POSITION = Map.of(
    "numpy.busday_offset", 3,
    "numpy.busday_count", 2,
    "numpy.is_busday", 1,
    "numpy.busdaycalendar", 0);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyWeekMaskValidationCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    var weekmaskArgumentOptional = Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(FUNCTIONS_PARAMETER_POSITION::containsKey)
      .map(FUNCTIONS_PARAMETER_POSITION::get)
      .map(position -> TreeUtils.nthArgumentOrKeyword(position, "weekmask", callExpression.arguments()));
    if (weekmaskArgumentOptional.isEmpty()) {
      return;
    }

    RegularArgument weekmaskArgument = weekmaskArgumentOptional.get();
    checkExpression(context, weekmaskArgument.expression(), weekmaskArgument.expression(), null);
  }

  private static void checkExpression(SubscriptionContext context, Expression expression, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    if (expression.is(Tree.Kind.STRING_LITERAL)) {
      checkString(context, ((StringLiteral) expression).trimmedQuotesValue(), primaryLocation, secondaryLocation);
    } else if (expression.is(Tree.Kind.LIST_LITERAL)) {
      checkList(context, ((ListLiteral) expression), primaryLocation, secondaryLocation);
    } else if (expression.is(Tree.Kind.NAME)) {
      Expressions.singleAssignedNonNameValue((Name) expression).ifPresent(assignedExpression -> checkExpression(context, assignedExpression, primaryLocation, assignedExpression));
    }
  }

  private static void checkString(SubscriptionContext context, String string, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    if (PATTERN_STRING1.matcher(string).matches() || PATTERN_STRING2.matcher(string).matches()) {
      return;
    }
    createIssue(context, MESSAGE_STRING, primaryLocation, secondaryLocation);
  }

  private static void checkList(SubscriptionContext context, ListLiteral listLiteral, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    ExpressionList listElements = listLiteral.elements();
    List<Expression> expressionList = listElements.expressions();
    if (expressionList.size() == 7 && expressionList.stream()
      .allMatch(e -> TreeUtils.toOptionalInstanceOf(NumericLiteral.class, e)
        .map(NumericLiteral::valueAsString)
        .filter(VALID_WEEKMASK_ARRAY_VALUES::contains)
        .isPresent())) {
      return;
    }
    createIssue(context, MESSAGE_ARRAY, primaryLocation, secondaryLocation);
  }

  private static void createIssue(SubscriptionContext context, String message, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    PreciseIssue issue = context.addIssue(primaryLocation, message);
    if (secondaryLocation != null) {
      issue.secondary(secondaryLocation, MESSAGE_SECONDARY_LOCATION);
    }
  }
}
