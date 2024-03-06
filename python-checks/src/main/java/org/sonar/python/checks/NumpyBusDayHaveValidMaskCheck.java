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

import java.util.List;
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
public class NumpyBusDayHaveValidMaskCheck extends PythonSubscriptionCheck {
  public static final Pattern PATTERN_STRING1 = Pattern.compile("^[01]{7}$");
  public static final Pattern PATTERN_STRING2 = Pattern.compile("^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)(\\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun))*+$");
  private static final String MESSAGE_ARRAY = "Array must have 7 elements, all of which are 0 or 1.";
  private static final String MESSAGE_STRING = "String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.";
  public static final String MESSAGE_SECONDARY_LOCATION = "Invalid mask is created here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyBusDayHaveValidMaskCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol == null) {
      return;
    }
    if (!"numpy.busday_offset".equals(calleeSymbol.fullyQualifiedName())) {
      return;
    }

    RegularArgument weekmaskArgument = TreeUtils.nthArgumentOrKeyword(3, "weekmask", callExpression.arguments());
    if (weekmaskArgument == null) {
      return;
    }

    if (weekmaskArgument.expression().is(Tree.Kind.STRING_LITERAL)) {
      checkString(context, ((StringLiteral) weekmaskArgument.expression()).trimmedQuotesValue(), weekmaskArgument.expression());
    } else if (weekmaskArgument.expression().is(Tree.Kind.LIST_LITERAL)) {
      checkList(context, ((ListLiteral) weekmaskArgument.expression()));
    } else if (weekmaskArgument.expression().is(Tree.Kind.NAME)) {
      Name name = (Name) weekmaskArgument.expression();
      Expressions.singleAssignedNonNameValue(name).ifPresent(assignedExpression -> {
        if (assignedExpression.is(Tree.Kind.STRING_LITERAL)) {
          checkString(context, ((StringLiteral) assignedExpression).trimmedQuotesValue(), weekmaskArgument.expression(), assignedExpression);
        } else if (assignedExpression.is(Tree.Kind.LIST_LITERAL)) {
          checkList(context, ((ListLiteral) assignedExpression), weekmaskArgument.expression(), assignedExpression);
        }
      });
    }
  }

  private static void checkString(SubscriptionContext context, String string, Tree primaryLocation) {
    checkString(context, string, primaryLocation, null);
  }

  private static void checkString(SubscriptionContext context, String string, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    if (PATTERN_STRING1.matcher(string).matches() || PATTERN_STRING2.matcher(string).matches()) {
      return;
    }
    createIssue(context, MESSAGE_STRING, primaryLocation, secondaryLocation);
  }

  private static void checkList(SubscriptionContext context, ListLiteral listLiteral) {
    checkList(context, listLiteral, listLiteral, null);
  }

  private static void checkList(SubscriptionContext context, ListLiteral listLiteral, Tree primaryLocation, @Nullable Tree secondaryLocation) {
    ExpressionList listElements = listLiteral.elements();
    List<Expression> expressionList = listElements.expressions();
    if (expressionList.stream()
      .filter(e -> e.is(Tree.Kind.NUMERIC_LITERAL))
      .map(NumericLiteral.class::cast)
      .allMatch(numericLiteral -> "0".equals(numericLiteral.valueAsString()) || "1".equals(numericLiteral.valueAsString()))
      && expressionList.size() == 7) {
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
