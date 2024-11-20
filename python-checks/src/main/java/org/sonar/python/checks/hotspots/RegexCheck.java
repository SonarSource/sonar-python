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
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = RegexCheck.CHECK_KEY)
public class RegexCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S4784";
  private static final String MESSAGE = "Make sure that using a regular expression is safe here.";
  private static final int REGEX_ARGUMENT = 0;
  private static final Set<String> questionableFunctions = new HashSet<>(Arrays.asList(
    "django.core.validators.RegexValidator", "django.urls.conf.re_path",
    "re.compile", "re.match", "re.search", "re.fullmatch", "re.split", "re.findall", "re.finditer", "re.sub", "re.subn",
    "regex.compile", "regex.match", "regex.search", "regex.fullmatch", "regex.split", "regex.findall", "regex.finditer", "regex.sub", "regex.subn",
    "regex.subf", "regex.subfn", "regex.splititer"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression call = (CallExpression) ctx.syntaxNode();
      Symbol symbol = call.calleeSymbol();
      if (symbol != null && questionableFunctions.contains(symbol.fullyQualifiedName()) && !call.arguments().isEmpty()) {
        checkRegexArgument(call.arguments().get(REGEX_ARGUMENT), ctx);
      }
    });
  }

  private static void checkRegexArgument(Argument arg, SubscriptionContext ctx) {
    String literal = null;
    IssueLocation secondaryLocation = null;
    if (!arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
      return;
    }
    Expression argExpression = getExpression(((RegularArgument) arg).expression());
    if (argExpression.is(Tree.Kind.NAME)) {
      Expression expression = getExpression(Expressions.singleAssignedValue((Name) argExpression));
      if (expression != null && expression.is(Tree.Kind.STRING_LITERAL)) {
        secondaryLocation = IssueLocation.preciseLocation(expression, "");
        literal = Expressions.unescape((StringLiteral) expression);
      }
    } else if (argExpression.is(Tree.Kind.STRING_LITERAL)) {
      literal = Expressions.unescape((StringLiteral) argExpression);
    }
    if (literal == null) {
      return;
    }
    if (isSuspiciousRegex(literal)) {
      PreciseIssue preciseIssue = ctx.addIssue(arg, MESSAGE);
      if (secondaryLocation != null) {
        preciseIssue.secondary(secondaryLocation);
      }
    }
  }

  private static Expression getExpression(@Nullable Expression expr) {
    if (expr == null) {
      return null;
    }
    expr = Expressions.removeParentheses(expr);
    if (expr.is(Tree.Kind.MODULO) || expr.is(Tree.Kind.PLUS)) {
      return getExpression(((BinaryExpression) expr).leftOperand());
    }
    if (expr.is(Tree.Kind.ASSIGNMENT_EXPRESSION)) {
      return getExpression(((AssignmentExpression) expr).expression());
    }
    return expr;
  }

  /**
   * This rule flags any execution of a hardcoded regular expression which has at least 3 characters and at least
   * two instances of any of the following characters: "*+{" (Example: (a+)*)
   */
  private static boolean isSuspiciousRegex(String regexp) {
    if (regexp.length() > 2) {
      int nOfSuspiciousChars = regexp.length() - regexp.replaceAll("[*+{]", "").length();
      return nOfSuspiciousChars > 1;
    }
    return false;
  }
}
