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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5714")
public class BooleanExpressionInExceptCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Rewrite this \"except\" expression as a tuple of exception classes.";
  public static final String QUICK_FIX_MESSAGE = "Replace with a tuple";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.EXCEPT_CLAUSE, BooleanExpressionInExceptCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(Kind.EXCEPT_GROUP_CLAUSE, BooleanExpressionInExceptCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    ExceptClause except = (ExceptClause) ctx.syntaxNode();
    Optional.of(except)
      .map(ExceptClause::exception)
      .map(Expressions::removeParentheses)
      .filter(exception -> exception.is(Kind.OR, Kind.AND))
      .ifPresent(exception -> {
        var issue = ctx.addIssue(exception, MESSAGE);
        addQuickFix(issue, exception);
      });
  }

  private static List<String> collectNames(Expression expression) {
    expression = Expressions.removeParentheses(expression);
    if (expression.is(Kind.OR, Kind.AND)) {
      var binaryExpression = (BinaryExpression) expression;
      var leftExceptions = collectNames(binaryExpression.leftOperand());
      var rightExceptions = collectNames(binaryExpression.rightOperand());
      var result = new ArrayList<String>();
      result.addAll(leftExceptions);
      result.addAll(rightExceptions);
      return result;
    } else if (expression.is(Kind.NAME)) {
      var name = (Name) expression;
      return List.of(name.name());
    } else if (expression.is(Kind.QUALIFIED_EXPR)) {
      var name = TreeUtils.tokens(expression)
        .stream()
        .map(Token::value)
        .collect(Collectors.joining());
      return List.of(name);
    }
    throw new IllegalArgumentException("Unsupported kind of tree element: " + expression.getKind().name());
  }

  private static void addQuickFix(PreciseIssue issue, Expression expression) {
    expression = Objects.requireNonNullElse((Expression) TreeUtils.firstAncestorOfKind(expression, Kind.PARENTHESIZED), expression);

    List<String> names;
    try {
      names = collectNames(expression);
    } catch (IllegalArgumentException e) {
      // expression contains subexpressions that are out of scope for quick fixing
      return;
    }

    var text = names.stream()
      .collect(Collectors.joining(", ", "(", ")"));

    var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.replace(expression, text))
      .build();

    issue.addQuickFix(quickFix);


  }

}
