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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5714")
public class BooleanExpressionInExceptCheck extends PythonSubscriptionCheck {

  private static final Logger LOG = Loggers.get(BooleanExpressionInExceptCheck.class);

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
        var issue = (IssueWithQuickFix) ctx.addIssue(exception, MESSAGE);
        addQuickFix(issue, exception);
      });
  }

  private static List<String> collectNames(Tree expression) {
    if (expression.is(Kind.OR, Kind.AND)) {
      var binaryExpression = (BinaryExpression) expression;
      var leftExceptions = collectNames(binaryExpression.leftOperand());
      var rightExceptions = collectNames(binaryExpression.rightOperand());
      var result = new ArrayList<String>();
      result.addAll(leftExceptions);
      result.addAll(rightExceptions);
      return result;
    } else if (expression.is(Kind.PARENTHESIZED)) {
      var parenthesizedExpression = (ParenthesizedExpression) expression;
      return collectNames(parenthesizedExpression.expression());
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

  private static void addQuickFix(IssueWithQuickFix issue, Tree expression) {
    expression = Objects.requireNonNullElse(TreeUtils.firstAncestorOfKind(expression, Kind.PARENTHESIZED), expression);

    List<String> names;
    try {
      names = collectNames(expression);
    } catch (IllegalArgumentException e) {
      LOG.error("Could not add quick fix", e);
      return;
    }

    var text = names.stream()
      .collect(Collectors.joining(", ", "(", ")"));

    var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(PythonTextEdit.replace(expression, text))
      .build();

    issue.addQuickFix(quickFix);


  }

}
