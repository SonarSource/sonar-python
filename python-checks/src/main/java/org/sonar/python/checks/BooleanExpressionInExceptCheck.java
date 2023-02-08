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

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;

@Rule(key = "S5714")
public class BooleanExpressionInExceptCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Rewrite this \"except\" expression as a tuple of exception classes.";
  public static final String QUICK_FIX_MESSAGE = "Replace boolean expression with tuple";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.EXCEPT_CLAUSE, BooleanExpressionInExceptCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(Kind.EXCEPT_GROUP_CLAUSE, BooleanExpressionInExceptCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    ExceptClause except = (ExceptClause) ctx.syntaxNode();
    Expression exception = Expressions.removeParentheses(except.exception());
    if (exception != null && exception.is(Kind.OR, Kind.AND)) {
      var issue = (IssueWithQuickFix) ctx.addIssue(exception, MESSAGE);
      var quickFixEdits = getQuickFixEdits(exception);
      issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE).addTextEdit(quickFixEdits).build());
    }
  }

  private static List<PythonTextEdit> getQuickFixEdits(Tree expression) {
    return IntStream.range(0, expression.children().size())
      .mapToObj(childIndex -> {
        var child = expression.children().get(childIndex);
        if (child.is(Kind.TOKEN)) {
          var previous = expression.children().get(childIndex - 1);
          var previousLastChildren = previous.children().get(previous.children().size() -1);
          var next = expression.children().get(childIndex + 1);
          return Stream.of(
            PythonTextEdit.insertAfter(previousLastChildren, ","),
            PythonTextEdit.removeUntil(child, next)
          );
        } else if (child.is(Kind.OR, Kind.AND)) {
          return getQuickFixEdits(child).stream();
        } else if (child.is(Kind.PARENTHESIZED)) {
          return getQuickFixEdits(((ParenthesizedExpression) child).expression()).stream();
        }
        return Stream.empty();
      })
      .flatMap(Function.identity())
      .map(PythonTextEdit.class::cast)
      .collect(Collectors.toList());
  }

}
