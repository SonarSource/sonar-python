/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5685")
public class ConfusingWalrusCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an assignment statement (\"=\") instead; \":=\" operator is confusing in this context.";
  private static final String MOVE_MESSAGE = "Move this assignment out of the %s; \":=\" operator is confusing in this context.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, this::checkAssignmentExpression);

    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
      StringElement stringElement = (StringElement) ctx.syntaxNode();
      for (Expression expression : stringElement.interpolatedExpressions()) {
        Expression nested = Expressions.removeParentheses(expression);
        if (nested.is(Tree.Kind.ASSIGNMENT_EXPRESSION)) {
          ctx.addIssue(nested, String.format(MOVE_MESSAGE, "f-string"));
        }
      }
    });
  }

  private void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Optional<Tree> parentTree = Optional.ofNullable(TreeUtils.firstAncestor(assignmentExpression, a -> !a.is(Tree.Kind.PARENTHESIZED)));
    parentTree.ifPresent(parent -> {
      if (parent.is(Tree.Kind.ASSIGNMENT_STMT)) {
        ctx.addIssue(assignmentExpression, MESSAGE);
      }
      if (parent.is(Tree.Kind.PARAMETER_TYPE_ANNOTATION)) {
        ctx.addIssue(assignmentExpression, String.format(MOVE_MESSAGE, "function definition"));
      }
      if (parent.is(Tree.Kind.PARAMETER)) {
        ctx.addIssue(assignmentExpression, String.format(MOVE_MESSAGE, "function definition"));
      }
      if (parent.is(Tree.Kind.REGULAR_ARGUMENT) && isInCallExprWithKeywordArguments(((RegularArgument) parent))) {
        ctx.addIssue(assignmentExpression, String.format(MOVE_MESSAGE, "call expression"));
      }
      if (parent.is(Tree.Kind.EXPRESSION_STMT)) {
        ctx.addIssue(assignmentExpression, MESSAGE);
      }
    });
  }

  private static boolean isInCallExprWithKeywordArguments(RegularArgument regularArgument) {
    ArgList argList = (ArgList) TreeUtils.firstAncestorOfKind(regularArgument, Tree.Kind.ARG_LIST);
    for (Argument argument : argList.arguments()) {
      if (argument.is(Tree.Kind.REGULAR_ARGUMENT) && ((RegularArgument) argument).keywordArgument() != null) {
        return true;
      }
    }
    return false;
  }
}
