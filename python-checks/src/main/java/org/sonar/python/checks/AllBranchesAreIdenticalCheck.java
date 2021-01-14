/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S3923")
public class AllBranchesAreIdenticalCheck extends PythonSubscriptionCheck {

  private static final List<ConditionalExpression> ignoreList = new ArrayList<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ignoreList.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> handleIfStatement((IfStatement) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Tree.Kind.CONDITIONAL_EXPR, ctx -> handleConditionalExpression((ConditionalExpression) ctx.syntaxNode(), ctx));
  }

  private static void handleIfStatement(IfStatement ifStmt, SubscriptionContext ctx) {
    if (ifStmt.elseBranch() == null) {
      return;
    }
    StatementList body = ifStmt.body();
    for (IfStatement elifBranch : ifStmt.elifBranches()) {
      StatementList elifBody = elifBranch.body();
      if (!CheckUtils.areEquivalent(body, elifBody)) {
        return;
      }
    }
    if (!CheckUtils.areEquivalent(body, ifStmt.elseBranch().body())) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(ifStmt.keyword(), "Remove this if statement or edit its code blocks so that they're not all the same.");
    issue.secondary(issueLocation(ifStmt.body()));
    ifStmt.elifBranches().forEach(e -> issue.secondary(issueLocation(e.body())));
    issue.secondary(issueLocation(ifStmt.elseBranch().body()));
  }

  private static IssueLocation issueLocation(StatementList body) {
    List<Token> tokens = TreeUtils.nonWhitespaceTokens(body);
    return IssueLocation.preciseLocation(tokens.get(0), tokens.get(tokens.size() - 1), null);
  }

  private static void handleConditionalExpression(ConditionalExpression conditionalExpression, SubscriptionContext ctx) {
    if (ignoreList.contains(conditionalExpression)) {
      return;
    }
    if (areIdentical(conditionalExpression.trueExpression(), conditionalExpression.falseExpression())) {
      PreciseIssue issue = ctx.addIssue(conditionalExpression.ifKeyword(), "This conditional expression returns the same value whether the condition is \"true\" or \"false\".");
      addSecondaryLocations(issue, conditionalExpression.trueExpression());
      addSecondaryLocations(issue, conditionalExpression.falseExpression());
    }
  }

  private static void addSecondaryLocations(PreciseIssue issue, Expression expression) {
    Expression unwrappedExpression = Expressions.removeParentheses(expression);
    if (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      ConditionalExpression conditionalExpression = (ConditionalExpression) unwrappedExpression;
      ignoreList.add(conditionalExpression);
      addSecondaryLocations(issue, conditionalExpression.trueExpression());
      addSecondaryLocations(issue, conditionalExpression.falseExpression());
    } else {
      issue.secondary(unwrappedExpression, null);
    }
  }

  private static boolean areIdentical(Expression trueExpression, Expression falseExpression) {
    Expression unwrappedTrueExpression = unwrapIdenticalExpressions(trueExpression);
    Expression unwrappedFalseExpression = unwrapIdenticalExpressions(falseExpression);
    return CheckUtils.areEquivalent(unwrappedTrueExpression, unwrappedFalseExpression);
  }

  private static Expression unwrapIdenticalExpressions(Expression expression) {
    Expression unwrappedExpression = Expressions.removeParentheses(expression);
    if (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      boolean identicalExpressions = areIdentical(((ConditionalExpression) unwrappedExpression).trueExpression(), ((ConditionalExpression) unwrappedExpression).falseExpression());
      if (identicalExpressions) {
        while (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
          unwrappedExpression = Expressions.removeParentheses(((ConditionalExpression) unwrappedExpression).trueExpression());
        }
      }
    }
    return unwrappedExpression;
  }
}
