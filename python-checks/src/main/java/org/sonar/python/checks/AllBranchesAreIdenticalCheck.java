/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S3923")
public class AllBranchesAreIdenticalCheck extends PythonSubscriptionCheck {

  private static final String CONDITIONAL_EXP_MSG = "This conditional operation returns the same value whether the condition is \"true\" or \"false\".";

  @Override
  public void initialize(Context context) {
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
    ctx.addIssue(ifStmt.keyword(), "Remove this if statement or edit its code blocks so that they're not all the same.");
  }

  private static void handleConditionalExpression(ConditionalExpression conditionalExpression, SubscriptionContext ctx) {
    if (CheckUtils.areEquivalent(conditionalExpression.trueExpression(), conditionalExpression.falseExpression())) {
      ctx.addIssue(conditionalExpression.ifKeyword(), CONDITIONAL_EXP_MSG);
    }
    if (conditionalExpression.falseExpression().is(Tree.Kind.CONDITIONAL_EXPR) && nestedConditionalExpressionsIdentical(conditionalExpression.trueExpression(),
      (ConditionalExpression) conditionalExpression.falseExpression())) {
      ctx.addIssue(conditionalExpression.ifKeyword(), CONDITIONAL_EXP_MSG);
    }
  }

  private static boolean nestedConditionalExpressionsIdentical(Expression trueExpression, ConditionalExpression conditionalExpression) {
    if (!CheckUtils.areEquivalent(trueExpression, conditionalExpression.trueExpression())) {
      return false;
    }
    if (CheckUtils.areEquivalent(trueExpression, conditionalExpression.falseExpression())) {
      return true;
    }
    if (conditionalExpression.falseExpression().is(Tree.Kind.CONDITIONAL_EXPR)) {
      return nestedConditionalExpressionsIdentical(trueExpression, (ConditionalExpression) conditionalExpression.falseExpression());
    }
    return false;
  }
}
