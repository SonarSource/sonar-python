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
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyDelStatementTree;
import org.sonar.python.api.tree.PyExceptClauseTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyReturnStatementTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.PyUnaryExpressionTree;
import org.sonar.python.api.tree.PyWhileStatementTree;
import org.sonar.python.api.tree.PyYieldExpressionTree;
import org.sonar.python.api.tree.Tree;

@Rule(key = "S1721")
public class UselessParenthesisAfterKeywordCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the parentheses after this \"%s\" keyword.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkExpr(((PyAssertStatementTree) ctx.syntaxNode()).condition(), ctx, "assert"));
    context.registerSyntaxNodeConsumer(Tree.Kind.DEL_STMT, ctx -> checkExpr(((PyDelStatementTree) ctx.syntaxNode()).expressions().get(0), ctx, "del"));
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> {
      PyIfStatementTree ifStmt = (PyIfStatementTree) ctx.syntaxNode();
      checkExpr(ifStmt.condition(), ctx, ifStmt.keyword().getValue());
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.WHILE_STMT, ctx -> {
      PyWhileStatementTree whileStmt = (PyWhileStatementTree) ctx.syntaxNode();
      checkExpr(whileStmt.condition(), ctx, whileStmt.whileKeyword().getValue());
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, ctx -> handleForStatement(ctx, (PyForStatementTree) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.RAISE_STMT, ctx -> handleRaiseStatement(ctx, (PyRaiseStatementTree) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, ctx -> handleReturnStatement(ctx, (PyReturnStatementTree) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.YIELD_EXPR, ctx -> handleYieldExpression(ctx, (PyYieldExpressionTree) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ctx -> {
      PyExpressionTree exception = ((PyExceptClauseTree) ctx.syntaxNode()).exception();
      if( exception != null) {
        checkExprExcludeTuples(exception, ctx, "except");
      }
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.NOT, ctx -> handleNotOperator(ctx, (PyUnaryExpressionTree) ctx.syntaxNode()));
  }

  private static void handleYieldExpression(SubscriptionContext ctx, PyYieldExpressionTree yieldExpr) {
    if (yieldExpr.fromKeyword() == null && yieldExpr.expressions().size() == 1) {
      yieldExpr.expressions().forEach(e -> checkExpr(e, ctx, "yield"));
    }
  }

  private static void handleReturnStatement(SubscriptionContext ctx, PyReturnStatementTree retStmt) {
    if (retStmt.expressions().size() == 1) {
      PyExpressionTree expr = retStmt.expressions().get(0);
      if ((expr.is(Tree.Kind.PARENTHESIZED) || (expr.is(Tree.Kind.TUPLE) && !((PyTupleTree) expr).elements().isEmpty()))
        && expr.firstToken().getLine() == expr.lastToken().getLine()) {
        ctx.addIssue(expr, String.format(MESSAGE, "return"));
      }
    }
  }

  private static void handleNotOperator(SubscriptionContext ctx, PyUnaryExpressionTree unary) {
    PyExpressionTree negatedExpr = unary.expression();
    if (negatedExpr.is(Tree.Kind.PARENTHESIZED)) {
      negatedExpr = ((PyParenthesizedExpressionTree) negatedExpr).expression();
      if (negatedExpr.is(Tree.Kind.COMPARISON) || !(negatedExpr instanceof PyBinaryExpressionTree)) {
        ctx.addIssue(negatedExpr, String.format(MESSAGE, "not"));
      }
    }
  }

  private static void handleRaiseStatement(SubscriptionContext ctx, PyRaiseStatementTree raiseStmt) {
    if (!raiseStmt.expressions().isEmpty()) {
      checkExpr(raiseStmt.expressions().get(0), ctx, "raise");
    }
  }

  private static void handleForStatement(SubscriptionContext ctx, PyForStatementTree forStmt) {
    if(forStmt.expressions().size() == 1) {
      checkExpr(forStmt.expressions().get(0), ctx, "for");
    }
    if(forStmt.testExpressions().size() == 1) {
      checkExpr(forStmt.testExpressions().get(0), ctx, "in");
    }
  }

  private static void checkExprExcludeTuples(PyExpressionTree expr, SubscriptionContext ctx, String keyword) {
    checkExpr(expr, ctx, keyword, false);
  }

  private static void checkExpr(PyExpressionTree expr, SubscriptionContext ctx, String keyword) {
    checkExpr(expr, ctx, keyword, true);
  }

  private static void checkExpr(PyExpressionTree expr, SubscriptionContext ctx, String keyword, boolean raiseForTuple) {
    if ((expr.is(Tree.Kind.PARENTHESIZED) || (raiseForTuple && expr.is(Tree.Kind.TUPLE)))
      && expr.firstToken().getLine() == expr.lastToken().getLine()) {
      ctx.addIssue(expr, String.format(MESSAGE, keyword));
    }
  }
}
