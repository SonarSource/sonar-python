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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1721")
public class UselessParenthesisAfterKeywordCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove the parentheses after this \"%s\" keyword.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkExpr(((AssertStatement) ctx.syntaxNode()).condition(), ctx, "assert"));
    context.registerSyntaxNodeConsumer(Tree.Kind.DEL_STMT, ctx -> checkExpr(((DelStatement) ctx.syntaxNode()).expressions().get(0), ctx, "del"));
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> {
      IfStatement ifStmt = (IfStatement) ctx.syntaxNode();
      checkExpr(ifStmt.condition(), ctx, ifStmt.keyword().value());
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.WHILE_STMT, ctx -> {
      WhileStatement whileStmt = (WhileStatement) ctx.syntaxNode();
      checkExpr(whileStmt.condition(), ctx, whileStmt.whileKeyword().value());
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, ctx -> handleForStatement(ctx, (ForStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.RAISE_STMT, ctx -> handleRaiseStatement(ctx, (RaiseStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, ctx -> handleReturnStatement(ctx, (ReturnStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.YIELD_EXPR, ctx -> handleYieldExpression(ctx, (YieldExpression) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.EXCEPT_CLAUSE, ctx -> {
      Expression exception = ((ExceptClause) ctx.syntaxNode()).exception();
      if (exception != null) {
        checkExprExcludeTuples(exception, ctx, "except");
      }
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.NOT, ctx -> handleNotOperator(ctx, (UnaryExpression) ctx.syntaxNode()));
  }

  private static void handleYieldExpression(SubscriptionContext ctx, YieldExpression yieldExpr) {
    if (yieldExpr.fromKeyword() == null && yieldExpr.expressions().size() == 1) {
      yieldExpr.expressions().forEach(e -> checkExpr(e, ctx, "yield"));
    }
  }

  private static void handleReturnStatement(SubscriptionContext ctx, ReturnStatement retStmt) {
    if (retStmt.expressions().size() == 1) {
      Expression expr = retStmt.expressions().get(0);
      if ((isParenthesisWithoutAssignmentInside(expr) || isTupleWithMoreThanOneElement(expr))
        && expr.firstToken().line() == expr.lastToken().line()) {
        ctx.addIssue(expr, String.format(MESSAGE, "return"));
      }
    }
  }

  private static void handleNotOperator(SubscriptionContext ctx, UnaryExpression unary) {
    Expression negatedExpr = unary.expression();
    if (isParenthesisWithoutAssignmentInside(negatedExpr)) {
      negatedExpr = ((ParenthesizedExpression) negatedExpr).expression();
      if (negatedExpr.is(Tree.Kind.COMPARISON) || !(negatedExpr instanceof BinaryExpression)) {
        ctx.addIssue(negatedExpr, String.format(MESSAGE, "not"));
      }
    }
  }

  private static void handleRaiseStatement(SubscriptionContext ctx, RaiseStatement raiseStmt) {
    if (!raiseStmt.expressions().isEmpty()) {
      checkExpr(raiseStmt.expressions().get(0), ctx, "raise");
    }
  }

  private static void handleForStatement(SubscriptionContext ctx, ForStatement forStmt) {
    if (forStmt.expressions().size() == 1) {
      checkExpr(forStmt.expressions().get(0), ctx, "for");
    }
    if (forStmt.testExpressions().size() == 1) {
      checkExpr(forStmt.testExpressions().get(0), ctx, "in");
    }
  }

  private static void checkExprExcludeTuples(Expression expr, SubscriptionContext ctx, String keyword) {
    checkExpr(expr, ctx, keyword, false);
  }

  private static void checkExpr(Expression expr, SubscriptionContext ctx, String keyword) {
    checkExpr(expr, ctx, keyword, true);
  }

  private static void checkExpr(Expression expr, SubscriptionContext ctx, String keyword, boolean raiseForTuple) {
    if ((isParenthesisWithoutAssignmentInside(expr) || (raiseForTuple && isTupleWithMoreThanOneElement(expr)))
      && expr.firstToken().line() == expr.lastToken().line()) {
      ctx.addIssue(expr, String.format(MESSAGE, keyword));
    }
  }

  private static boolean isParenthesisWithoutAssignmentInside(Expression expr) {
    return expr.is(Tree.Kind.PARENTHESIZED) && !((ParenthesizedExpression) expr).expression().is(Tree.Kind.ASSIGNMENT_EXPRESSION);
  }

  private static boolean isTupleWithMoreThanOneElement(Expression expr) {
    return expr.is(Tree.Kind.TUPLE) && ((Tuple) expr).elements().size() > 1;
  }
}
