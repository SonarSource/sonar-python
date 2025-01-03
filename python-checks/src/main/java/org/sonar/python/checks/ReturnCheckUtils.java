/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.types.DeclaredType;

public class ReturnCheckUtils {
  private ReturnCheckUtils() {
  }

  /**
   * Calls {@code ctx.addIssue} for a return statement such that...
   *
   * ...all returned expressions are marked as the source of the issue if the return statement contains such expressions
   * ...the return keyword is marked as the source of the issue if the return statement does not contain any expressions
   */
  static void addIssueOnReturnedExpressions(SubscriptionContext ctx, ReturnStatement returnStatement, String message) {
    List<Expression> returnedExpressions = returnStatement.expressions();

    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStatement.returnKeyword(), message);
    } else {
      Token firstExpressionToken = returnedExpressions.get(0).firstToken();
      Token lastExpressionToken = returnedExpressions.get(returnedExpressions.size() - 1).lastToken();

      ctx.addIssue(firstExpressionToken, lastExpressionToken, message);
    }
  }

  static void addIssueIfAsync(SubscriptionContext ctx, FunctionDef funDef, String message) {
    var asyncKeyword = funDef.asyncKeyword();
    if (asyncKeyword != null) {
      ctx.addIssue(asyncKeyword, message);
    }
  }

  static class ReturnStmtCollector extends BaseTreeVisitor {
    private final List<ReturnStatement> returnStmts = new ArrayList<>();
    private final List<Token> yieldKeywords = new ArrayList<>();
    private boolean raisesExceptions = false;

    private static final List<String> BOTTOM_TYPES = List.of("typing.NoReturn", "typing.Never");

    private ReturnStmtCollector() {
    }

    static ReturnStmtCollector collect(FunctionDef funDef) {
      var returnStmtCollector = new ReturnStmtCollector();
      funDef.body().accept(returnStmtCollector);

      return returnStmtCollector;
    }

    public List<ReturnStatement> getReturnStmts() {
      return returnStmts;
    }

    public List<Token> getYieldKeywords() {
      return yieldKeywords;
    }

    public boolean containsYield() {
      return !yieldKeywords.isEmpty();
    }

    /**
     * Users often raise a TypeError or NotImplementedError inside special methods to explicitly indicate that a method is not supported.
     * For example, list objects are unhashable, i.e. the __hash__() method raises a TypeError:
     *
     * <pre>
     * >>> hash([])
     * Traceback (most recent call last):
     *   File "<stdin>", line 1, in <module>
     * TypeError: unhashable type: 'list'
     * </pre>
     *
     * Hence, in order to avoid too many FPs when checking method return types, rules usually should not be triggered on methods that
     * contain no return statements if they do raise exceptions.
     */
    public boolean raisesExceptions() {
      return raisesExceptions;
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStmt) {
      returnStmts.add(returnStmt);
    }

    @Override
    public void visitFunctionDef(FunctionDef funDef) {
      // We do not visit nested function definitions as they may contain irrelevant return statements
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStmt) {
      yieldKeywords.add(yieldStmt.yieldExpression().yieldKeyword());
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpr) {
      yieldKeywords.add(yieldExpr.yieldKeyword());
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpr) {
      // We do not visit nested lambda definitions as they may contain irrelevant yield expressions
    }

    @Override
    public void visitRaiseStatement(RaiseStatement raiseStmt) {
      raisesExceptions = true;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      var returnType = callExpression.type();
      if (!(returnType instanceof DeclaredType)) {
        return;
      }

      var fullyQualifiedTypeName = ((DeclaredType) returnType).getTypeClass().fullyQualifiedName();

      // We can not check for bottom type annotations with mustBeOrExtend, because the type analysis does not fully resolve declared
      // typing.NoReturn and typing.Never types.
      raisesExceptions |= BOTTOM_TYPES.stream().anyMatch(bottomType -> bottomType.equals(fullyQualifiedTypeName));
    }
  }
}
