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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2876")
public class IterMethodReturnTypeCheck extends PythonSubscriptionCheck {
  private static final String INVALID_RETURN_VALUE_MESSAGE = "Return an object complying with iterator protocol.";
  private static final String NO_RETURN_STMTS_MESSAGE = INVALID_RETURN_VALUE_MESSAGE
    + " Consider explicitly raising a NotImplementedError if this class is not (yet) meant to support this method.";
  private static final String COROUTINE_METHOD_MESSAGE = INVALID_RETURN_VALUE_MESSAGE + " The method can not be a coroutine and have the `async` keyword.";

  private static final List<String> ABC_ABSTRACTMETHOD_DECORATORS = List.of("abstractmethod", "abc.abstractmethod");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> checkFunctionDefinition(ctx, (FunctionDef) ctx.syntaxNode()));
  }

  private static void checkFunctionDefinition(SubscriptionContext ctx, FunctionDef funDef) {
    if (!funDef.isMethodDefinition()) {
      return;
    }

    String funNameString = funDef.name().name();
    if (!"__iter__".equals(funNameString)) {
      return;
    }

    checkForAsync(ctx, funDef);

    var returnStmtCollector = collectReturnStmts(funDef);

    // // If there are yield keywords, then the method always returns a generator that supports the iterator protocol
    if (returnStmtCollector.containsYield()) {
      return;
    }

    List<ReturnStatement> returnStmts = returnStmtCollector.getReturnStmts();
    // If there are no return statements, we trigger this rule since this effectively means that the method is returning `None`.
    // However, there are exceptions to this:
    // * if the method raises exceptions, then it likely does not return a value on purpose
    // * if the method is marked as abstract, then it is likely not implemented on purpose
    if (returnStmts.isEmpty() &&
      !returnStmtCollector.raisesExceptions() &&
      !isAbstract(funDef)) {
      ctx.addIssue(funDef.defKeyword(), funDef.colon(), NO_RETURN_STMTS_MESSAGE);
      return;
    }

    for (ReturnStatement returnStmt : returnStmts) {
      checkReturnStmt(ctx, returnStmt);
    }
  }

  private static void checkForAsync(SubscriptionContext ctx, FunctionDef funDef) {
    Token asyncKeyword = funDef.asyncKeyword();
    if (asyncKeyword != null) {
      ctx.addIssue(asyncKeyword, COROUTINE_METHOD_MESSAGE);
    }
  }

  private static void checkReturnStmt(SubscriptionContext ctx, ReturnStatement returnStmt) {
    List<Expression> returnedExpressions = returnStmt.expressions();
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStmt.returnKeyword(), INVALID_RETURN_VALUE_MESSAGE);
      return;
    }

    InferredType returnStmtType = returnStmt.returnValueType();
    if (!returnStmtType.canHaveMember("__iter__") ||
      !returnStmtType.canHaveMember("__next__")) {
      addIssueOnReturnedExpressions(ctx, returnStmt, INVALID_RETURN_VALUE_MESSAGE);
    }
  }

  private static boolean isAbstract(FunctionDef funDef) {
    return funDef
      .decorators()
      .stream()
      .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
      .anyMatch(foundDeco -> ABC_ABSTRACTMETHOD_DECORATORS.stream().anyMatch(abcDeco -> abcDeco.equals(foundDeco)));
  }

  private static void addIssueOnReturnedExpressions(SubscriptionContext ctx, ReturnStatement returnStatement, String message) {
    List<Expression> returnedExpressions = returnStatement.expressions();

    // Not strictly necessary as this method currently is never called for an empty expression list.
    // Still, it should be well-behaved if it is ever used in a different context.
    if (returnedExpressions.isEmpty()) {
      ctx.addIssue(returnStatement.returnKeyword(), message);
    } else {
      Token firstExpressionToken = returnedExpressions.get(0).firstToken();
      Token lastExpressionToken = returnedExpressions.get(returnedExpressions.size() - 1).lastToken();

      ctx.addIssue(firstExpressionToken, lastExpressionToken, message);
    }
  }

  private static ReturnStmtCollector collectReturnStmts(FunctionDef funDef) {
    ReturnStmtCollector collector = new ReturnStmtCollector();
    funDef.body().accept(collector);

    return collector;
  }

  private static class ReturnStmtCollector extends BaseTreeVisitor {
    private final List<ReturnStatement> returnStmts = new ArrayList<>();
    private boolean containsYield = false;
    private boolean raisesExceptions = false;

    public List<ReturnStatement> getReturnStmts() {
      return returnStmts;
    }

    public boolean containsYield() {
      return containsYield;
    }

    public boolean raisesExceptions() {
      return raisesExceptions;
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStmt) {
      returnStmts.add(returnStmt);
    }

    @Override
    public void visitFunctionDef(FunctionDef funDef) {
      // We do not visit nested function definitions as they may contain irrelevant return statements or yield statements
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStmt) {
      containsYield = true;
    }

    @Override
    public void visitYieldExpression(YieldExpression yieldExpr) {
      containsYield = true;
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpr) {
      // We do not visit nested lambda definitions as they may contain irrelevant yield expressions
    }

    @Override
    public void visitRaiseStatement(RaiseStatement raiseStmt) {
      raisesExceptions = true;
    }
  }
}
