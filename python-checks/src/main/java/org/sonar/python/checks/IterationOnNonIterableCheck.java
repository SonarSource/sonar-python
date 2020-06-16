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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.api.PythonPunctuator;

@Rule(key = "S3862")
public class IterationOnNonIterableCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this expression with an iterable object.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.UNPACKING_EXPR, IterationOnNonIterableCheck::checkUnpackingExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, IterationOnNonIterableCheck::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, IterationOnNonIterableCheck::checkForStatement);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, IterationOnNonIterableCheck::checkForComprehension);
    context.registerSyntaxNodeConsumer(Tree.Kind.YIELD_STMT, IterationOnNonIterableCheck::checkYieldStatement);
  }

  private static void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
    ExpressionList expressionList = assignmentStatement.lhsExpressions().get(0);
    if (isLhsIterable(expressionList) && !isValidIterable(assignmentStatement.assignedValue())) {
      ctx.addIssue(assignmentStatement.assignedValue(), MESSAGE);
    }
  }

  private static boolean isLhsIterable(ExpressionList expressionList) {
    if (expressionList.expressions().size() > 1) {
      return true;
    }
    Expression expression = expressionList.expressions().get(0);
    return expression.is(Tree.Kind.LIST_LITERAL) || expression.is(Tree.Kind.TUPLE);
  }

  private static void checkForComprehension(SubscriptionContext ctx) {
    ComprehensionFor comprehensionFor = (ComprehensionFor) ctx.syntaxNode();
    Expression expression = comprehensionFor.iterable();
    if (!isValidIterable(expression)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static void checkYieldStatement(SubscriptionContext ctx) {
    YieldStatement yieldStatement = (YieldStatement) ctx.syntaxNode();
    YieldExpression yieldExpression = yieldStatement.yieldExpression();
    if (yieldExpression.fromKeyword() == null) {
      return;
    }
    Expression expression = yieldExpression.expressions().get(0);
    if (!isValidIterable(expression)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static void checkUnpackingExpression(SubscriptionContext ctx) {
    UnpackingExpression unpackingExpression = (UnpackingExpression) ctx.syntaxNode();
    if (unpackingExpression.starToken().type().equals(PythonPunctuator.MUL_MUL)) {
      return;
    }
    Expression expression = unpackingExpression.expression();
    if (!isValidIterable(expression)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static void checkForStatement(SubscriptionContext ctx) {
    ForStatement forStatement = (ForStatement) ctx.syntaxNode();
    List<Expression> testExpressions = forStatement.testExpressions();
    boolean isAsync = forStatement.asyncKeyword() != null;
    if (testExpressions.size() > 1) {
      return;
    }
    Expression expression = testExpressions.get(0);
    if (!isAsync && !isValidIterable(expression)) {
      String message = isAsyncIterable(expression) ? "Add \"async\" before \"for\"; This expression is an async generator." : MESSAGE;
      ctx.addIssue(expression, message);
    }
  }

  private static boolean isAsyncIterable(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && calleeSymbol.is(Symbol.Kind.FUNCTION)) {
        return ((FunctionSymbol) calleeSymbol).isAsynchronous();
      }
    }
    return expression.type().canHaveMember("__aiter__");
  }

  private static boolean isValidIterable(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && calleeSymbol.is(Symbol.Kind.FUNCTION) && ((FunctionSymbol) calleeSymbol).isAsynchronous()) {
        return false;
      }
    }
    if (expression instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) expression).symbol();
      if (symbol != null) {
        if (symbol.is(Symbol.Kind.FUNCTION)) {
          FunctionSymbol functionSymbol = (FunctionSymbol) symbol;
          return functionSymbol.hasDecorators();
        }
        if (symbol.is(Symbol.Kind.CLASS)) {
          return false;
        }
      }
    }
    return expression.type().canHaveMember("__iter__") || expression.type().canHaveMember("__getitem__");
  }
}
