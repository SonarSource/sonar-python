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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.api.PythonPunctuator;

public abstract class IterationOnNonIterable extends PythonSubscriptionCheck {

  static final String SECONDARY_MESSAGE = "Definition of \"%s\".";
  static final String DEFAULT_SECONDARY_MESSAGE = "Type definition.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.UNPACKING_EXPR, this::checkUnpackingExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::checkForStatement);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, this::checkForComprehension);
    context.registerSyntaxNodeConsumer(Tree.Kind.YIELD_STMT, this::checkYieldStatement);
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
    ExpressionList expressionList = assignmentStatement.lhsExpressions().get(0);
    Map<LocationInFile, String> secondaries = new HashMap<>();
    if (isLhsIterable(expressionList) && !isValidIterable(assignmentStatement.assignedValue(), secondaries)) {
      reportIssue(ctx, assignmentStatement.assignedValue(), secondaries, message(assignmentStatement.assignedValue(), false));
    }
  }

  private static boolean isLhsIterable(ExpressionList expressionList) {
    if (expressionList.expressions().size() > 1) {
      return true;
    }
    Expression expression = expressionList.expressions().get(0);
    return expression.is(Tree.Kind.LIST_LITERAL) || expression.is(Tree.Kind.TUPLE);
  }

  private void checkForComprehension(SubscriptionContext ctx) {
    ComprehensionFor comprehensionFor = (ComprehensionFor) ctx.syntaxNode();
    Expression expression = comprehensionFor.iterable();
    Map<LocationInFile, String> secondaries = new HashMap<>();
    if (!isValidIterable(expression, secondaries)) {
      reportIssue(ctx, expression, secondaries, message(expression, false));
    }
  }

  private static void reportIssue(SubscriptionContext ctx, Expression expression, Map<LocationInFile, String> secondaries, String message) {
    PreciseIssue preciseIssue = ctx.addIssue(expression, message);
    secondaries.keySet().stream().filter(Objects::nonNull).forEach(location -> preciseIssue.secondary(location, secondaries.get(location)));
  }

  private void checkYieldStatement(SubscriptionContext ctx) {
    YieldStatement yieldStatement = (YieldStatement) ctx.syntaxNode();
    YieldExpression yieldExpression = yieldStatement.yieldExpression();
    if (yieldExpression.fromKeyword() == null) {
      return;
    }
    Expression expression = yieldExpression.expressions().get(0);
    Map<LocationInFile, String> secondaries = new HashMap<>();
    if (!isValidIterable(expression, secondaries)) {
      reportIssue(ctx, expression, secondaries, message(expression, false));
    }
  }

  private void checkUnpackingExpression(SubscriptionContext ctx) {
    UnpackingExpression unpackingExpression = (UnpackingExpression) ctx.syntaxNode();
    if (unpackingExpression.starToken().type().equals(PythonPunctuator.MUL_MUL)) {
      return;
    }
    Expression expression = unpackingExpression.expression();
    Map<LocationInFile, String> secondaries = new HashMap<>();
    if (!isValidIterable(expression, secondaries)) {
      reportIssue(ctx, expression, secondaries, message(expression, false));
    }
  }

  private void checkForStatement(SubscriptionContext ctx) {
    ForStatement forStatement = (ForStatement) ctx.syntaxNode();
    List<Expression> testExpressions = forStatement.testExpressions();
    boolean isAsync = forStatement.asyncKeyword() != null;
    if (testExpressions.size() > 1) {
      return;
    }
    Expression expression = testExpressions.get(0);
    Map<LocationInFile, String> secondaries = new HashMap<>();
    if (!isAsync && !isValidIterable(expression, secondaries)) {
      reportIssue(ctx, expression, secondaries, message(expression, true));
    }
  }

  abstract boolean isAsyncIterable(Expression expression);

  abstract boolean isValidIterable(Expression expression, Map<LocationInFile, String> secondaries);

  abstract String message(Expression expression, boolean isForLoop);
}
