/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7931")
public class NotImplementedBooleanContextCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "NotImplemented should not be used in boolean contexts.";

  private boolean isPython314OrGreater = false;
  private TypeCheckBuilder isBoolCallCheck;
  private TypeCheckBuilder isNotImplementedCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, this::initializeState);
    context.registerSyntaxNodeConsumer(Kind.IF_STMT, ctx -> raiseIfIsNotImplemented(ctx, ((IfStatement) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Kind.COMP_IF, ctx -> raiseIfIsNotImplemented(ctx, ((ComprehensionIf) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Kind.CONDITIONAL_EXPR, ctx -> raiseIfIsNotImplemented(ctx, ((ConditionalExpression) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Kind.AND, this::checkBooleanOperation);
    context.registerSyntaxNodeConsumer(Kind.WHILE_STMT, ctx -> raiseIfIsNotImplemented(ctx, ((WhileStatement) ctx.syntaxNode()).condition()));
    context.registerSyntaxNodeConsumer(Kind.IS, this::checkIsExpression);
    context.registerSyntaxNodeConsumer(Kind.OR, this::checkBooleanOperation);
    context.registerSyntaxNodeConsumer(Kind.NOT, ctx -> raiseIfIsNotImplemented(ctx, ((UnaryExpression) ctx.syntaxNode()).expression()));
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, this::checkBoolCall);
  }

  private void initializeState(SubscriptionContext ctx) {
    isBoolCallCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("bool");
    isNotImplementedCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("_NotImplementedType");
    isPython314OrGreater = PythonVersionUtils.areSourcePythonVersionsGreaterOrEqualThan(ctx.sourcePythonVersions(), PythonVersionUtils.Version.V_314);
  }

  private void checkIsExpression(SubscriptionContext ctx) {
    IsExpression isExpression = (IsExpression) ctx.syntaxNode();
    if (isBooleanLiteral(isExpression.leftOperand()) || isBooleanLiteral(isExpression.rightOperand())) {
      raiseIfIsNotImplemented(ctx, isExpression.leftOperand());
      raiseIfIsNotImplemented(ctx, isExpression.rightOperand());
    }
  }

  private void raiseIfIsNotImplemented(SubscriptionContext ctx, Expression expression) {
    if (isPython314OrGreater && expression instanceof Name && isNotImplementedCheck.check(expression.typeV2()).isTrue()) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static boolean isBooleanLiteral(Expression expression) {
    return TreeUtils.isBooleanLiteral(expression);
  }

  private void checkBooleanOperation(SubscriptionContext ctx) {
    BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
    raiseIfIsNotImplemented(ctx, binaryExpression.leftOperand());
    raiseIfIsNotImplemented(ctx, binaryExpression.rightOperand());
  }

  private void checkBoolCall(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();
    if (isBoolCallCheck.check(callExpr.callee().typeV2()).isTrue() && callExpr.arguments().size() == 1 && callExpr.arguments().get(0) instanceof RegularArgument regularArgument) {
      raiseIfIsNotImplemented(ctx, regularArgument.expression());
    }
  }

}
