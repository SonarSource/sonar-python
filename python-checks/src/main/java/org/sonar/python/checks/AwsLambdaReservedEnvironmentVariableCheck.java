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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7617")
public class AwsLambdaReservedEnvironmentVariableCheck extends PythonSubscriptionCheck {

  private static final Set<String> AWS_RESERVED_ENVIRONMENT_VARIABLES = Set.of(
    "_HANDLER",
    "_X_AMZN_TRACE_ID",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_LAMBDA_FUNCTION_MEMORY_SIZE",
    "AWS_LAMBDA_FUNCTION_VERSION",
    "AWS_LAMBDA_INITIALIZATION_TYPE",
    "AWS_LAMBDA_LOG_GROUP_NAME",
    "AWS_LAMBDA_LOG_STREAM_NAME",
    "AWS_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_LAMBDA_RUNTIME_API",
    "LAMBDA_TASK_ROOT",
    "LAMBDA_RUNTIME_DIR"
  );

  private TypeCheckBuilder isOsEnvironTypeCheck;

  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
  }

  private void setupTypeChecker(SubscriptionContext ctx) {
    isOsEnvironTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn("os._Environ");
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();

    if (!isInAWSLambdaFunction(assignment, ctx)) {
      return;
    }

    ExpressionList lhs = assignment.lhsExpressions().get(0);
    for (Expression lhsExpression : lhs.expressions()) {
      checkIfOsEnvironVariableAssignedToReservedName(ctx, assignment, lhsExpression);
    }
  }

  private void checkIfOsEnvironVariableAssignedToReservedName(SubscriptionContext ctx, AssignmentStatement assignment, Expression lhsExpression) {
    if (!lhsExpression.is(Tree.Kind.SUBSCRIPTION)) {
      return;
    }
    SubscriptionExpression lhsSubscription = (SubscriptionExpression) lhsExpression;

    Expression object = lhsSubscription.object();
    if (!isOsEnvironTypeCheck.check(object.typeV2().unwrappedType()).equals(TriBool.TRUE)) {
      return;
    }

    String subscriptValue = getSubscriptString(lhsSubscription);
    if (subscriptValue != null && AWS_RESERVED_ENVIRONMENT_VARIABLES.contains(subscriptValue)) {
      ctx.addIssue(assignment, "Do not override reserved environment variable names in Lambda functions.");
    }
  }

  private static String getSubscriptString(SubscriptionExpression lhsSubscription) {
    String subscriptValue = null;
    Expression subscriptExpression = lhsSubscription.subscripts().expressions().get(0);
    if (subscriptExpression.is(Tree.Kind.STRING_LITERAL)) {
      subscriptValue = ((StringLiteral) subscriptExpression).trimmedQuotesValue();
    } else if (subscriptExpression.is(Tree.Kind.NAME)) {
      Name subscriptName = (Name) subscriptExpression;
      Expression singleAssignedValueExpression = Expressions.singleAssignedValue(subscriptName);
      if (singleAssignedValueExpression != null && singleAssignedValueExpression.is(Tree.Kind.STRING_LITERAL)) {
        subscriptValue = ((StringLiteral) singleAssignedValueExpression).trimmedQuotesValue();
      }
    }
    return subscriptValue;
  }

  private static boolean isInAWSLambdaFunction(AssignmentStatement statement, SubscriptionContext ctx) {
    Tree parentFunctionDef = TreeUtils.firstAncestorOfKind(statement.parent(), Tree.Kind.FUNCDEF);
    if(parentFunctionDef == null) {
      return false;
    }

    return AwsLambdaChecksUtils.isLambdaHandler(ctx, (FunctionDef) parentFunctionDef);
  }
}


