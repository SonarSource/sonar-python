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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WhileStatement;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7621")
public class AwsWaitersInsteadOfCustomPollingCheck extends PythonSubscriptionCheck {

  private static final Set<String> NON_WAITERS_BOTO3_METHODS = Set.of(
    // EC2
    "botocore.client.BaseClient.describe_instances",
    "botocore.client.BaseClient.describe_instance_status",
    "botocore.client.BaseClient.describe_volumes",
    "botocore.client.BaseClient.describe_snapshots",
    "botocore.client.BaseClient.describe_images",
    "botocore.client.BaseClient.describe_vpcs",
    "botocore.client.BaseClient.describe_subnets",
    "botocore.client.BaseClient.describe_nat_gateways",
    "botocore.client.BaseClient.describe_key_pairs",
    "botocore.client.BaseClient.get_password_data",
    // S3
    "botocore.client.BaseClient.head_bucket",
    "botocore.client.BaseClient.head_object",
    // RDS
    "botocore.client.BaseClient.describe_db_instances",
    "botocore.client.BaseClient.describe_db_clusters",
    "botocore.client.BaseClient.describe_db_snapshots",
    // DynamoDB
    "botocore.client.BaseClient.describe_table",
    // ECS
    "botocore.client.BaseClient.describe_services",
    "botocore.client.BaseClient.describe_tasks",
    // EKS
    "botocore.client.BaseClient.describe_cluster",
    "botocore.client.BaseClient.describe_nodegroup",
    // CloudFormation
    "botocore.client.BaseClient.describe_stacks",
    "botocore.client.BaseClient.describe_change_set",
    // Lambda
    "botocore.client.BaseClient.get_function_configuration",
    "botocore.client.BaseClient.get_function"
  );

  private final TypeCheckMap<Boolean> nonWaitersBoto3MethodsTypeCheckMap = new TypeCheckMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initializeCheck(SubscriptionContext ctx) {
    NON_WAITERS_BOTO3_METHODS.stream()
      .map(fqn -> ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn))
      .forEach(check -> nonWaitersBoto3MethodsTypeCheckMap.put(check, true));
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    if (nonWaitersBoto3MethodsTypeCheckMap.containsForType(TreeUtils.inferSingleAssignedExpressionType(callExpression.callee()))
        && TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.WHILE_STMT) instanceof WhileStatement whileStatement
        && Expressions.isTruthy(whileStatement.condition())
        && TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF) instanceof FunctionDef functionDef
        && AwsLambdaChecksUtils.isLambdaHandler(ctx, functionDef)
    ) {
      ctx.addIssue(callExpression, "Use AWS waiters instead of custom polling loops.");
    }
  }


}
