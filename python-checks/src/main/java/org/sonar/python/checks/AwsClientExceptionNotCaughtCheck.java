/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7619")
public class AwsClientExceptionNotCaughtCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Wrap this AWS client call in a try-except block to handle \"botocore.exceptions.ClientError\".";
  private static final String SECONDARY_MESSAGE = "This try does not catch the ClientError.";

  private static final Set<String> EXCEPTION_THROWING_METHODS = Set.of(
      // S3 operations
      "botocore.client.BaseClient.get_object",
      "botocore.client.BaseClient.put_object",
      "botocore.client.BaseClient.create_bucket",
      "botocore.client.BaseClient.delete_object",
      "botocore.client.BaseClient.delete_bucket",
      "botocore.client.BaseClient.list_objects_v2",
      "botocore.client.BaseClient.copy_object",
      "botocore.client.BaseClient.head_object",
      "botocore.client.BaseClient.get_bucket_location",
      "botocore.client.BaseClient.put_bucket_policy",
      "botocore.client.BaseClient.get_bucket_policy",
      "botocore.client.BaseClient.delete_bucket_policy",

      // EC2 operations
      "botocore.client.BaseClient.describe_instances",
      "botocore.client.BaseClient.run_instances",
      "botocore.client.BaseClient.terminate_instances",
      "botocore.client.BaseClient.start_instances",
      "botocore.client.BaseClient.stop_instances",
      "botocore.client.BaseClient.create_security_group",
      "botocore.client.BaseClient.delete_security_group",
      "botocore.client.BaseClient.describe_security_groups",
      "botocore.client.BaseClient.create_vpc",
      "botocore.client.BaseClient.delete_vpc",
      "botocore.client.BaseClient.describe_vpcs",

      // Lambda operations
      "botocore.client.BaseClient.create_function",
      "botocore.client.BaseClient.update_function_code",
      "botocore.client.BaseClient.update_function_configuration",
      "botocore.client.BaseClient.delete_function",
      "botocore.client.BaseClient.invoke",
      "botocore.client.BaseClient.get_function",
      "botocore.client.BaseClient.list_functions",

      // DynamoDB operations
      "botocore.client.BaseClient.get_item",
      "botocore.client.BaseClient.put_item",
      "botocore.client.BaseClient.delete_item",
      "botocore.client.BaseClient.update_item",
      "botocore.client.BaseClient.query",
      "botocore.client.BaseClient.scan",
      "botocore.client.BaseClient.create_table",
      "botocore.client.BaseClient.delete_table",
      "botocore.client.BaseClient.describe_table",

      // RDS operations
      "botocore.client.BaseClient.create_db_instance",
      "botocore.client.BaseClient.delete_db_instance",
      "botocore.client.BaseClient.describe_db_instances",
      "botocore.client.BaseClient.modify_db_instance",
      "botocore.client.BaseClient.reboot_db_instance",

      // IAM operations
      "botocore.client.BaseClient.create_user",
      "botocore.client.BaseClient.delete_user",
      "botocore.client.BaseClient.get_user",
      "botocore.client.BaseClient.create_role",
      "botocore.client.BaseClient.delete_role",
      "botocore.client.BaseClient.get_role",
      "botocore.client.BaseClient.attach_user_policy",
      "botocore.client.BaseClient.detach_user_policy",

      // CloudFormation operations
      "botocore.client.BaseClient.create_stack",
      "botocore.client.BaseClient.delete_stack",
      "botocore.client.BaseClient.describe_stacks",
      "botocore.client.BaseClient.update_stack",

      // SNS operations
      "botocore.client.BaseClient.create_topic",
      "botocore.client.BaseClient.delete_topic",
      "botocore.client.BaseClient.publish",
      "botocore.client.BaseClient.subscribe",
      "botocore.client.BaseClient.unsubscribe",

      // SQS operations
      "botocore.client.BaseClient.create_queue",
      "botocore.client.BaseClient.delete_queue",
      "botocore.client.BaseClient.send_message",
      "botocore.client.BaseClient.receive_message",
      "botocore.client.BaseClient.delete_message");

  private TypeCheckMap<Object> exceptionThrowingMethodsTypeCheckMap;

  private static final Set<String> EXCEPTIONS = Set.of(
      "botocore.exceptions.ClientError",
      "Exception",
      "BaseException");

  private TypeCheckMap<Object> exceptionsTypeCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }

  private void initTypeChecks(SubscriptionContext ctx) {
    exceptionThrowingMethodsTypeCheckMap = new TypeCheckMap<>();
    var marker = new Object();
    EXCEPTION_THROWING_METHODS.forEach(fqn -> exceptionThrowingMethodsTypeCheckMap
        .put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn), marker));

    exceptionsTypeCheckMap = new TypeCheckMap<>();
    EXCEPTIONS.forEach(
        exception -> exceptionsTypeCheckMap.put(ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(exception), marker));
  }

  private void checkCall(SubscriptionContext ctx) {
    var call = (CallExpression) ctx.syntaxNode();

    FunctionDef functionDef = TreeUtils.firstAncestorOfClass(call, FunctionDef.class);

    if (functionDef == null || !AwsLambdaChecksUtils.isOnlyLambdaHandler(ctx, functionDef)) {
      return;
    }

    if (!isThrowingClientException(call)) {
      return;
    }

    TryStatement tryStmt = getEnclosingTryStatement(call);
    if (tryStmt == null) {
      ctx.addIssue(call.callee(), MESSAGE);
    } else if (!isWrappedInClientErrorHandler(tryStmt)) {
      ctx.addIssue(call.callee(), MESSAGE)
        .secondary(tryStmt.tryKeyword(), SECONDARY_MESSAGE);
    }
  }

  private boolean isThrowingClientException(CallExpression call) {
    return exceptionThrowingMethodsTypeCheckMap
        .containsForType(TreeUtils.inferSingleAssignedExpressionType(call.callee()));
  }

  @CheckForNull
  private static TryStatement getEnclosingTryStatement(Tree tree) {
    Tree potentiallyTryStmt = TreeUtils.firstAncestorOfKind(tree, Tree.Kind.TRY_STMT, Tree.Kind.FUNCDEF,
        Tree.Kind.LAMBDA);
    if (potentiallyTryStmt instanceof TryStatement tryStmt) {
      return tryStmt;
    }
    return null;
  }

  private boolean isWrappedInClientErrorHandler(@Nullable TryStatement tryStmt) {
    return tryStmt != null && (doesTryCatchClientError(tryStmt)
        // handles nested try-except statements
        || isWrappedInClientErrorHandler(getEnclosingTryStatement(tryStmt)));
  }

  private boolean doesTryCatchClientError(TryStatement tryStmt) {
    return tryStmt.exceptClauses()
        .stream()
        .anyMatch(this::doesExceptClauseCatchClientError);
  }

  private boolean doesExceptClauseCatchClientError(ExceptClause except) {
    Expression exception = except.exception();
    if (exception == null) {
      // bare except clause, e.g. `except:` -> catches all exceptions
      return true;
    }

    if (isClientErrorOrParent(exception)) {
      return true;
    }

    if (exception instanceof Tuple tuple) {
      return tuple.elements().stream().anyMatch(this::isClientErrorOrParent);
    }

    return false;
  }

  private boolean isClientErrorOrParent(Expression exceptionExpr) {
    return exceptionsTypeCheckMap.containsForType(exceptionExpr.typeV2());
  }
}
