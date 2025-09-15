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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7608")
public class AwsExpectedBucketOwnerCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add the 'ExpectedBucketOwner' parameter to verify S3 bucket ownership.";
  private static final String MESSAGE_EXTRA_ARGS = "Add the 'ExpectedBucketOwner' to the 'ExtraArgs' parameter to verify S3 bucket ownership.";
  private static final List<String> S3_CLIENT_FQN_PREFIX = List.of("botocore.client.BaseClient", "aiobotocore.client.AioBaseClient");
  private static final List<String> S3_METHODS_REQUIRING_EXPECTED_BUCKET_OWNER = List.of(
    "copy_object",
    "create_bucket_metadata_configuration",
    "create_bucket_metadata_table_configuration",
    "create_multipart_upload",
    "delete_bucket",
    "delete_bucket_analytics_configuration",
    "delete_bucket_cors",
    "delete_bucket_encryption",
    "delete_bucket_intelligent_tiering_configuration",
    "delete_bucket_inventory_configuration",
    "delete_bucket_lifecycle",
    "delete_bucket_metadata_configuration",
    "delete_bucket_metadata_table_configuration",
    "delete_bucket_metrics_configuration",
    "delete_bucket_ownership_controls",
    "delete_bucket_policy",
    "delete_bucket_replication",
    "delete_bucket_tagging",
    "delete_bucket_website",
    "delete_object",
    "delete_object_tagging",
    "delete_objects",
    "delete_public_access_block",
    "get_bucket_accelerate_configuration",
    "get_bucket_acl",
    "get_bucket_analytics_configuration",
    "get_bucket_cors",
    "get_bucket_encryption",
    "get_bucket_intelligent_tiering_configuration",
    "get_bucket_inventory_configuration",
    "get_bucket_lifecycle",
    "get_bucket_lifecycle_configuration",
    "get_bucket_location",
    "get_bucket_logging",
    "get_bucket_metadata_configuration",
    "get_bucket_metadata_table_configuration",
    "get_bucket_metrics_configuration",
    "get_bucket_notification",
    "get_bucket_notification_configuration",
    "get_bucket_ownership_controls",
    "get_bucket_policy",
    "get_bucket_policy_status",
    "get_bucket_replication",
    "get_bucket_request_payment",
    "get_bucket_tagging",
    "get_bucket_versioning",
    "get_bucket_website",
    "get_object",
    "get_object_acl",
    "get_object_attributes",
    "get_object_legal_hold",
    "get_object_lock_configuration",
    "get_object_retention",
    "get_object_tagging",
    "get_object_torrent",
    "get_public_access_block",
    "head_bucket",
    "head_object",
    "list_bucket_analytics_configurations",
    "list_bucket_intelligent_tiering_configurations",
    "list_bucket_inventory_configurations",
    "list_bucket_metrics_configurations",
    "list_multipart_uploads",
    "list_object_versions",
    "list_objects",
    "list_objects_v2",
    "list_parts",
    "put_bucket_accelerate_configuration",
    "put_bucket_acl",
    "put_bucket_analytics_configuration",
    "put_bucket_cors",
    "put_bucket_encryption",
    "put_bucket_intelligent_tiering_configuration",
    "put_bucket_inventory_configuration",
    "put_bucket_lifecycle",
    "put_bucket_lifecycle_configuration",
    "put_bucket_logging",
    "put_bucket_metrics_configuration",
    "put_bucket_notification",
    "put_bucket_notification_configuration",
    "put_bucket_ownership_controls",
    "put_bucket_policy",
    "put_bucket_replication",
    "put_bucket_request_payment",
    "put_bucket_tagging",
    "put_bucket_versioning",
    "put_bucket_website",
    "put_object",
    "put_object_acl",
    "put_object_legal_hold",
    "put_object_lock_configuration",
    "put_object_retention",
    "put_object_tagging",
    "put_public_access_block",
    "rename_object",
    "restore_object",
    "select_object_content",
    "update_bucket_metadata_inventory_table_configuration",
    "update_bucket_metadata_journal_table_configuration",
    "upload_part",
    "upload_part_copy");

  private static final List<String> UPLOAD_DOWNLOAD_FILE_FQN = List.of(
    "upload_file",
    "upload_fileobj",
    "download_file",
    "download_fileobj");

  private static final List<String> FQN_TO_CHECK = S3_CLIENT_FQN_PREFIX.stream()
    .flatMap(prefix -> S3_METHODS_REQUIRING_EXPECTED_BUCKET_OWNER.stream().map(method -> prefix + "." + method))
    .toList();

  private static final List<String> UPLOAD_DOWNLOAD_FQN_TO_CHECK = S3_CLIENT_FQN_PREFIX.stream()
    .flatMap(prefix -> UPLOAD_DOWNLOAD_FILE_FQN.stream().map(method -> prefix + "." + method))
    .toList();

  private TypeCheckMap<Object> isS3MethodCall;
  private TypeCheckMap<Object> isUploadDownloadFileMethodCall;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::setupTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }

  private void setupTypeChecker(SubscriptionContext ctx) {
    Object object = new Object();
    isS3MethodCall = new TypeCheckMap<>();
    isUploadDownloadFileMethodCall = new TypeCheckMap<>();

    for (var fqn : FQN_TO_CHECK) {
      var typeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      isS3MethodCall.put(typeChecker, object);
    }
    for (var fqn : UPLOAD_DOWNLOAD_FQN_TO_CHECK) {
      var typeChecker = ctx.typeChecker().typeCheckBuilder().isTypeWithFqn(fqn);
      isUploadDownloadFileMethodCall.put(typeChecker, object);
    }
  }

  private void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    PythonType type = TreeUtils.inferSingleAssignedExpressionType(callExpression.callee());

    if(isUploadDownloadFileMethodCallWithoutExtraArgs(type, callExpression)){
      ctx.addIssue(callExpression.callee(), MESSAGE_EXTRA_ARGS);
      return;
    }

    if (isS3MethodCall.getOptionalForType(type).isEmpty()) {
      return;
    }

    if(hasArgsOrKwargsParams(callExpression)){
      return;
    }

    if (hasExpectedBucketOwnerParameter(callExpression)) {
      return;
    }

    ctx.addIssue(callExpression.callee(), MESSAGE);
  }

  private boolean isUploadDownloadFileMethodCallWithoutExtraArgs(PythonType type, CallExpression callExpression){
    return isUploadDownloadFileMethodCall.getOptionalForType(type)
      .filter(obj -> !hasExtraArgsParameter(callExpression)).isPresent();
  }

  private static boolean hasExtraArgsParameter(CallExpression callExpression){
    return TreeUtils.argumentByKeyword("ExtraArgs", callExpression.arguments()) != null;
  }

  private static boolean hasArgsOrKwargsParams(CallExpression callExpression){
    return callExpression.arguments().stream().anyMatch(arg -> arg.is(Tree.Kind.UNPACKING_EXPR));
  }


  private static boolean hasExpectedBucketOwnerParameter(CallExpression callExpression) {
    return TreeUtils.argumentByKeyword("ExpectedBucketOwner", callExpression.arguments()) != null;
  }
}
