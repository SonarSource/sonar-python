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
package org.sonar.python.checks.cdk;

import java.util.List;
import java.util.Optional;
import java.util.function.BiConsumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6265")
public class S3BucketGrantedAccessCheck extends AbstractS3BucketCheck {

  public static final String MESSAGE_POLICY = "Make sure granting %s access is safe here.";
  public static final String MESSAGE_GRANT = "Make sure allowing unrestricted access to objects from this bucket is safe here.";
  private static final String S3_BUCKET_DEPLOYMENT_FQN = "aws_cdk.aws_s3_deployment.BucketDeployment";
  private static final String S3_BUCKET_AUTHENTICATED_READ = "aws_cdk.aws_s3.BucketAccessControl.AUTHENTICATED_READ";
  private static final String S3_BUCKET_PUBLIC_READ = "aws_cdk.aws_s3.BucketAccessControl.PUBLIC_READ";
  private static final String S3_BUCKET_PUBLIC_READ_WRITE = "aws_cdk.aws_s3.BucketAccessControl.PUBLIC_READ_WRITE";
  private static final List<String> S3_BUCKET_FQNS = List.of(S3_BUCKET_FQN, S3_BUCKET_DEPLOYMENT_FQN);
  private static final List<String> S3_BUCKET_SENSITIVE_POLICIES = List.of(S3_BUCKET_AUTHENTICATED_READ, S3_BUCKET_PUBLIC_READ, S3_BUCKET_PUBLIC_READ_WRITE);

  private boolean isAwsCdkImported = false;

  @Override
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> isAwsCdkImported = false);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, this::checkAWSImport);
  }

  private void checkAWSImport(SubscriptionContext ctx) {
    ImportFrom imports = (ImportFrom) ctx.syntaxNode();
    Optional.ofNullable(imports.module())
      .filter(dottedName -> dottedName.names()
        .stream()
        .map(Name::name)
        .anyMatch("aws_cdk"::equals))
      .ifPresent(n -> isAwsCdkImported = true);
  }

  @Override
  protected void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional<Symbol> symbol = Optional.ofNullable(node.calleeSymbol());

    symbol
      .map(Symbol::fullyQualifiedName)
      .filter(S3_BUCKET_FQNS::contains)
      .ifPresent(s -> visitBucketConstructor().accept(ctx, node));

    if (isAwsCdkImported) {
      symbol
        .map(Symbol::name)
        .filter("grant_public_access"::equals)
        .ifPresent(s -> ctx.addIssue(node.callee(), MESSAGE_GRANT));
    }
  }

  @Override
  BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor() {
    return (ctx, bucket) -> getArgument(ctx, bucket, "access_control")
      .ifPresent(argument -> argument.addIssueIf(CdkPredicate.isFqnOf(S3_BUCKET_SENSITIVE_POLICIES), getSensitivePolicyMessage(argument)));
  }

  private static String getSensitivePolicyMessage(CdkUtils.ExpressionFlow flow){
    String attribute = ((QualifiedExpression) flow.locations().getLast()).name().name();
    return String.format(MESSAGE_POLICY, attribute);
  }

}
