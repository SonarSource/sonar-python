/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.cdk;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6265")
public class S3BucketGrantedAccessCheck extends AbstractS3BucketCheck {

  private static final String S3_BUCKET_DEPLOYMENT_FQN = "aws_cdk.aws_s3_deployment.BucketDeployment";
  private static final List<String> S3_BUCKET_FQNS = Arrays.asList(S3_BUCKET_FQN, S3_BUCKET_DEPLOYMENT_FQN);
  private boolean isAwsCdkImported = false;
  private static final String S3_BUCKET_PRIVATE_ACCESS_POLICY = "aws_cdk.aws_s3.BucketAccessControl.PRIVATE";
  public static final String MESSAGE = "Make sure granting access to [AllUsers|AuthenticatedUsers] group is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> isAwsCdkImported = false);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, this::checkAWSImport);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  private void checkAWSImport(SubscriptionContext ctx) {
    ImportFrom imports = (ImportFrom) ctx.syntaxNode();
    DottedName moduleName = imports.module();
    if (moduleName != null && moduleName.names().stream().map(Name::name).anyMatch("aws_cdk"::equals)) {
      isAwsCdkImported = true;
    }
  }

  @Override
  protected void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional<Symbol> symbol = Optional.ofNullable(node.calleeSymbol());

    symbol
      .map(Symbol::fullyQualifiedName)
      .filter(S3_BUCKET_FQNS::contains)
      .ifPresent(s -> visitBucketConstructor(ctx, node));

    if (isAwsCdkImported) {
      symbol
        .map(Symbol::name)
        .filter("grant_public_access"::equals)
        .ifPresent(s -> ctx.addIssue(node.callee(), MESSAGE));
    }
  }

  @Override
  void visitBucketConstructor(SubscriptionContext ctx, CallExpression bucket) {
    getArgument(ctx, bucket, "access_control")
      .ifPresent(argument -> argument.addIssueIf(this::isNotPrivate, MESSAGE));
  }

  protected boolean isNotPrivate(Expression expression) {
    return Optional.ofNullable(expression)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::symbol)
      .map(Symbol::fullyQualifiedName)
      .map(s -> !S3_BUCKET_PRIVATE_ACCESS_POLICY.equals(s))
      .orElse(false);
  }

}
