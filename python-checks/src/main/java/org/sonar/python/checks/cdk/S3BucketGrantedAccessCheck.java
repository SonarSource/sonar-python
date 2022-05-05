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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.AbstractS3BucketCheck.getArgument;

@Rule(key = "S6265")
public class S3BucketGrantedAccessCheck extends PythonSubscriptionCheck {

  private static final String S3_BUCKET_PRIVATE_ACCESS_POLICY = "aws_cdk.aws_s3.BucketAccessControl.PRIVATE";
  public static final String MESSAGE = "Make sure granting access to [AllUsers|AuthenticatedUsers] group is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  public void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(node.calleeSymbol())
      .filter(nodeSymbol -> ("aws_cdk.aws_s3.Bucket".equals(nodeSymbol.fullyQualifiedName()))
        || "aws_cdk.aws_s3_deployment.BucketDeployment".equals(nodeSymbol.fullyQualifiedName()))
      .ifPresent(nodeSymbol -> {
        Optional<AbstractS3BucketCheck.ArgumentTrace> accessParameter = getArgument(ctx, node, "access_control");// getAccessControlArgument(node.arguments());
        accessParameter.ifPresent(argument -> argument.addIssueIf(this::isNotPrivate, MESSAGE));
      });
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
