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
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.isCallExpression;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isTrue;
import static org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6329")
public class PublicNetworkAccessToCloudResourcesCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure allowing public network access is safe here.";
  private static final Set<String> SAFE_SUBNET_TYPES = Set.of("ISOLATED", "PRIVATE_ISOLATED", "PRIVATE", "PRIVATE_WITH_NAT");
  public static final String PUBLICLY_ACCESSIBLE_ARG_NAME = "publicly_accessible";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_dms.CfnReplicationInstance", (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresentOrElse(
        argument -> argument.addIssueIf(isTrue(), MESSAGE),
        () -> subscriptionContext.addIssue(callExpression, MESSAGE)
      )
    );

    checkFqn("aws_cdk.aws_rds.DatabaseInstance", PublicNetworkAccessToCloudResourcesCheck::checkDatabaseInstance);

    checkFqn("aws_cdk.aws_rds.CfnDBInstance", (subscriptionContext, callExpression) ->
      getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresent(argument -> {
        if (argument.hasExpression(isTrue()) && !isSafeNetwork(subscriptionContext, callExpression)) {
          argument.addIssue(MESSAGE);
        }
      }));
  }

  private static boolean isSafeNetwork(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    Optional<QualifiedExpression> subnetGroupName = getArgument(subscriptionContext, callExpression, "db_subnet_group_name")
      .flatMap(expressionFlow -> expressionFlow.getExpression(isQualifiedExpression()))
      .map(QualifiedExpression.class::cast);

    if (subnetGroupName.isEmpty()) {
      return false;
    }

    QualifiedExpression qualifiedExpression = subnetGroupName.get();
    if (!"ref".equals(qualifiedExpression.name().name())) {
      return false;
    }

    ExpressionFlow qualifierFlow = ExpressionFlow.build(subscriptionContext, qualifiedExpression.qualifier());
    Optional<QualifiedExpression> subnetIds = qualifierFlow.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_rds.CfnDBSubnetGroup")))
      .map(CallExpression.class::cast)
      .flatMap(call -> getArgument(subscriptionContext, call, "subnet_ids"))
      .flatMap(inner -> inner.getExpression(isQualifiedExpression()))
      .map(QualifiedExpression.class::cast);

    if (subnetIds.isEmpty()) {
      return false;
    }

    if (!"subnet_ids".equals(subnetIds.get().name().name())) {
      return false;
    }

    ExpressionFlow subnetIdVpcQualifier = ExpressionFlow.build(subscriptionContext, subnetIds.get().qualifier());
    Optional<CallExpression> vpcSelectSubnetCall = subnetIdVpcQualifier
      .getExpression(isCallExpression())
      .map(CallExpression.class::cast);

    if (vpcSelectSubnetCall.isPresent()) {
      return getArgument(subscriptionContext, vpcSelectSubnetCall.get(), "subnet_type")
        .map(subnetType -> subnetType.hasExpression(isFqn("aws_cdk.aws_ec2.SubnetType.PUBLIC").negate()))
        .orElse(false);
    }

    return false;
  }

  private static Predicate<Expression> isQualifiedExpression() {
    return expression -> expression.is(Tree.Kind.QUALIFIED_EXPR);
  }

  private static void checkDatabaseInstance(SubscriptionContext ctx, CallExpression call) {
    Optional<ExpressionFlow> vpcSubnets = getArgument(ctx, call, "vpc_subnets");

    Optional<ExpressionFlow> subnetType = vpcSubnets
      .flatMap(flow -> flow.getExpression(isCallExpression().and(isFqn("aws_cdk.aws_ec2.SubnetSelection"))))
      .map(CallExpression.class::cast)
      .flatMap(subnetSelection -> getArgument(ctx, subnetSelection, "subnet_type"));

    if (subnetType.filter(isSafeSubnetSelection()).isPresent()) {
      return;
    }

    // Raise issue if
    //  - vpcSubnets is public and publicly_accessible is true
    //  - vpcSubnets is unknown and publicly_accessible is true
    //  - vpcSubnets is public and publicly_accessible is not set
    getArgument(ctx, call, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresentOrElse(access -> access.addIssueIf(isTrue(), MESSAGE),
      () -> subnetType.filter(isPublicSubnetSelection()).ifPresent(subnets -> subnets.addIssue(MESSAGE)));
  }

  /**
   * The `vpc_subnets` is safe if it is an `SubnetSelection` object with `subnet_type` of type `SubnetType` and not `PUBLIC`
   */
  private static Predicate<ExpressionFlow> isSafeSubnetSelection() {
    return subnetType -> SAFE_SUBNET_TYPES.stream()
      .anyMatch(safeType -> subnetType.hasExpression(isFqn("aws_cdk.aws_ec2.SubnetType." + safeType)));
  }

  private static Predicate<ExpressionFlow> isPublicSubnetSelection() {
    return subnetType -> subnetType.hasExpression(isFqn("aws_cdk.aws_ec2.SubnetType.PUBLIC"));
  }

}
