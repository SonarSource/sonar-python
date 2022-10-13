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

@Rule(key = "S6329")
public class PublicNetworkAccessToCloudResourcesCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure allowing public network access is safe here.";
  private static final Set<String> SAFE_SUBNET_TYPES = Set.of("ISOLATED", "PRIVATE_ISOLATED", "PRIVATE", "PRIVATE_WITH_NAT");
  public static final String PUBLICLY_ACCESSIBLE_ARG_NAME = "publicly_accessible";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_dms.CfnReplicationInstance", (subscriptionContext, callExpression) ->
      CdkUtils.getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresentOrElse(
        argument -> argument.addIssueIf(CdkPredicate.isTrue(), MESSAGE),
        () -> subscriptionContext.addIssue(callExpression, MESSAGE)
      )
    );

    checkFqn("aws_cdk.aws_rds.DatabaseInstance", PublicNetworkAccessToCloudResourcesCheck::checkDatabaseInstance);

    checkFqn("aws_cdk.aws_rds.CfnDBInstance", (subscriptionContext, callExpression) ->
      CdkUtils.getArgument(subscriptionContext, callExpression, PUBLICLY_ACCESSIBLE_ARG_NAME).ifPresent(
        argument -> argument.addIssueIf(CdkPredicate.isTrue(), MESSAGE)
      )
    );
  }

  private static void checkDatabaseInstance(SubscriptionContext ctx, CallExpression call) {
    Optional<CdkUtils.ExpressionFlow> vpcSubnets = CdkUtils.getArgument(ctx, call, "vpc_subnets");

    Optional<CdkUtils.ExpressionFlow> subnetType = vpcSubnets.flatMap(flow ->
      CdkUtils.getCall(flow.getLast(), "aws_cdk.aws_ec2.SubnetSelection")
        .flatMap(subnetSelection -> CdkUtils.getArgument(flow.ctx(), subnetSelection, "subnet_type")));

    if (subnetType.filter(isSafeSubnetSelection()).isPresent()) {
      return;
    }

    // Raise issue if
    //  - vpcSubnets is public and publicly_accessible is true
    //  - vpcSubnets is unknown and publicly_accessible is true
    //  - vpcSubnets is public and publicly_accessible is not set
    Optional<CdkUtils.ExpressionFlow> publiclyAccessible = CdkUtils.getArgument(ctx, call, PUBLICLY_ACCESSIBLE_ARG_NAME);
    publiclyAccessible.ifPresentOrElse(access -> access.addIssueIf(CdkPredicate.isTrue(), MESSAGE),
      () -> subnetType.filter(isPublicSubnetSelection()).ifPresent(subnets -> subnets.addIssue(MESSAGE)));
  }

  /**
   * The `vpc_subnets` is safe if it is an `SubnetSelection` object with `subnet_type` of type `SubnetType` and not `PUBLIC`
   */
  private static Predicate<CdkUtils.ExpressionFlow> isSafeSubnetSelection() {
    return subnetType -> SAFE_SUBNET_TYPES.stream()
      .anyMatch(safeType -> subnetType.hasExpression(CdkPredicate.isFqn("aws_cdk.aws_ec2.SubnetType." + safeType)));
  }

  private static Predicate<CdkUtils.ExpressionFlow> isPublicSubnetSelection() {
    return subnetType -> subnetType.hasExpression(CdkPredicate.isFqn("aws_cdk.aws_ec2.SubnetType.PUBLIC"));
  }

}
