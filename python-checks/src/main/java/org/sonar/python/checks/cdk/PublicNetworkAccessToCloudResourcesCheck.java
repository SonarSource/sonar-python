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

import org.sonar.check.Rule;

@Rule(key = "S6329")
public class PublicNetworkAccessToCloudResourcesCheck extends AbstractCdkResourceCheck {
  private static final String ERROR_MESSAGE = "Make sure allowing public network access is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_dms.CfnReplicationInstance", (subscriptionContext, callExpression) ->
      CdkUtils.getArgument(subscriptionContext, callExpression, "publicly_accessible").ifPresentOrElse(
        argument -> argument.addIssueIf(CdkPredicate.isTrue(), ERROR_MESSAGE),
        () -> subscriptionContext.addIssue(callExpression, ERROR_MESSAGE)
      )
    );

    checkFqn("aws_cdk.aws_rds.DatabaseInstance", (ctx, call) -> {});
  }
}
