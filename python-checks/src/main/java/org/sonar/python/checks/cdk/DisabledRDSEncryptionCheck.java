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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkPredicate.isNone;
import static org.sonar.python.checks.cdk.CdkPredicate.startsWith;

@Rule(key = "S6303")
public class DisabledRDSEncryptionCheck extends AbstractCdkResourceCheck {
  private static final String UNENCRYPTED_MESSAGE = "Make sure that using unencrypted databases is safe here.";
  private static final String ARG_ENCRYPTED = "storage_encrypted";
  private static final String ARG_ENCRYPTION_KEY = "storage_encryption_key";
  private static final String DB_OMITTING_MESSAGE = "Omitting \""+ARG_ENCRYPTED+"\" and \""+ARG_ENCRYPTION_KEY+"\" disables RDS encryption. Make sure it is safe here.";
  private static final String CFNDB_OMITTING_MESSAGE = "Omitting \""+ARG_ENCRYPTED+"\" disables RDS encryption. Make sure it is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqns(List.of("aws_cdk.aws_rds.DatabaseCluster", "aws_cdk.aws_rds.DatabaseInstance", "aws_cdk.aws_rds.CfnDBCluster"),
      this::checkDatabaseArguments);

    checkFqn("aws_cdk.aws_rds.CfnDBInstance", (subscriptionContext, callExpression) -> {
      if (!isEngineAurora(subscriptionContext, callExpression)) {
        checkCfnDatabaseArguments(subscriptionContext, callExpression);
      }
    });
  }

  protected void checkDatabaseArguments(SubscriptionContext ctx, CallExpression resourceConstructor) {
    Optional<CdkUtils.ExpressionFlow> argEncrypted = CdkUtils.getArgument(ctx, resourceConstructor, ARG_ENCRYPTED);
    Optional<CdkUtils.ExpressionFlow> argEncryptionKey = CdkUtils.getArgument(ctx, resourceConstructor, ARG_ENCRYPTION_KEY);

    if (argEncrypted.isEmpty() && argEncryptionKey.isEmpty()) {
      ctx.addIssue(resourceConstructor.callee(), DB_OMITTING_MESSAGE);
    } else if (argEncrypted.isEmpty()) {
      argEncryptionKey.get().addIssueIf(isNone(), UNENCRYPTED_MESSAGE);
    } else if (argEncryptionKey.isEmpty()) {
      argEncrypted.get().addIssueIf(isFalse(), UNENCRYPTED_MESSAGE);
    } else {
      if (argEncryptionKey.get().hasExpression(isNone())
        && argEncrypted.get().hasExpression(isFalse())) {
        argEncrypted.get().addIssue(UNENCRYPTED_MESSAGE);
      }
    }
  }

  protected void checkCfnDatabaseArguments(SubscriptionContext ctx, CallExpression resourceConstructor) {
    CdkUtils.getArgument(ctx, resourceConstructor, ARG_ENCRYPTED).ifPresentOrElse(
      flow -> flow.addIssueIf(isFalse(), UNENCRYPTED_MESSAGE),
      () -> ctx.addIssue(resourceConstructor.callee(), CFNDB_OMITTING_MESSAGE)
    );
  }

  protected boolean isEngineAurora(SubscriptionContext ctx, CallExpression resourceConstructor) {
    return CdkUtils.getArgument(ctx, resourceConstructor, "engine")
      .filter(flow -> flow.hasExpression(startsWith("aurora"))).isPresent();
  }
}

