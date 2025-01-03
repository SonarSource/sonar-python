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
package org.sonar.python.checks.cdk;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

import static org.sonar.python.checks.cdk.CdkPredicate.isFalse;
import static org.sonar.python.checks.cdk.CdkPredicate.isNone;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6330")
public class UnencryptedSqsQueueCheck extends AbstractCdkResourceCheck {
  private static final String SQS_MANAGED_DISABLED_MESSAGE = "Setting \"sqs_managed_sse_enabled\" to \"false\" disables SQS queues encryption. Make sure it is safe here.";
  private static final String CFN_NONE_MESSAGE = "Setting \"kms_master_key_id\" to \"None\" disables SQS queues encryption. Make sure it is safe here.";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_sqs.CfnQueue", this::checkCfnQueue);
  }

  protected void checkCfnQueue(SubscriptionContext ctx, CallExpression resourceConstructor) {
    getArgument(ctx, resourceConstructor, "sqs_managed_sse_enabled")
        .ifPresent(flow -> flow.addIssueIf(isFalse(), SQS_MANAGED_DISABLED_MESSAGE));
  }
}
