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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

@Rule(key = "S6330")
public class UnencryptedSqsQueueCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    new AwsSqsQueueResourceCheck().initialize(context);
    new AwsSqsCfnQueueResourceCheck().initialize(context);
  }

  static class AwsSqsQueueResourceCheck extends AbstractCdkResourceCheck {
    private static final String UNENCRYPTED_MESSAGE = "Setting \"encryption\" to \"QueueEncryption.UNENCRYPTED\" disables SQS queues encryption. Make sure it is safe here.";
    private static final String NONE_MESSAGE = "Setting \"encryption\" to \"None\" disables SQS queues encryption. Make sure it is safe here.";
    private static final String OMITTING_MESSAGE = "Omitting \"encryption\" disables SQS queues encryption. Make sure it is safe here.";

    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_sqs.Queue";
    }

    @Override
    protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {
      getArgument(ctx, resourceConstructor, "encryption").ifPresentOrElse(
        argumentTrace -> {
          argumentTrace.addIssueIf(argTrace -> AbstractCdkResourceCheck.isFqnValue(argTrace, "aws_cdk.aws_sqs.QueueEncryption.UNENCRYPTED"), UNENCRYPTED_MESSAGE);
          argumentTrace.addIssueIf(AbstractCdkResourceCheck::isNone, NONE_MESSAGE);
        },
        () -> ctx.addIssue(resourceConstructor.callee(), OMITTING_MESSAGE)
      );
    }
  }

  static class AwsSqsCfnQueueResourceCheck extends AbstractCdkResourceCheck {
    private static final String NONE_MESSAGE = "Setting \"kms_master_key_id\" to \"None\" disables SQS queues encryption. Make sure it is safe here.";
    private static final String OMITTING_MESSAGE = "Omitting \"kms_master_key_id\" disables SQS queues encryption. Make sure it is safe here.";

    @Override
    protected String resourceFqn() {
      return "aws_cdk.aws_sqs.CfnQueue";
    }

    @Override
    protected void visitResourceConstructor(SubscriptionContext ctx, CallExpression resourceConstructor) {
      getArgument(ctx, resourceConstructor, "kms_master_key_id").ifPresentOrElse(
        argumentTrace -> argumentTrace.addIssueIf(AbstractCdkResourceCheck::isNone, NONE_MESSAGE),
        () -> ctx.addIssue(resourceConstructor.callee(), OMITTING_MESSAGE)
      );
    }
  }
}
