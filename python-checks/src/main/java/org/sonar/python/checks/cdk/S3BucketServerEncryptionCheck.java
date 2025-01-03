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

import java.util.function.BiConsumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

@Rule(key = "S6245")
public class S3BucketServerEncryptionCheck extends AbstractS3BucketCheck {

  private static final String S3_BUCKET_UNENCRYPTED_FQN = "aws_cdk.aws_s3.BucketEncryption.UNENCRYPTED";
  public static final String MESSAGE = "Objects in the bucket are not encrypted. Make sure it is safe here.";
  public static final String OMITTING_MESSAGE = "Omitting 'encryption' disables server-side encryption. Make sure it is safe here.";

  @Override
  BiConsumer<SubscriptionContext, CallExpression> visitBucketConstructor() {
    return (ctx, bucket) -> getArgument(ctx, bucket, "encryption")
      .ifPresentOrElse(
        encryption -> encryption.addIssueIf(isFqn(S3_BUCKET_UNENCRYPTED_FQN), MESSAGE),
        () -> checkEncryptionKey(ctx, bucket));
  }

  private static void checkEncryptionKey(SubscriptionContext ctx, CallExpression bucket) {
    if (getArgument(ctx, bucket, "encryption_key").isEmpty()) {
      ctx.addIssue(bucket.callee(), OMITTING_MESSAGE);
    }
  }
}
